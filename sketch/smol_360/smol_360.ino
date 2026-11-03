// SmolFem-360M (smolFem by Ambika, a fine-tune of SmolLM2-360M, 4-bit) running on the ESP32-S3 N16R8, streamed layer by layer from the SD card.
// Question in over Bluetooth LE (phone terminal app) or USB serial; answer on the 2" screen, in Bluetooth and on serial.
// Wiring: see docs/HOWTO.md.  Files on the SD card: /SMOL360/model.bin and bpe.bin (made by scripts/export_sd360.py); head.bin goes into the flash partition "head".
// Port of sd-stream/engine/engine.c: batch of tokens per pass for the prompt, one token per pass for the answer, bfloat16 K/V cache.
#include <Arduino.h>
#include <SPI.h>
#include <SD.h>
#include <FS.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ST7789.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include "esp_heap_caps.h"
#include "esp_partition.h"
#include <Wire.h>
#include <math.h>
#include <stdarg.h>
#include <string.h>
#include "dsps_dotprod.h"    // esp-dsp (ships inside the ESP32 Arduino core): fast dot products on the S3
#include "draft_data.h"      // token ids of the training answers, used to guess ahead (see "guessing" below)

// ------------------------------------------------------------------ pins
#define SD_CS   10
#define SD_MOSI 11
#define SD_SCK  12
#define SD_MISO 13
#define TFT_SCK  39
#define TFT_MOSI 40
#define TFT_CS   41
#define TFT_DC   42
#define TFT_RST  2
// screen BL wire goes to 3V3 (since 3 Nov): the backlight is always on at full brightness and GPIO 14 is free

#define MAXPOS 96            // prompt + answer must fit (K/V cache in PSRAM: 2.2 MB)
#define MAXB   32            // tokens per pass: prompt tokens plus guesses, or the token plus its guesses
#define DEMO_TOKENS 30       // default answer length; "/full question" lifts it
#define FULL_TOKENS 60       // longest answer with /full (about 45 words is the longest dataset answer); a plain question stops at DEMO_TOKENS
#define MAX_ANSWER_S 0       // optional time limit per answer in seconds (0 = off), cut off after the round that passes it
#define USE_BLE 1           // Bluetooth on. Its radio bursts add current peaks: if the board browns out, set this to 0 and use USB serial
#define BLE_NAME "SmolFem-360M"
#define NUS_SERVICE "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
#define NUS_RX      "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
#define NUS_TX      "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

SPIClass sdspi(FSPI);
SPIClass lcdspi(HSPI);
Adafruit_ST7789 tft(&lcdspi, TFT_CS, TFT_DC, TFT_RST);

// ------------------------------------------------------------------ model description (from model.bin header)
static int L, D, F, H, KV, HD, KVD, VOC, GROUP;
static float rope_theta, eps;
static uint32_t layers_off, layer_stride, embed_off, embed_stride;
static File gModel;

static uint16_t *norms16;                  // (2L+1) x D, fp16
static uint16_t *kcache, *vcache;          // bfloat16 [L][MAXPOS][KVD]
static int RB;                             // rows per block (the SD file stores every tensor in blocks of RB rows)
static float *X, *Hn, *Q, *ATT, *TMP, *G, *U, *Kt, *Vt;      // batch activations, PSRAM
static float *gAllHid;                     // MAXB x D, the hidden state of every token of the last batch (checks the guesses)
static float cosT[MAXB][32], sinT[MAXB][32];
static float scoresBuf[MAXPOS];

// output head (allowed words)
static uint8_t *headbuf;
static int N_allowed, eos_id;
static uint16_t *head_ids;
static uint16_t *head_scale16;
static int8_t *head_rows;
static uint32_t *head_offs;
static uint8_t *head_blob;

// tokenizer
static uint8_t *bpebuf;
struct Merge { uint16_t a, b, c, rank; };
static uint16_t *byte_to_id;
static Merge *merges;
static uint32_t n_slots;

static bool g_dbg0 = false;

// ------------------------------------------------------------------ small helpers
static float h2f(uint16_t h) {
  uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023, f;
  if (e == 0) {
    if (m == 0) f = s << 31;
    else { e = 127 - 14; while (!(m & 1024)) { m <<= 1; e--; } m &= 1023; f = (s << 31) | (e << 23) | (m << 13); }
  } else if (e == 31) f = (s << 31) | 0x7F800000 | (m << 13);
  else f = (s << 31) | ((e + 112) << 23) | (m << 13);
  float out; memcpy(&out, &f, 4); return out;
}
static inline float bf2f(uint16_t v) { uint32_t b = ((uint32_t)v) << 16; float f; memcpy(&f, &b, 4); return f; }
static inline uint16_t f2bf(float f) { uint32_t b; memcpy(&b, &f, 4); b += 0x7FFFu + ((b >> 16) & 1u); return (uint16_t)(b >> 16); }
static inline float bfround(float f) { return bf2f(f2bf(f)); }

static void *psalloc(size_t n, const char *what) {
  void *p = heap_caps_malloc(n, MALLOC_CAP_SPIRAM);
  if (!p) { Serial.printf("FATAL: PSRAM allocation failed for %s (%u bytes)\n", what, (unsigned)n); while (1) delay(1000); }
  return p;
}

static void ui_tick();
static void ui_error(const char *m);

static bool sd_read_once(uint32_t off, void *dst, size_t n) {
  if (!gModel.seek(off)) return false;
  size_t done = 0;
  while (done < n) {
    size_t k = n - done > 32768 ? 32768 : n - done;
    int g = gModel.read((uint8_t *)dst + done, k);
    if (g <= 0) return false;
    done += g;
    ui_tick();
  }
  return true;
}
// a failed read is retried (and the card re-mounted) until it works, so a loose wire does not stop the run for good
static bool sd_read(uint32_t off, void *dst, size_t n) {
  for (int attempt = 0;; attempt++) {
    if (sd_read_once(off, dst, n)) return true;
    Serial.printf("SD read failed (offset %u, attempt %d), re-mounting the card\n", (unsigned)off, attempt + 1);
    ui_error("SD read error: check reader power (XH board) and wires");
    gModel.close(); SD.end(); delay(500);
    sdspi.end(); sdspi.begin(SD_SCK, SD_MISO, SD_MOSI, SD_CS);
    if (SD.begin(SD_CS, sdspi, 20000000)) gModel = SD.open("/SMOL360/model.bin", FILE_READ);
    delay(attempt < 5 ? 500 : 3000);
  }
}

// ------------------------------------------------------------------ screen
#define BLACK ST77XX_BLACK
#define WHITE ST77XX_WHITE
static int ansX = 4, ansY = 58;
static uint32_t gStart = 0;
static int gWords = 0;

static void ui_title() {
  tft.fillRect(0, 0, 320, 16, 0x001F);
  tft.setTextSize(1); tft.setTextColor(WHITE);
  tft.setCursor(4, 4); tft.print("SmolFem 360M  |  4-bit  |  streamed from SD");
}
static void ui_clear_answer() { tft.fillRect(0, 56, 320, 138, BLACK); ansX = 4; ansY = 58; }
static void ui_question(const char *q) {
  tft.fillRect(0, 18, 320, 36, BLACK);
  tft.setTextSize(1); tft.setTextColor(ST77XX_YELLOW);
  tft.setCursor(4, 20); tft.print("Q: ");
  int shown = 0;
  for (const char *p = q; *p && shown < 150; p++, shown++) tft.print((*p >= 32 && *p < 127) ? *p : '?');
}
static void ui_answer_char(char c) {
  if (c == '\n' || c == '\r') { ansX = 4; ansY += 17; if (ansY > 58 + 7 * 17) ui_clear_answer(); return; }
  if (ansX > 320 - 12 - 4) { ansX = 4; ansY += 17; }
  if (ansY > 58 + 7 * 17) ui_clear_answer();
  if (c < 32 || c > 126) c = '?';
  tft.drawChar(ansX, ansY, c, ST77XX_GREEN, BLACK, 2);
  ansX += 12;
}
static void ui_status(const char *a, float frac, const char *b) {
  tft.fillRect(0, 196, 320, 44, BLACK);
  tft.setTextSize(1); tft.setTextColor(WHITE);
  tft.setCursor(4, 198); tft.print(a);
  tft.drawRect(4, 210, 312, 10, 0x7BEF);
  if (frac > 0) tft.fillRect(5, 211, (int)(310 * min(frac, 1.0f)), 8, ST77XX_CYAN);
  tft.setCursor(4, 224); tft.print(b);
}
static void fmtTime(char *dst, size_t n, uint32_t ms) {
  uint32_t s = ms / 1000; snprintf(dst, n, "%02u:%02u:%02u", (unsigned)(s / 3600), (unsigned)((s / 60) % 60), (unsigned)(s % 60));
}
// heartbeat: redraw the elapsed time once a second even inside a long step, so a frozen number means a real freeze
static bool ina_ok = false;
static volatile float ina_v = 0, ina_ma = 0, ina_mwh = 0, ina_peak_ma = 0;   // load voltage (V), current (mA), energy since boot (mWh)
static float gE0 = 0;   // energy meter reading (mWh) when the current question arrived
static uint32_t gLastTick = 0;
static void ui_tick() {
  if (gStart == 0) return;
  uint32_t now = millis();
  if (now - gLastTick < 1000) return;
  gLastTick = now;
  char tm[16], b[64]; fmtTime(tm, sizeof tm, now - gStart);
  if (ina_ok) snprintf(b, sizeof b, "words %d | %s | %.1f mWh | %.0f mW *", gWords, tm, ina_mwh - gE0, (float)ina_v * (float)ina_ma);
  else snprintf(b, sizeof b, "words %d  |  elapsed %s  *", gWords, tm);
  tft.fillRect(0, 222, 320, 18, BLACK); tft.setTextSize(1); tft.setTextColor(WHITE); tft.setCursor(4, 224); tft.print(b);
}
static void ui_error(const char *m) {
  tft.fillRect(0, 196, 320, 44, BLACK); tft.setTextSize(1); tft.setTextColor(ST77XX_RED); tft.setCursor(4, 200); tft.print(m);
}

// ------------------------------------------------------------------ Bluetooth LE (Nordic UART service)
static BLECharacteristic *ble_tx = NULL;
static volatile bool ble_connected = false;
static char ble_line[200];
static volatile int ble_len = 0;
static volatile bool ble_ready = false;
static volatile uint32_t ble_last_ms = 0;

class SrvCb : public BLEServerCallbacks {
  // This core uses the NimBLE stack: the callbacks get the connection description and no disconnect reason.
  void onConnect(BLEServer *s, ble_gap_conn_desc *d) override {
    ble_connected = true;
    Serial.printf("\n[BLE connected at %lu s, handle %u]\n", (unsigned long)(millis() / 1000), (unsigned)d->conn_handle);
    // ask for a 30 to 50 ms interval, no skipped events and a 6 s supervision timeout: the board is busy for stretches and must not be dropped for a late reply
    s->updateConnParams(d->conn_handle, 24, 40, 0, 600);
  }
  void onDisconnect(BLEServer *, ble_gap_conn_desc *d) override {
    ble_connected = false;
    Serial.printf("\n[BLE disconnected at %lu s, handle %u]\n", (unsigned long)(millis() / 1000), (unsigned)d->conn_handle);
    BLEDevice::startAdvertising();
  }
};
class RxCb : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *c) override {
    String v = c->getValue();
    for (size_t i = 0; i < v.length(); i++) {
      char ch = v[i];
      if (ch == '\n' || ch == '\r') { if (ble_len > 0) ble_ready = true; }
      else if (ble_len < (int)sizeof(ble_line) - 1) ble_line[ble_len++] = ch;
    }
    ble_last_ms = millis();
  }
};
static void ble_send(const char *s, size_t n) {
  if (!ble_tx || !ble_connected) return;
  size_t off = 0;
  while (off < n) {
    size_t k = n - off > 20 ? 20 : n - off;
    ble_tx->setValue((uint8_t *)(s + off), k); ble_tx->notify();
    off += k; delay(8);
  }
}
static void ble_begin() {
  BLEDevice::init(BLE_NAME);
  // Low radio power: the board's 3.3 V supply is marginal, and the transmit bursts on top of both cores running flat out reset it
  // (brownout / watchdog resets with Bluetooth on and big batches). A phone next to the board does not need full power.
  esp_ble_tx_power_set(ESP_BLE_PWR_TYPE_DEFAULT, ESP_PWR_LVL_N12);
  esp_ble_tx_power_set(ESP_BLE_PWR_TYPE_ADV, ESP_PWR_LVL_N12);
  BLEServer *srv = BLEDevice::createServer(); srv->setCallbacks(new SrvCb());
  BLEService *svc = srv->createService(NUS_SERVICE);
  ble_tx = svc->createCharacteristic(NUS_TX, BLECharacteristic::PROPERTY_NOTIFY); ble_tx->addDescriptor(new BLE2902());
  BLECharacteristic *rx = svc->createCharacteristic(NUS_RX, BLECharacteristic::PROPERTY_WRITE | BLECharacteristic::PROPERTY_WRITE_NR);
  rx->setCallbacks(new RxCb());
  svc->start();
  BLEAdvertising *adv = BLEDevice::getAdvertising(); adv->addServiceUUID(NUS_SERVICE); adv->setScanResponse(true);
  adv->setMinInterval(1600); adv->setMaxInterval(1600);           // one advertisement per second (units of 0.625 ms): fewer radio bursts, the phone finds it a little slower
  BLEDevice::startAdvertising();
}
// text out to serial and phone
static void out_text(const char *fmt, ...) {
  char buf[200]; va_list ap; va_start(ap, fmt); int n = vsnprintf(buf, sizeof buf, fmt, ap); va_end(ap);
  if (n < 0) return; if (n > (int)sizeof buf - 1) n = sizeof buf - 1;
  Serial.write((const uint8_t *)buf, n); ble_send(buf, n);
}

// ------------------------------------------------------------------ tokenizer (same as the PC engine)
static uint32_t mhash(uint32_t a, uint32_t b) { return (uint32_t)(a * 0x9E3779B1u + b * 0x85EBCA6Bu) >> 16; }
static const Merge *find_merge(uint16_t a, uint16_t b) {
  uint32_t s = mhash(a, b) & (n_slots - 1);
  while (merges[s].a != 0xFFFF) { if (merges[s].a == a && merges[s].b == b) return &merges[s]; s = (s + 1) & (n_slots - 1); }
  return NULL;
}
static int bpe_word(const uint8_t *w, int n, uint16_t *out) {
  uint16_t seq[128]; int m = 0;
  for (int i = 0; i < n && m < 128; i++) seq[m++] = byte_to_id[w[i]];
  for (;;) {
    int best = -1, best_rank = 1 << 30; uint16_t best_c = 0, ba = 0, bb = 0;
    for (int i = 0; i + 1 < m; i++) {
      const Merge *mg = find_merge(seq[i], seq[i + 1]);
      if (mg && mg->rank < best_rank) { best_rank = mg->rank; best = i; best_c = mg->c; ba = seq[i]; bb = seq[i + 1]; }
    }
    if (best < 0) break;
    int k = 0;
    for (int i = 0; i < m;) { if (i + 1 < m && seq[i] == ba && seq[i + 1] == bb) { seq[k++] = best_c; i += 2; } else seq[k++] = seq[i++]; }
    m = k;
  }
  memcpy(out, seq, (size_t)m * 2);
  return m;
}
enum { CS = 0, CL = 1, CN = 2, CO = 3 };
static int cp_at(const unsigned char *s, int i, int n, int *len) {
  unsigned c0 = s[i];
  if (c0 < 0x80) { *len = 1; return (int)c0; }
  if ((c0 >> 5) == 6 && i + 1 < n) { *len = 2; return (int)(((c0 & 31) << 6) | (s[i + 1] & 63)); }
  if ((c0 >> 4) == 14 && i + 2 < n) { *len = 3; return (int)(((c0 & 15) << 12) | ((s[i + 1] & 63) << 6) | (s[i + 2] & 63)); }
  if ((c0 >> 3) == 30 && i + 3 < n) { *len = 4; return (int)(((c0 & 7) << 18) | ((s[i + 1] & 63) << 12) | ((s[i + 2] & 63) << 6) | (s[i + 3] & 63)); }
  *len = 1; return 0xFFFD;
}
static int cls_of(int cp) {
  if (cp == ' ' || cp == '\n' || cp == '\t' || cp == '\r' || cp == 0x0B || cp == 0x0C || cp == 0xA0 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x3000) return CS;
  if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z')) return CL;
  if (cp >= '0' && cp <= '9') return CN;
  if (cp == 0xAA || cp == 0xB5 || cp == 0xBA) return CL;
  if (cp >= 0xC0 && cp <= 0x24F && cp != 0xD7 && cp != 0xF7) return CL;
  if ((cp >= 0x370 && cp <= 0x3FF) || (cp >= 0x400 && cp <= 0x52F) || (cp >= 0x1E00 && cp <= 0x1FFF)) return CL;
  if (cp >= 0x3040 && cp <= 0x9FFF) return CL;
  return CO;
}
static int cls_at(const char *s, int i, int n, int *len) { return cls_of(cp_at((const unsigned char *)s, i, n, len)); }
static int encode(const char *s, uint16_t *ids, int max_ids) {
  int n = (int)strlen(s), i = 0, total = 0, len;
  while (i < n) {
    int j = i;
    if (s[i] == '\'' && i + 1 < n && (s[i + 1] == 's' || s[i + 1] == 't' || s[i + 1] == 'm' || s[i + 1] == 'd')) j = i + 2;
    else if (s[i] == '\'' && i + 2 < n && ((s[i + 1] == 'r' && s[i + 2] == 'e') || (s[i + 1] == 'v' && s[i + 2] == 'e') || (s[i + 1] == 'l' && s[i + 2] == 'l'))) j = i + 3;
    else {
      int k = i;
      if (s[k] == ' ' && k + 1 < n) { int nl; if (cls_at(s, k + 1, n, &nl) != CS) k++; }
      int cl = cls_at(s, k, n, &len);
      if (cl == CL || cl == CN || cl == CO) {
        while (k < n) { int l2, c2 = cls_at(s, k, n, &l2); if (c2 != cl) break; k += l2; }
        j = k;
      } else {
        int r = i, last = i;
        while (r < n) { int l2, c2 = cls_at(s, r, n, &l2); if (c2 != CS) break; last = r; r += l2; }
        j = (r < n && last > i) ? last : r;
      }
    }
    if (j <= i) j = i + 1;
    if (total + 64 > max_ids) break;
    total += bpe_word((const uint8_t *)s + i, j - i, ids + total);
    i = j;
  }
  return total;
}

// ------------------------------------------------------------------ math
static void rms_row(float *o, const float *x, const uint16_t *w16, int n) {
  float ss = 0; for (int i = 0; i < n; i++) ss += x[i] * x[i];
  float inv = 1.0f / sqrtf(ss / n + eps);
  for (int i = 0; i < n; i++) o[i] = x[i] * inv * h2f(w16[i]);
}

// Y[t][o] = sum_j W[o][j] * X[t][j], W stored as 4-bit codes with an fp16 scale and minimum per group of 64
// Maths for one BLOCK of rows. The weights of a block sit in buf as [codes of nr rows][fp16 scales][fp16 mins] (4-bit codes, groups of 64).
// Unpack one group of 64 weights to floats once, then esp-dsp's hand-tuned dot product (ESP32-S3 assembly) does the 64 multiply-adds per token.
// The input rows are copied in chunks into internal RAM (xi): a PSRAM read that misses the cache is slow.
static float lut[512];                 // lut[2*b] = low nibble, lut[2*b+1] = high nibble
static void init_lut() { for (int b = 0; b < 256; b++) { lut[2 * b] = (float)(b & 15); lut[2 * b + 1] = (float)(b >> 4); } }
#define XI_BYTES 40960
#define MAXGN 40                       // groups of 64 per input row (2560 / 64)
static float *xi;                      // internal RAM, XI_BYTES, allocated in setup()

// Y[t * ystride + o] = sum_j W[o][j] * X[t][j] for the nr rows of this block
static void matmul_blk(const uint8_t *buf, int nr, int in, const float *Xin, int n, float *Y, int ystride) {
  const int Gn = in / 64;
  const uint8_t *codes = buf;
  const uint16_t *sc = (const uint16_t *)(buf + (size_t)nr * in / 2);
  const uint16_t *mn = sc + (size_t)nr * Gn;
  static float xs[MAXB][MAXGN];
  alignas(16) static float w[64];
  const int tb = XI_BYTES / (in * 4);                       // tokens per chunk: 17 for in = 960, 4 for in = 2560
  for (int t0 = 0; t0 < n; t0 += tb) {
    const int nt = min(tb, n - t0);
    memcpy(xi, Xin + (size_t)t0 * in, (size_t)nt * in * 4);
    for (int t = 0; t < nt; t++)
      for (int g = 0; g < Gn; g++) { float s = 0; const float *x = xi + (size_t)t * in + g * 64; for (int j = 0; j < 64; j++) s += x[j]; xs[t][g] = s; }
    for (int o = 0; o < nr; o++) {
      if ((o & 63) == 0) ui_tick();
      const uint8_t *row = codes + (size_t)o * in / 2;
      float acc[MAXB];
      for (int t = 0; t < nt; t++) acc[t] = 0;
      for (int g = 0; g < Gn; g++) {
        const uint8_t *p = row + g * 32;
        for (int j = 0; j < 32; j++) { const float *a = lut + 2 * p[j]; w[2 * j] = a[0]; w[2 * j + 1] = a[1]; }
        float s = h2f(sc[(size_t)o * Gn + g]), m = h2f(mn[(size_t)o * Gn + g]);
        for (int t = 0; t < nt; t++) {
          float d; dsps_dotprod_f32(w, xi + (size_t)t * in + g * 64, &d, 64);
          acc[t] += s * d + m * xs[t][g];
        }
      }
      for (int t = 0; t < nt; t++) Y[(size_t)(t0 + t) * ystride + o] = acc[t];
    }
  }
}

static const char *TNAME[7] = {"q", "k", "v", "o", "gate", "up", "down"};
static uint32_t gLastUi = 0;
static const char *gPhase = "";
static float gFrac0 = 0, gFrac1 = 1;   // progress window of this pass inside the bar

// ---- reading ahead: a task on the other core reads the NEXT block from the SD card into the second buffer while this core
// is busy multiplying the current one. A layer is a list of UPL blocks (7 tensors, RB rows each); block number i = layer * UPL + unit
// uses buffer wb[i & 1]. semFree[s] = buffer s may be filled, semReady[s] = buffer s holds its block.
// Only the reader task touches the SD card during a pass.
static uint8_t *wb[2];
static SemaphoreHandle_t semFree[2], semReady[2], semGo;
static volatile int pf_layers = 0;                 // how many layers the pass in progress has
static int tOut[7], tIn[7], tBlocks[7], tUnit0[7]; // shape of each tensor, its number of blocks, index of its first block inside a layer
static int UPL;                                    // blocks per layer
static uint32_t uOff[64], uBytes[64]; static int uRows[64];

static bool sd_read_bg(uint32_t off, void *dst, size_t n) {
  for (int attempt = 0;; attempt++) {
    bool ok = gModel.seek(off);
    size_t done = 0;
    while (ok && done < n) {
      int g = gModel.read((uint8_t *)dst + done, n - done > 32768 ? 32768 : n - done);
      if (g <= 0) ok = false; else done += g;
    }
    if (ok) return true;
    Serial.printf("SD read failed (offset %u, attempt %d), re-mounting the card\n", (unsigned)off, attempt + 1);
    gModel.close(); SD.end(); delay(500);
    sdspi.end(); sdspi.begin(SD_SCK, SD_MISO, SD_MOSI, SD_CS);
    if (SD.begin(SD_CS, sdspi, 20000000)) gModel = SD.open("/SMOL360/model.bin", FILE_READ);
    delay(attempt < 5 ? 500 : 3000);
  }
}

static void reader_task(void *) {
  for (;;) {
    xSemaphoreTake(semGo, portMAX_DELAY);
    const int total = pf_layers * UPL;
    for (int i = 0; i < total; i++) {
      const int s = i & 1, l = i / UPL, u = i % UPL;
      xSemaphoreTake(semFree[s], portMAX_DELAY);
      sd_read_bg(layers_off + (uint32_t)l * layer_stride + uOff[u], wb[s], uBytes[u]);
      xSemaphoreGive(semReady[s]);
      if ((u & 7) == 7) vTaskDelay(1);     // the SPI read busy-waits and would starve core 0's idle task (task watchdog)
    }
  }
}

static void start_reader(int nlayers) { pf_layers = nlayers; xSemaphoreGive(semGo); }

static bool init_reader() {
  const int shp[7][2] = {{D, D}, {KVD, D}, {KVD, D}, {D, D}, {F, D}, {F, D}, {D, F}};
  uint32_t o = 0, biggest = 0; int u = 0;
  for (int k = 0; k < 7; k++) {
    tOut[k] = shp[k][0]; tIn[k] = shp[k][1]; tBlocks[k] = (tOut[k] + RB - 1) / RB; tUnit0[k] = u;
    for (int b = 0; b < tBlocks[k]; b++, u++) {
      if (u >= 64) return false;
      int nr = min(RB, tOut[k] - b * RB);
      uRows[u] = nr; uOff[u] = o; uBytes[u] = (uint32_t)((size_t)nr * tIn[k] / 2 + 2 * (size_t)nr * (tIn[k] / 64) * 2);
      o += uBytes[u]; if (uBytes[u] > biggest) biggest = uBytes[u];
    }
  }
  UPL = u;
  const size_t bufb = (biggest + 4095) / 4096 * 4096;
  wb[0] = (uint8_t *)psalloc(bufb, "block buffer 1"); wb[1] = (uint8_t *)psalloc(bufb, "block buffer 2");
  for (int s = 0; s < 2; s++) { semFree[s] = xSemaphoreCreateBinary(); semReady[s] = xSemaphoreCreateBinary(); xSemaphoreGive(semFree[s]); }
  semGo = xSemaphoreCreateBinary();
  xTaskCreatePinnedToCore(reader_task, "sdread", 20480, NULL, 1, NULL, 0);     // the SD/FAT/SPI path needs far more than 8 KB of stack
  Serial.printf("blocks: %d per layer, %u KB per buffer\n", UPL, (unsigned)(bufb / 1024));
  return true;
}

static void tensor_mm(int layer, int idx, int out, int in, uint32_t *off, const float *Xin, int n, float *Y) {
  (void)off; (void)in; (void)out;
  uint32_t tw = 0, tm = 0;
  for (int b = 0; b < tBlocks[idx]; b++) {
    const int u = tUnit0[idx] + b, slot = ((long)layer * UPL + u) & 1;
    uint32_t t0 = millis();
    xSemaphoreTake(semReady[slot], portMAX_DELAY);             // waits only if the reader has not finished this block yet
    uint32_t t1 = millis();
    matmul_blk(wb[slot], uRows[u], tIn[idx], Xin, n, Y + (size_t)b * RB, tOut[idx]);
    xSemaphoreGive(semFree[slot]);                             // the reader may now fill this buffer with the block after next
    tw += t1 - t0; tm += millis() - t1;
  }
  float frac = ((layer * 7 + idx + 1) / (float)(L * 7));
  char a[64], b[96], tmb[16]; fmtTime(tmb, sizeof tmb, millis() - gStart);
  snprintf(a, sizeof a, "%s: layer %d/%d  %s  (wait %.1fs, math %.1fs)", gPhase, layer + 1, L, TNAME[idx], tw / 1000.0f, tm / 1000.0f);
  if (ina_ok) snprintf(b, sizeof b, "words %d | %s | %.1f mWh | %.0f mW", gWords, tmb, ina_mwh - gE0, (float)ina_v * (float)ina_ma);
  else snprintf(b, sizeof b, "words %d  |  elapsed %s", gWords, tmb);
  ui_status(a, frac, b);
  if (idx == 6) Serial.printf("[%s] layer %d/%d done, %lu s since start\n", gPhase, layer + 1, L, (unsigned long)((millis() - gStart) / 1000));
  vTaskDelay(1);
}

static void stats(const char *label, const float *v, int n) {
  double s = 0, ss = 0; for (int i = 0; i < n; i++) { s += v[i]; ss += (double)v[i] * v[i]; }
  out_text("%s: sum=%.5f sumsq=%.5f first4=%.5f %.5f %.5f %.5f\n", label, s, ss, v[0], v[1], v[2], v[3]);
}

// n tokens at positions pos0 .. pos0+n-1; returns the final normalised hidden state of the LAST token in hid
// hidsAll (optional, n x D floats) also receives the normalised hidden state of EVERY token in the batch (needed to check guesses)
static void forward_batch(const int *tokens, int n, int pos0, int nlayers, float *hid, float *hidsAll = nullptr) {
  for (int t = 0; t < n; t++) {
    uint8_t row[1024];
    if (!sd_read(embed_off + (uint32_t)tokens[t] * embed_stride, row, embed_stride)) { Serial.println("embedding read failed"); while (1) delay(1000); }
    uint16_t sh = row[D] | (row[D + 1] << 8); float es = h2f(sh);
    for (int i = 0; i < D; i++) X[t * D + i] = ((int8_t)row[i]) * es;
    for (int i = 0; i < 32; i++) { float ang = (pos0 + t) * powf(rope_theta, -2.0f * i / HD); cosT[t][i] = cosf(ang); sinT[t][i] = sinf(ang); }
  }
  const int rep = H / KV;
  start_reader(nlayers);
  for (int l = 0; l < nlayers; l++) {
    uint32_t off = 0;
    for (int t = 0; t < n; t++) rms_row(Hn + t * D, X + t * D, norms16 + (size_t)(2 * l) * D, D);
    tensor_mm(l, 0, D, D, &off, Hn, n, Q);
    tensor_mm(l, 1, KVD, D, &off, Hn, n, Kt);
    tensor_mm(l, 2, KVD, D, &off, Hn, n, Vt);
    for (int t = 0; t < n; t++) {
      for (int hh = 0; hh < H + KV; hh++) {
        float *v = hh < H ? Q + t * D + hh * HD : Kt + t * KVD + (hh - H) * HD;
        for (int i = 0; i < HD / 2; i++) { float a = v[i], b = v[i + HD / 2]; v[i] = a * cosT[t][i] - b * sinT[t][i]; v[i + HD / 2] = b * cosT[t][i] + a * sinT[t][i]; }
      }
      uint16_t *kc = kcache + ((size_t)l * MAXPOS + pos0 + t) * KVD, *vc = vcache + ((size_t)l * MAXPOS + pos0 + t) * KVD;
      for (int i = 0; i < KVD; i++) { kc[i] = f2bf(Kt[t * KVD + i]); vc[i] = f2bf(Vt[t * KVD + i]); }
    }
    for (int t = 0; t < n; t++) {
      int pos = pos0 + t;
      for (int hh = 0; hh < H; hh++) {
        const float *qh = Q + t * D + hh * HD; int kvh = hh / rep; float mx = -1e30f;
        for (int p = 0; p <= pos; p++) {
          const uint16_t *kp = kcache + ((size_t)l * MAXPOS + p) * KVD + kvh * HD;
          float s = 0; for (int i = 0; i < HD; i++) s += qh[i] * bf2f(kp[i]);
          scoresBuf[p] = s / sqrtf((float)HD); if (scoresBuf[p] > mx) mx = scoresBuf[p];
        }
        float sum = 0; for (int p = 0; p <= pos; p++) { scoresBuf[p] = expf(scoresBuf[p] - mx); sum += scoresBuf[p]; }
        float *oh = ATT + t * D + hh * HD; for (int i = 0; i < HD; i++) oh[i] = 0;
        for (int p = 0; p <= pos; p++) {
          const uint16_t *vp = vcache + ((size_t)l * MAXPOS + p) * KVD + kvh * HD; float w = scoresBuf[p] / sum;
          for (int i = 0; i < HD; i++) oh[i] += w * bf2f(vp[i]);
        }
      }
    }
    tensor_mm(l, 3, D, D, &off, ATT, n, TMP);
    for (int i = 0; i < n * D; i++) X[i] += TMP[i];
    for (int t = 0; t < n; t++) rms_row(Hn + t * D, X + t * D, norms16 + (size_t)(2 * l + 1) * D, D);
    tensor_mm(l, 4, F, D, &off, Hn, n, G);
    tensor_mm(l, 5, F, D, &off, Hn, n, U);
    for (int i = 0; i < n * F; i++) { float g = G[i]; G[i] = (g / (1.0f + expf(-g))) * U[i]; }
    tensor_mm(l, 6, D, F, &off, G, n, TMP);
    for (int i = 0; i < n * D; i++) X[i] += TMP[i];
    if (g_dbg0 && l == 0) stats("  after layer 0", X + (size_t)(n - 1) * D, D);
  }
  if (hidsAll) for (int t = 0; t < n; t++) rms_row(hidsAll + (size_t)t * D, X + (size_t)t * D, norms16 + (size_t)(2 * L) * D, D);
  rms_row(hid, X + (size_t)(n - 1) * D, norms16 + (size_t)(2 * L) * D, D);
}

static int best_word(const float *hid, float *best_logit) {
  int best = 0; float bv = -1e30f;
  for (int w = 0; w < N_allowed; w++) {
    const int8_t *r = head_rows + (size_t)w * D; float s = 0;
    for (int i = 0; i < D; i++) s += r[i] * hid[i];
    s *= h2f(head_scale16[w]);
    if (s > bv) { bv = s; best = w; }
  }
  if (best_logit) *best_logit = bv;
  return best;
}

// ------------------------------------------------------------------ loading
static bool load_file_psram(const char *path, uint8_t **dst, size_t *len, const char *what) {
  File f = SD.open(path, FILE_READ);
  if (!f) return false;
  size_t n = f.size(); uint8_t *p = (uint8_t *)psalloc(n + 16, what);
  size_t done = 0;
  while (done < n) { int g = f.read(p + done, min((size_t)32768, n - done)); if (g <= 0) break; done += g; }
  f.close(); *dst = p; *len = n;
  return done == n;
}

static void boot_msg(const char *m, uint16_t color = WHITE) {
  static int y = 22;
  tft.setTextSize(1); tft.setTextColor(color); tft.setCursor(4, y); tft.print(m); y += 11;
  Serial.println(m);
}

static bool load_everything() {
  static uint8_t hdr[4096];
  gModel = SD.open("/SMOL360/model.bin", FILE_READ);
  if (!gModel) { boot_msg("cannot open /SMOL/model.bin", ST77XX_RED); return false; }
  if (!sd_read(0, hdr, sizeof hdr) || memcmp(hdr, "SMOLSD02", 8)) { boot_msg("bad model.bin header", ST77XX_RED); return false; }
  uint32_t *u = (uint32_t *)(hdr + 8);
  L = u[1]; D = u[2]; F = u[3]; H = u[4]; KV = u[5]; HD = u[6]; VOC = u[7]; GROUP = u[8];
  memcpy(&rope_theta, &u[10], 4); memcpy(&eps, &u[11], 4);
  uint32_t norms_off = u[12]; layers_off = u[13]; layer_stride = u[14]; embed_off = u[15]; embed_stride = u[16]; RB = u[18];
  if (GROUP != 64 || D > 960 || F > 2560 || KV * HD > 320 || RB <= 0) { boot_msg("model shape not supported by this firmware", ST77XX_RED); return false; }
  KVD = KV * HD;
  char m[96]; snprintf(m, sizeof m, "model: %d layers, dim %d, %d heads (%d kv), %d words", L, D, H, KV, VOC); boot_msg(m);

  size_t nb = (size_t)(2 * L + 1) * D * 2;
  norms16 = (uint16_t *)psalloc(nb, "norms");
  if (!sd_read(norms_off, norms16, nb)) { boot_msg("norms read failed", ST77XX_RED); return false; }

  size_t hl, bl;
  const esp_partition_t *hp = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40, "head");
  if (!hp) { boot_msg("flash partition 'head' not found", ST77XX_RED); return false; }
  const void *hmap = NULL; esp_partition_mmap_handle_t hh;
  if (esp_partition_mmap(hp, 0, hp->size, ESP_PARTITION_MMAP_DATA, &hmap, &hh) != ESP_OK) { boot_msg("cannot map the 'head' partition", ST77XX_RED); return false; }
  headbuf = (uint8_t *)hmap; hl = hp->size;
  if (memcmp(headbuf, "SMOLHD01", 8)) { boot_msg("bad head partition (flash head.bin with esptool)", ST77XX_RED); return false; }
  uint32_t *hu = (uint32_t *)(headbuf + 8); N_allowed = hu[0]; eos_id = hu[2];
  head_ids = (uint16_t *)(headbuf + 24);
  head_scale16 = (uint16_t *)(headbuf + 24 + 2 * (size_t)N_allowed);
  head_rows = (int8_t *)(headbuf + 24 + 4 * (size_t)N_allowed);
  head_offs = (uint32_t *)(headbuf + 24 + 4 * (size_t)N_allowed + (size_t)N_allowed * D);
  head_blob = (uint8_t *)(head_offs + N_allowed + 1);
  snprintf(m, sizeof m, "head: %d allowed words (%u KB)", N_allowed, (unsigned)(hl / 1024)); boot_msg(m);

  if (!load_file_psram("/SMOL360/bpe.bin", &bpebuf, &bl, "bpe")) { boot_msg("bpe.bin failed", ST77XX_RED); return false; }
  if (memcmp(bpebuf, "SMOLBP01", 8)) { boot_msg("bad bpe.bin", ST77XX_RED); return false; }
  n_slots = ((uint32_t *)(bpebuf + 8))[1];
  byte_to_id = (uint16_t *)(bpebuf + 16);
  merges = (Merge *)(bpebuf + 16 + 512);
  boot_msg("tokenizer loaded");

  size_t kvb = (size_t)L * MAXPOS * KVD * 2;
  kcache = (uint16_t *)psalloc(kvb, "k cache"); vcache = (uint16_t *)psalloc(kvb, "v cache");
  gAllHid = (float *)psalloc((size_t)MAXB * D * 4, "all hidden");
  if (!init_reader()) { boot_msg("too many blocks per layer", ST77XX_RED); return false; }
  X = (float *)psalloc((size_t)MAXB * D * 4, "X"); Hn = (float *)psalloc((size_t)MAXB * D * 4, "H"); Q = (float *)psalloc((size_t)MAXB * D * 4, "Q");
  ATT = (float *)psalloc((size_t)MAXB * D * 4, "ATT"); TMP = (float *)psalloc((size_t)MAXB * D * 4, "TMP");
  G = (float *)psalloc((size_t)MAXB * F * 4, "G"); U = (float *)psalloc((size_t)MAXB * F * 4, "U");
  Kt = (float *)psalloc((size_t)MAXB * KVD * 4, "Kt"); Vt = (float *)psalloc((size_t)MAXB * KVD * 4, "Vt");
  snprintf(m, sizeof m, "memory ready, PSRAM free %u KB", (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024)); boot_msg(m, ST77XX_GREEN);
  return true;
}

// ------------------------------------------------------------------ power meter (INA219 in series on the 5V line, read over I2C)
// Wiring: supply 5V -> INA219 VIN+, INA219 VIN- -> board 5V pin, grounds shared, VCC -> 3V3, SDA -> GPIO4, SCL -> GPIO5.
// The current comes from the shunt voltage register (10 uV per step, shunt 0.1 ohm on the usual module), so no calibration register is needed.
// A task adds up the energy about 10 times a second from boot. With no INA219 found the task is not started and the energy lines say so.
#define INA_SDA 4
#define INA_SCL 5
#define INA_ADDR 0x40
#define INA_SHUNT_OHM 0.1f           // R100 on the common module; change if the resistor on yours says otherwise

// Log for the dashboard: while a question runs, the meter task keeps one reading every LOG_EVERY_MS in RAM; the rows are written to the
// SD card when the question is done (the read-ahead task is idle then, so the card is not shared). 2000 readings = 2.8 h of answer.
#define LOG_EVERY_MS 5000
#define LOG_MAX 2000
struct LogSample { uint16_t t; uint16_t mv; int16_t ma; };       // seconds since the question arrived, millivolt, milliamp
static LogSample logSamples[LOG_MAX];
static volatile int logN = 0; static volatile bool logOn = false; static uint32_t logT0 = 0, logLast = 0;
static const uint32_t gBootId = esp_random() & 0xFFFF;           // tells one power-up from the next in the log files
static int64_t gEpochAtBoot = 0;                                  // unix time at boot; 0 = clock not set (send /time <unix seconds>)
static int gQnum = 0;

static bool ina_read(uint8_t reg, int16_t *val) {
  Wire.beginTransmission(INA_ADDR); Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)INA_ADDR, 2) != 2) return false;
  uint8_t hi = Wire.read(), lo = Wire.read(); *val = (int16_t)((hi << 8) | lo);
  return true;
}

static void ina_task(void *) {
  uint32_t last = millis();
  for (;;) {
    int16_t sh, bus;
    if (ina_read(0x01, &sh) && ina_read(0x02, &bus)) {
      float v = (bus >> 3) * 0.004f, ma = sh * 0.01f / INA_SHUNT_OHM;       // 4 mV per step; 10 uV / ohm = mA
      uint32_t now = millis(); float dt = (now - last) / 3600000.0f; last = now;   // dt in hours
      ina_v = v; ina_ma = ma; ina_mwh += v * ma * dt;
      if (ma > ina_peak_ma) ina_peak_ma = ma;
      if (logOn && logN < LOG_MAX && now - logLast >= LOG_EVERY_MS) {
        logLast = now; logSamples[logN].t = (uint16_t)((now - logT0) / 1000); logSamples[logN].mv = (uint16_t)(v * 1000); logSamples[logN].ma = (int16_t)ma;
        logN = logN + 1;
      }
    }
    vTaskDelay(pdMS_TO_TICKS(100));
  }
}

static void ina_begin() {
  Wire.begin(INA_SDA, INA_SCL, 100000);
  Wire.beginTransmission(INA_ADDR);
  if (Wire.endTransmission() != 0) { Serial.println("INA219 not found (check wiring, address 0x40): no energy numbers"); return; }
  ina_ok = true;
  xTaskCreatePinnedToCore(ina_task, "ina", 3072, NULL, 1, NULL, 0);
  Serial.println("INA219 found: energy is counted from boot");
}

// Append one question and its readings to the SD card. Files (root of the card, next to /SMOL360):
//   energy_questions.csv  boot,q,epoch,question,full,words,seconds,question_pass_s,mwh,avg_mw,peak_ma,volts,how_it_ended
//   energy_samples.csv    boot,q,t_s,millivolt,milliamp   (one row per 5 s)
static void csv_text(char *dst, size_t n, const char *s) {        // quote a text for CSV, newlines and quotes made harmless
  size_t o = 0; dst[o++] = '"';
  for (; *s && o + 3 < n; s++) { char c = *s; if (c == '"') dst[o++] = '\''; else dst[o++] = (c == '\n' || c == '\r') ? ' ' : c; }
  dst[o++] = '"'; dst[o] = 0;
}
static bool log_to_sd(const char *q, bool full, int words, uint32_t totalMs, uint32_t promptMs, float mwh, float peakMa, float voltsNow, const char *ended) {
  bool fresh = !SD.exists("/energy_questions.csv");
  File f = SD.open("/energy_questions.csv", FILE_APPEND);
  if (!f) return false;
  if (fresh) f.println("boot,q,epoch,question,full,words,seconds,question_pass_s,mwh,avg_mw,peak_ma,volts,how_it_ended");
  char qs[260]; csv_text(qs, sizeof qs, q);
  long long ep = gEpochAtBoot ? gEpochAtBoot + (long long)((millis() - totalMs) / 1000) : 0;   // unix time the question arrived, 0 if unknown
  f.printf("%04X,%d,%lld,%s,%d,%d,%.1f,%.1f,%.3f,%.0f,%.0f,%.2f,%s\n", (unsigned)gBootId, gQnum, ep, qs, full ? 1 : 0, words, totalMs / 1000.0f,
           promptMs / 1000.0f, mwh, totalMs ? mwh * 3600000.0f / totalMs : 0.0f, peakMa, voltsNow, ended);
  f.close();
  fresh = !SD.exists("/energy_samples.csv");
  f = SD.open("/energy_samples.csv", FILE_APPEND);
  if (!f) return false;
  if (fresh) f.println("boot,q,t_s,millivolt,milliamp");
  for (int i = 0; i < logN; i++) f.printf("%04X,%d,%u,%u,%d\n", (unsigned)gBootId, gQnum, logSamples[i].t, logSamples[i].mv, logSamples[i].ma);
  f.close();
  return true;
}

// ------------------------------------------------------------------ answering
static float gHid[960];

static void emit_piece(int w) {
  const uint8_t *p = head_blob + head_offs[w]; int len = head_offs[w + 1] - head_offs[w];
  for (int i = 0; i < len; i++) ui_answer_char((char)p[i]);
  Serial.write(p, len); ble_send((const char *)p, len);
}

// ------------------------------------------------------------------ guessing the next tokens (speculative decoding)
// ctx = every token seen so far (the last one included). Find the place in the training table where the longest run of
// tokens before it equals the end of ctx, and return the tokens that follow it there. The model checks these guesses in
// one pass, so a wrong guess costs nothing but a right one saves a whole pass over the weights.
static bool g_spec = true;

static int draft_lookup(const int *ctx, int nctx, int *out, int maxout) {
  if (maxout <= 0 || nctx < 1) return 0;
  int bestLen = 0, bestPos = -1;
  for (int i = 0; i < DRAFT_N; i++) {
    if (draft_tab[i] != (uint16_t)ctx[nctx - 1]) continue;
    int ml = 1;
    while (ml < 32 && ml < nctx && i - ml >= 0 && draft_tab[i - ml] != 0xFFFF && draft_tab[i - ml] == (uint16_t)ctx[nctx - 1 - ml]) ml++;
    if (ml > bestLen) { bestLen = ml; bestPos = i; }
  }
  if (bestLen < 2) return 0;             // nothing in the table ends the same way
  // A weak match is a weak guess, but a short batch costs almost nothing here (a pass is limited by reading the model from the SD card,
  // and up to about 12 tokens are calculated while it reads), so guess a few tokens instead of none. A strong match guesses the full length.
  if (bestLen < 8 && maxout > 12) maxout = 12;
  int k = 0;
  for (int i = bestPos + 1; i < DRAFT_N && k < maxout && draft_tab[i] != 0xFFFF; i++) out[k++] = draft_tab[i];
  return k;
}

static void answer_question(const char *q, bool full) {
  char prompt[400]; snprintf(prompt, sizeof prompt, "Instruction: %s\nResponse:", q);
  static uint16_t ids16[160]; int n = encode(prompt, ids16, 160);
  static int ids[160]; for (int i = 0; i < n; i++) ids[i] = ids16[i];
  ui_question(q); ui_clear_answer();
  if (n > MAXPOS - 8) { out_text("question too long (%d tokens)\n", n); ui_status("question too long", 0, ""); return; }
  out_text("\nQ: %s\n(%d prompt tokens; guessing ahead is %s)\nA:", q, n, g_spec ? "ON" : "OFF");
  gStart = millis(); gWords = 0;
  uint32_t t0 = millis();
  gQnum++; logN = 0; logT0 = millis(); logLast = logT0 - LOG_EVERY_MS; logOn = ina_ok;   // first reading goes in right away
  const float e0 = ina_mwh; gE0 = e0; ina_peak_ma = 0;                       // energy meter reading when the question arrived
  gPhase = "reading the question";
  const int maxNew = full ? min(FULL_TOKENS, MAXPOS - n) : min(DEMO_TOKENS, MAXPOS - n);
  static int ctx[MAXPOS + MAXB];
  int nctx = n; for (int i = 0; i < n; i++) ctx[i] = ids[i];
  int pos = n, produced = 0, rounds = 0, guessedOk = 0, guessedTotal = 0; bool finished = false;
  for (int off = 0; off < n; off += MAXB) {
    int m = min(MAXB, n - off);
    if (off + m < n || !g_spec) { forward_batch(ids + off, m, off, L, gHid); continue; }
    // Last piece of the question: it also carries the first guesses of the answer, so the model reads the question and checks
    // those guesses in ONE pass over the weights. Guess i is right if the model, standing after the token before it, picks it.
    int tk[MAXB], widx0[MAXB];
    for (int i = 0; i < m; i++) tk[i] = ids[off + i];
    int G = draft_lookup(ctx, n, tk + m, min(MAXB - m, min(maxNew, MAXPOS - n)));
    forward_batch(tk, m + G, off, L, gHid, gAllHid);
    int a = 0;
    while (a < G) {
      int w = best_word(gAllHid + (size_t)(m - 1 + a) * D, NULL);
      if (head_ids[w] != tk[m + a]) break;
      widx0[a] = w; a++;
    }
    memcpy(gHid, gAllHid + (size_t)(m - 1 + a) * D, (size_t)D * 4);      // hidden state after the last accepted token
    guessedTotal += G; guessedOk += a;
    for (int i = 0; i < a; i++) {
      if (tk[m + i] == eos_id) { finished = true; break; }
      emit_piece(widx0[i]); gWords++; produced++;
      ctx[nctx++] = tk[m + i];
    }
    pos += a;
    Serial.printf("\n[question pass: %d guesses, %d right]\n", G, a);
  }
  uint32_t tPrompt = millis() - t0;
  gPhase = "writing the answer";
  bool timedOut = false;
  while (!finished && produced < maxNew && pos < MAXPOS) {
    if (MAX_ANSWER_S > 0 && millis() - t0 > (uint32_t)MAX_ANSWER_S * 1000UL) { timedOut = true; break; }
    int w0 = best_word(gHid, NULL), id0 = head_ids[w0];
    if (id0 == eos_id) break;
    int batch[MAXB], widx[MAXB]; batch[0] = id0; widx[0] = w0;
    ctx[nctx] = id0;
    int room = min(maxNew - produced, MAXPOS - pos) - 1;            // tokens still allowed after the first one
    int k = 1 + (g_spec ? draft_lookup(ctx, nctx + 1, batch + 1, min(MAXB - 1, room)) : 0);
    uint32_t tr = millis();
    forward_batch(batch, k, pos, L, gHid, gAllHid);
    // check the guesses: guess i+1 is right if the model, standing after token i, picks exactly that token
    int m = 0;
    while (m + 1 < k) {
      int w = best_word(gAllHid + (size_t)m * D, NULL);
      if (head_ids[w] != batch[m + 1]) break;
      widx[m + 1] = w; m++;
    }
    memcpy(gHid, gAllHid + (size_t)m * D, (size_t)D * 4);
    guessedTotal += k - 1; guessedOk += m; rounds++;
    for (int i = 0; i <= m; i++) {
      if (batch[i] == eos_id) { finished = true; break; }
      emit_piece(widx[i]); gWords++; produced++;
      ctx[nctx++] = batch[i];
    }
    pos += m + 1;
    Serial.printf("\n[round %d: batch of %d, %d guesses right, %lu s]\n", rounds, k, m, (unsigned long)((millis() - tr) / 1000));
  }
  Serial.printf("\n[%d rounds, %d of %d guesses right]\n", rounds, guessedOk, guessedTotal);
  uint32_t total = millis() - t0;
  char a[96], b[96], tm[16]; fmtTime(tm, sizeof tm, total);
  snprintf(a, sizeof a, "done: %d words in %s (question pass %.0f s)", gWords, tm, tPrompt / 1000.0f);
  float eq = ina_mwh - e0;                                           // mWh used by this question
  if (ina_ok) snprintf(b, sizeof b, "%.1f s/word, %.1f mWh, avg %.0f mW", gWords ? (total - tPrompt) / 1000.0f / gWords : 0.0f, eq, eq * 3600000.0f / total);
  else snprintf(b, sizeof b, "%.1f s per word", gWords ? (total - tPrompt) / 1000.0f / gWords : 0.0f);
  ui_status(a, 1.0f, b);
  out_text("\n[%s]\n", a);
  if (timedOut) out_text("[stopped: time limit of %d min reached, the answer is cut off]\n", MAX_ANSWER_S / 60);
  if (ina_ok) out_text("[energy: %.2f mWh for this question, average %.0f mW, peak %.0f mA, %.2f V now]\n", eq, eq * 3600000.0f / total, (float)ina_peak_ma, (float)ina_v);
  logOn = false;
  if (ina_ok) out_text("%s", log_to_sd(q, full, gWords, total, tPrompt, eq, (float)ina_peak_ma, (float)ina_v, timedOut ? "time_limit" : (finished ? "finished" : "word_limit"))
                       ? "[log: saved to the SD card]\n" : "[log: could NOT be written to the SD card]\n");
  out_text("ready for the next question\n");
  gStart = 0;
}

// /idle: the board waits for 60 s with nothing to do (screen on, Bluetooth as it is now) and the meter averages the draw. Appended to
// /energy_idle.csv (boot,epoch,seconds,mwh,avg_mw,volts,peak_ma) so the dashboard can subtract it, like the Pi page subtracts its idle watts.
// Do not send anything while it runs; best done with the phone connected, as during a question.
static void run_idle() {
  if (!ina_ok) { out_text("no INA219 found\n"); return; }
  ui_status("measuring idle power for 60 s: send nothing", 0, "screen on, Bluetooth as it is");
  out_text("idle measurement started: 60 s, send nothing\n");
  const float e0 = ina_mwh; ina_peak_ma = 0; const uint32_t t0 = millis();
  while (millis() - t0 < 60000UL) delay(200);
  const uint32_t ms = millis() - t0; const float mwh = ina_mwh - e0, mw = mwh * 3600000.0f / ms;
  char b[96]; snprintf(b, sizeof b, "idle: %.0f mW (%.2f mWh in %lu s), %.2f V", mw, mwh, (unsigned long)(ms / 1000), (float)ina_v);
  ui_status(b, 1.0f, "ready for the next question");
  out_text("[%s, peak %.0f mA]\n", b, (float)ina_peak_ma);
  bool fresh = !SD.exists("/energy_idle.csv");
  File f = SD.open("/energy_idle.csv", FILE_APPEND);
  if (!f) { out_text("[log: could NOT be written to the SD card]\n"); return; }
  if (fresh) f.println("boot,epoch,seconds,mwh,avg_mw,volts,peak_ma");
  long long ep = gEpochAtBoot ? gEpochAtBoot + (long long)(millis() / 1000) - (long long)(ms / 1000) : 0;
  f.printf("%04X,%lld,%.1f,%.3f,%.0f,%.2f,%.0f\n", (unsigned)gBootId, ep, ms / 1000.0f, mwh, mw, (float)ina_v, (float)ina_peak_ma);
  f.close();
  out_text("[log: saved to the SD card]\n");
}

static void run_selftest(int which) {
  g_dbg0 = true; gStart = millis(); gWords = 0;
  ui_question(which == 1 ? "/test: one token through all layers" : "/test2: two tokens in one batch");
  ui_clear_answer(); gPhase = "self test";
  if (which == 1) {
    int tk[1] = {25464};
    forward_batch(tk, 1, 0, L, gHid);
    stats("test1 token 25464 alone, final hidden", gHid, D);
  } else {
    int tk[2] = {25464, 42};
    forward_batch(tk, 2, 0, L, gHid);
    stats("test2 tokens 25464, 42 (second token), final hidden", gHid, D);
    float bl; int w = best_word(gHid, &bl);
    out_text("test2 best allowed word index %d (token id %d), logit %.5f\n", w, head_ids[w], bl);
  }
  g_dbg0 = false;
  out_text("self test finished in %lu s\n", (unsigned long)((millis() - gStart) / 1000));
  ui_status("self test finished", 1.0f, "see serial output");
  gStart = 0;
}

// ------------------------------------------------------------------ setup and loop
static char serLine[200]; static int serLen = 0;

void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println("\n=== SmolFem-360M fast demo ===");
  init_lut();
  xi = (float *)heap_caps_aligned_alloc(16, XI_BYTES, MALLOC_CAP_INTERNAL);
  if (!xi) Serial.println("FATAL: no internal RAM for the input chunk");
  ina_begin();
  lcdspi.begin(TFT_SCK, -1, TFT_MOSI, TFT_CS);
  tft.init(240, 320, SPI_MODE3); tft.setSPISpeed(40000000); tft.setRotation(1); tft.fillScreen(BLACK);
  ui_title();
  sdspi.begin(SD_SCK, SD_MISO, SD_MOSI, SD_CS);
  if (!SD.begin(SD_CS, sdspi, 20000000)) { boot_msg("SD card: mount FAILED (reader power?)", ST77XX_RED); return; }
  char m[64]; snprintf(m, sizeof m, "SD card: %llu MB", SD.cardSize() / 1048576ULL); boot_msg(m, ST77XX_GREEN);
  if (!load_everything()) return;
#if USE_BLE
  ble_begin();
  boot_msg("Bluetooth: advertising as SmolFem-360M", ST77XX_GREEN);
#else
  boot_msg("Bluetooth: OFF (set USE_BLE 1 once the power supply is solid)");
#endif
  delay(1500);
  tft.fillRect(0, 18, 320, 220, BLACK);
  ui_status("ready: send a question by Bluetooth or USB serial", 0, "e.g.  Who is Fei-Fei Li?");
  tft.setTextSize(2); tft.setTextColor(ST77XX_CYAN); tft.setCursor(4, 70); tft.print("Ask me something!");
  tft.setTextSize(1); tft.setTextColor(WHITE); tft.setCursor(4, 100); tft.print("Phone: Serial Bluetooth Terminal -> SmolFem-360M");
  tft.setCursor(4, 112); tft.print("Each answer takes many minutes (SD card speed).");
  Serial.println("ready. Type a question and press Enter. Commands: /test /test2 /mem /idle /time <unix> /full <question>");
}

static void handle_line(char *line) {
  while (*line == ' ') line++;
  if (!*line) return;
  if (!strcmp(line, "/test")) { run_selftest(1); return; }
  if (!strcmp(line, "/test2")) { run_selftest(2); return; }
  if (!strcmp(line, "/spec")) { g_spec = !g_spec; out_text("guessing ahead is now %s\n", g_spec ? "ON" : "OFF"); return; }
  if (!strcmp(line, "/idle")) { run_idle(); return; }
  if (!strncmp(line, "/time ", 6)) {                       // /time 1792000000 : tells the board the real time (unix seconds) for the log
    gEpochAtBoot = atoll(line + 6) - (long long)(millis() / 1000);
    out_text("clock set; the log now carries real times\n"); return;
  }
  if (!strcmp(line, "/power")) {
    if (ina_ok) out_text("%.2f V, %.0f mA, %.0f mW now; %.2f mWh since boot\n", (float)ina_v, (float)ina_ma, (float)ina_v * (float)ina_ma, (float)ina_mwh);
    else out_text("no INA219 found\n");
    return;
  }
  if (!strcmp(line, "/mem")) {
    out_text("PSRAM free %u KB, internal free %u KB\n", (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024), (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024));
    return;
  }
  bool full = false;
  if (!strncmp(line, "/full ", 6)) { full = true; line += 6; }
  answer_question(line, full);
}

void loop() {
  if (L == 0) { delay(1000); return; }          // setup failed, message is on the screen
  while (Serial.available()) {
    char ch = Serial.read();
    if (ch == '\r') continue;
    if (ch == '\n') { serLine[serLen] = 0; if (serLen) handle_line(serLine); serLen = 0; }
    else if (serLen < (int)sizeof(serLine) - 1) serLine[serLen++] = ch;
  }
  if (!ble_ready && ble_len > 0 && millis() - ble_last_ms > 600) ble_ready = true;     // phone apps that send no line ending
  if (ble_ready) {
    static char q[200]; int n = ble_len; memcpy(q, ble_line, n); q[n] = 0; ble_len = 0; ble_ready = false;
    handle_line(q);
  }
  delay(5);
}






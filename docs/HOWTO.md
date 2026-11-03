---
title: "How to run a small language model on an ESP32-S3 N16R8"
date: 2026-11-03
draft: true
summary: "Run SmolFem 360M on a 5 euro chip by streaming the model from an SD card, and measure how much energy one answer costs."
---

# How to run a small language model on an ESP32-S3 N16R8

## Streaming SmolFem 360M from an SD card

This guide is for the **ESP32-S3 DevKitC-1 board with the WROOM-1 N16R8 module** (16 MB flash, 8 MB octal PSRAM, two USB-C ports).

The idea: the model (224 MB at 4-bit) does not fit in the chip, so the board reads it from an SD card one layer at a time while it calculates. It works, but it is slow and it is a memoriser: it answers the questions it was trained on and makes things up for the rest. The point is the experiment, and measuring how much energy one answer takes compared to the Raspberry Pi 5 in the solarbot.

### What you need

* ESP32-S3 N16R8 DevKitC-1 board (check that the chip says N16R8, or you will have no PSRAM)
* microSD card, 32 GB, formatted FAT32, and an SPI card reader that runs on **3.3 V** (not the 5 V type, see the warning at the wiring)
* 2.0 inch SPI screen with an ST7789V controller (240 x 320, pins VCC GND DIN CLK CS DC RST BL)
* Breadboard and jumper wires
* A laptop with Windows or a Mac, a USB-C cable that carries data
* A phone with the app **Serial Bluetooth Terminal** (Kai Morich) to ask questions

> Read the small numbers printed on the board next to each pin hole. Do not trust the coloured boxes on the seller's poster, their rows are shifted by one.

### Wiring

Hold the board with the antenna at the top and the two USB-C ports at the bottom. The board has only two 3V3 pins but three parts need 3V3, so put a wire from a 3V3 pin to the 3V3 rail of the breadboard and a wire from a GND pin to the GND rail, and connect the parts to the rails.

**Screen (ST7789V)**

    Screen VCC  to  3V3 rail
    Screen BL   to  3V3 rail
    Screen GND  to  GND rail
    Screen DIN  to  GPIO 40
    Screen CLK  to  GPIO 39
    Screen CS   to  GPIO 41
    Screen DC   to  GPIO 42
    Screen RST  to  GPIO 2

**SD card reader (SPI)**

    Reader VCC   to  3V3 rail  (the reader must be a 3.3 V one)
    Reader GND   to  GND rail
    Reader CS    to  GPIO 10
    Reader MOSI  to  GPIO 11
    Reader SCK   to  GPIO 12
    Reader MISO  to  GPIO 13

The SD card gets GPIO 10 to 13 on purpose: those are the fast SPI pins of the chip. The screen can live on slower pins.

> **Important: use a 3.3 V SD card reader, not the common 5 V one.** The common card reader modules (with a little AMS1117 regulator and level shifter, "supports 5 V") do not work on this board. On 3V3 the card answers but never starts, and the 5V0 pin of this board does not give out the USB 5 V, so the reader cannot be fed from there either. Only a reader that is built for 3.3 V worked. Check the product page or the chip on the module before you buy.

**Pins to stay away from:** 35, 36, 37 (octal PSRAM), 0 (boot button), 19 and 20 (USB), 3, 45, 46 (they set the chip's start-up mode), 48 (RGB LED). Free and safe: 1, 8, 17, 18, 21, 38, 47.

Always unplug the USB before changing wires, and check that no two jumpers touch.

**Which USB port?** The port marked UART (a CH343 chip, shows up as a COM port) uploads without pressing any button and prints the serial text. Use that one.

### Install the tools

Install **arduino-cli** (or the Arduino IDE) and add the Espressif board index:

    arduino-cli config add board_manager.additional_urls https://espressif.github.io/arduino-esp32/package_esp32_index.json
    arduino-cli core update-index
    arduino-cli core install esp32:esp32@3.3.12

Install the libraries **Adafruit GFX Library**, **Adafruit BusIO** and **Adafruit ST7735 and ST7789 Library**:

    arduino-cli lib install "Adafruit GFX Library" "Adafruit BusIO" "Adafruit ST7735 and ST7789 Library"

The Bluetooth, SD and fast maths (esp-dsp) libraries come with the ESP32 core. Newer ESP32 cores may behave differently, we used 3.3.12.

### Get the files

All the files are in one repository: https://github.com/lekkerYing/esp32s3-bot

    sketch/smol_360/    the Arduino sketch, its partitions.csv and draft_data.h (the guess table)
    scripts/            qat360.py, export_sd360.py, make_draft360.py: shrink the model, export the files, build the guess table
    docs/HOWTO.md       this guide

The big model files (`model.bin`, `bpe.bin`, `head.bin`) are not in git because `model.bin` is 224 MB. Download them from **MODEL-LINK-TO-ADD**, or make them yourself with the scripts as described below.

### Get the model onto the SD card

You need the model files. We made them from smolFem (a LoRA fine-tune of SmolLM2-360M-Instruct, made by Ambika on her own data). The scripts are in the `scripts` folder of the repository (see "Get the files") and the Python needs torch, transformers and numpy.

1. **Shrink the model to 4 bit.** Plain rounding to 4 bit drops the word accuracy on the dataset from 71.8 percent to about 50 percent. So we retrain the 4-bit copy to copy the full model (quantization-aware distillation): `python qat360.py` (set `SMOLFEM_DIR` to the folder with the full smolFem model and `DATA_DIR` to the folder with the dataset). After that it scores 71.0 percent and agrees with the full model on 92 percent of its words.
2. **Export the files:** `python export_sd360.py`. This writes three files:

        model.bin   224 MB   the 32 layers, 4-bit with a scale and minimum per 64 weights, plus the word embeddings
        bpe.bin     0.5 MB   the tokenizer
        head.bin    5.6 MB   the output words that occur in the dataset (5,748 words), with their rows

   Limiting the output to words that occur in the answers is a trick from the [Barista project](https://huggingface.co/slvDev/esp32-ai-barista) (an espresso question answerer by slvDev that runs on the ESP32): the chip then does not have to calculate a score for all 49,152 words.
3. **Build the guess table:** `python make_draft360.py`. It writes `draft_data.h` into the sketch folder (see "Why it is not slower" below).
4. **Copy to the SD card:** make a folder `SMOL360` on the card and put `model.bin` and `bpe.bin` in it. The `head.bin` does not go on the card, it goes into the flash of the board (next step).

### Flash the board

The sketch is `sketch/smol_360` in the repository. It has its own `partitions.csv`, which gives the board a 2 MB program slot and a 6 MB data partition called `head` at flash address `0x210000`.

Compile and upload (close any serial monitor first, an open monitor blocks the port):

    arduino-cli compile --fqbn esp32:esp32:esp32s3:CDCOnBoot=default,CPUFreq=240,FlashMode=qio,FlashSize=16M,PartitionScheme=app3M_fat9M_16MB,PSRAM=opi --build-property compiler.optimization_flags=-O3 smol_360
    arduino-cli upload -p COM15 --fqbn esp32:esp32:esp32s3:CDCOnBoot=default,CPUFreq=240,FlashMode=qio,FlashSize=16M,PartitionScheme=app3M_fat9M_16MB,PSRAM=opi smol_360

Replace COM15 with your port. On a Mac it looks like `/dev/cu.usbserial-...`. Then write `head.bin` into the `head` partition with esptool:

    python -m esptool --chip esp32s3 --port COM15 write_flash 0x210000 head.bin

If the board says `bad head partition (flash head.bin with esptool)` on its screen, this step was skipped or went to the wrong address.

If the port does not show up: hold the BOOT button, tap RST, release BOOT.

### Ask a question

1. Put the SD card in the reader and power the board.
2. On the phone open Serial Bluetooth Terminal, menu Devices, tab **Bluetooth LE**, scan, connect to **SmolFem-360M**. You can also use the serial monitor at 115200 baud on the UART port.
3. Type a question with a capital letter, for example `/full Who is Fei-Fei Li?` and send it.

The screen shows the question in yellow, the answer as it is written in green, a bar for the layer being read, and the elapsed time. Commands:

    /full <question>   lift the 30 word limit (answers stop at 60 tokens, about 45 words)
    /power             show the live INA219 reading
    /idle              measure how much the board draws doing nothing
    /time <unix time>  set the clock (the board has none after power-up)
    /mem               print free memory
    /test              check the maths against the laptop

### What to expect

* **A question the model learned**, with `/full`: the answer arrives in about 6 min 35 s for 45 words (the first pass over the model takes about 3 minutes, then one more round). The words are identical to the laptop's 4-bit model.
* **A question it never learned:** about 2 minutes per word. "What is your favorite color?" with `/full` took 2 h 19 min for 73 words and rambled. That is why a plain question stops after 30 words.
* The SD card is the bottleneck. In SPI mode we measured at most 1.8 MB/s, and every word needs the whole 224 MB model read once.

**Why it is not slower: guessing ahead.** The board guesses the next words from a table of the model's own answers and then checks up to 32 guesses in one pass over the SD card. A guess is only accepted if the model's own best word is the same, so the answer is exactly what the model would have written anyway. For questions in the dataset this brought a 26 minute answer (135M model) down to 2 minutes and for the 360M model to about 6.5 minutes. For other questions there is nothing to guess from.

**Speed ups that worked:** reading the next layer on one core while the other core calculates; the chip's own dot product routine (esp-dsp); keeping the input in internal RAM.

### Measure the energy

To compare with the Pi 5, the INA219 sits in the middle of the 5 V wire between the battery and the board, so all current goes through it.

    PiSugar 5V output  to  INA219 VIN+
    INA219 VIN-        to  ESP32 5V pin
    PiSugar GND        to  ESP32 GND
    INA219 VCC         to  3V3
    INA219 GND         to  GND (the same ground)
    INA219 SDA         to  GPIO 4
    INA219 SCL         to  GPIO 5

VIN+ and VIN- are the big terminals, VCC is the small logic pin. Do not mix them up. If another wire still joins the battery to the board, the INA219 reads zero.

Unplug the USB when you measure, or the laptop shares the load. The board prints the energy of each question on the screen, on serial and over Bluetooth, and writes a line to the SD card (`energy_questions.csv` and `energy_samples.csv`): words, seconds, mWh, average mW, peak mA and volts. Send `/time` after every power-up or the rows get no date. Run `/idle` too, so you can subtract what the board uses doing nothing.

First results (screen backlight dimmed, battery PiSugar 3, USB unplugged):

    Who is Fei-Fei Li?   30 words   5 min 35 s   79.2 mWh   average 853 mW   peak 238 mA
    what is your name?   30 words   59 min       845.6 mWh  average 857 mW   peak 242 mA

The power stays about the same, so the energy per answer depends almost only on how long it takes. Idle was 497 mW (dimmed backlight). Compare it with the Pi 5 carefully: the Pi page reads the Pi's own power chip (board only), the INA219 measures the whole ESP32 board including screen and SD card.

### If something goes wrong

* **Screen black, SD "mount FAILED":** almost always a loose wire or solder joint on 3V3 or GND. Wiggle the wires, check VCC and GND of the screen and the reader.
* **Upload fails with "Access is denied":** a serial monitor is still open.
* **"waiting for download" in the serial text:** the board is still in download mode, press RST on its own.
* **Colours or picture wrong on the screen:** the sketch uses SPI mode 3, 40 MHz, rotation 1 and the Adafruit ST7789 library.
* **Phone loses the Bluetooth connection:** the board keeps working on its screen. Ask again, or read the answer on the screen or serial.

### Sources and thanks

* Model smolFem and its dataset: Ambika, Computational Mama (a LoRA fine-tune of SmolLM2-360M-Instruct)
* [SmolLM2, Hugging Face](https://huggingface.co/HuggingFaceTB)
* [DaveBben, esp32-llm](https://github.com/DaveBben/esp32-llm)
* [slvDev, esp32-ai](https://github.com/slvDev/esp32-ai) and its [Barista model](https://huggingface.co/slvDev/esp32-ai-barista) (the idea of limiting the output words)
* [Karpathy, llama2.c](https://github.com/karpathy/llama2.c)
* [Espressif ESP-DSP library](https://github.com/espressif/esp-dsp)
* [Circuit Digest, Running 260K Tiny-LLM on ESP32](https://circuitdigest.com/articles/running-260K-tiny-llm-on-esp32)
* [Solarbot, Computational Mama](https://github.com/computational-mama/solarbot)
* [PiSugar 3 battery](https://www.tindie.com/products/pisugar/pisugar-3-plus-battery-for-raspberry-pi-3b3b4b/)

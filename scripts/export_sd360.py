"""Export the 4-bit smolFem (SmolLM2-360M based) to the files for the SD card and the flash.

usage: python export_sd360.py [model folder] [group=64]

Files written to OUT (model.bin, bpe.bin go into the folder SMOL on the SD card; head.bin is written to a flash partition):
  model.bin  header | norms (fp16) | L layer blocks (int4, fp16 scale + min) | int8 embedding rows
             every tensor is stored in blocks of RB rows: [codes of RB rows][scales][mins], so the board reads one block at a time
  head.bin   the allowed output words: ids, int8 head rows, decoded bytes of each word (lives in flash, memory-mapped)
  bpe.bin    tokenizer: byte -> id table and a hash table of the 48,900 merge rules

All integers little endian. Layout is repeated in the C firmware; change both together.
"""
import json, os, re, struct, sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = sys.argv[1] if len(sys.argv) > 1 else "smolFem360-qat-int4g64"
DATA = os.environ.get("DATA_DIR", "dataset")   # folder with the cleaned train and valid files
OUT = os.environ.get("OUT_DIR", "sdpack")
GROUP = int(sys.argv[2]) if len(sys.argv) > 2 else 64
RB = 256                                                   # rows per block
ALIGN = 4096
os.makedirs(OUT, exist_ok=True)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
cfg = model.config
L, D, F, H, KV = cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size, cfg.num_attention_heads, cfg.num_key_value_heads
HD = D // H
V = cfg.vocab_size
assert (L, D, F, H, KV, HD, V) == (32, 960, 2560, 15, 5, 64, 49152), (L, D, F, H, KV, HD, V)


def pad(n, a=ALIGN):
    return (n + a - 1) // a * a


def quant4(w):
    """w [out, in] already on the 4-bit grid (after QAT). Returns packed codes, fp16 scale, fp16 min"""
    o, i = w.shape
    wg = w.reshape(o, i // GROUP, GROUP)
    mn, mx = wg.amin(-1), wg.amax(-1)
    scale = (mx - mn).clamp(min=1e-8) / 15
    q = torch.round((wg - mn[..., None]) / scale[..., None]).clamp(0, 15).to(torch.uint8).reshape(o, i)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).numpy()               # low nibble = even column
    return packed, scale.half().numpy(), mn.half().numpy()


TENSORS = [("q", lambda ly: ly.self_attn.q_proj.weight), ("k", lambda ly: ly.self_attn.k_proj.weight),
           ("v", lambda ly: ly.self_attn.v_proj.weight), ("o", lambda ly: ly.self_attn.o_proj.weight),
           ("gate", lambda ly: ly.mlp.gate_proj.weight), ("up", lambda ly: ly.mlp.up_proj.weight),
           ("down", lambda ly: ly.mlp.down_proj.weight)]


def tensor_bytes(o, i):
    return o * i // 2 + 2 * o * (i // GROUP) * 2          # codes + scales + mins


shapes = {"q": (D, D), "k": (KV * HD, D), "v": (KV * HD, D), "o": (D, D), "gate": (F, D), "up": (F, D), "down": (D, F)}
layer_payload = sum(tensor_bytes(*shapes[n]) for n, _ in TENSORS)
layer_stride = pad(layer_payload)

# ----------------------------------------------------------------------------- model.bin
norms_off = ALIGN
norms_bytes = (L * 2 + 1) * D * 2
layers_off = norms_off + pad(norms_bytes)
embed_off = layers_off + L * layer_stride
embed_stride = D + 2                                    # int8 row + fp16 scale

_raw_cfg = json.load(open(os.path.join(MODEL, "config.json"), encoding="utf-8"))
_rope = _raw_cfg.get("rope_theta") or (_raw_cfg.get("rope_parameters") or {}).get("rope_theta") or 100000.0
assert float(_rope) == 100000.0, _rope                  # SmolLM2 uses theta 1e5
header = struct.pack("<8s10I2f7I", b"SMOLSD02", 2, L, D, F, H, KV, HD, V, GROUP, 4,
                     float(_rope), float(cfg.rms_norm_eps),
                     norms_off, layers_off, layer_stride, embed_off, embed_stride, layer_payload, RB)
assert len(header) < ALIGN

with open(os.path.join(OUT, "model.bin"), "wb") as f:
    f.write(header); f.write(b"\0" * (ALIGN - len(header)))
    # norms: for each layer input_layernorm, post_attention_layernorm; then the final norm
    norms = []
    for ly in model.model.layers:
        norms += [ly.input_layernorm.weight, ly.post_attention_layernorm.weight]
    norms.append(model.model.norm.weight)
    nb = torch.cat([n.detach().reshape(-1) for n in norms]).half().numpy().tobytes()
    f.write(nb); f.write(b"\0" * (pad(norms_bytes) - len(nb)))
    for li, ly in enumerate(model.model.layers):
        written = 0
        for name, get in TENSORS:
            w = get(ly).detach()
            for r0 in range(0, w.shape[0], RB):                  # one block = RB rows: codes, then scales, then mins
                codes, sc, mn = quant4(w[r0:r0 + RB])
                for arr in (codes, sc, mn):
                    b = arr.tobytes(); f.write(b); written += len(b)
        assert written == layer_payload
        f.write(b"\0" * (layer_stride - written))
    # embedding table, int8 per row (the same table is the tied output head)
    E = model.model.embed_tokens.weight.detach()
    es = (E.abs().amax(-1).clamp(min=1e-8) / 127)
    eq = torch.round(E / es[:, None]).clamp(-127, 127).to(torch.int8).numpy()
    es16 = es.half().numpy()
    for r in range(V):
        f.write(eq[r].tobytes()); f.write(es16[r].tobytes())
print("model.bin", os.path.getsize(os.path.join(OUT, "model.bin")) / 1e6, "MB | layer stride", layer_stride, "| payload", layer_payload)

# ----------------------------------------------------------------------------- head.bin (allowed output words)
def load(fn):
    rows = []
    for l in open(os.path.join(DATA, fn), encoding="utf-8"):
        if l.strip():
            t = json.loads(l)["text"]
            m = re.match(r"^Instruction:\s*(.*?)\nResponse:\s*(.*)$", t, re.S)
            rows.append((m.group(1).strip(), m.group(2).strip()))
    return rows


qa = load("train_clean_text.jsonl") + load("valid_clean_text.jsonl")
allowed = {tok.eos_token_id}
for q, a in qa:
    allowed.update(tok(" " + a, add_special_tokens=False)["input_ids"]); allowed.update(tok(a, add_special_tokens=False)["input_ids"])
ids = sorted(allowed)
N = len(ids)


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


b2u = bytes_to_unicode(); u2b = {v: k for k, v in b2u.items()}
tj = json.load(open(os.path.join(MODEL, "tokenizer.json"), encoding="utf-8"))
vocab = tj["model"]["vocab"]; inv = {i: s for s, i in vocab.items()}


def token_bytes(i):
    if i == tok.eos_token_id:
        return b""                                            # end of text prints nothing
    return bytes(u2b[c] for c in inv[i])


blobs = [token_bytes(i) for i in ids]
offs = np.cumsum([0] + [len(b) for b in blobs]).astype(np.uint32)
rows_i8 = eq[ids]; scales16 = es16[ids]
with open(os.path.join(OUT, "head.bin"), "wb") as f:
    f.write(struct.pack("<8s4I", b"SMOLHD01", N, D, tok.eos_token_id, 0))
    f.write(np.array(ids, dtype="<u2").tobytes())
    f.write(scales16.astype("<f2").tobytes())
    f.write(rows_i8.astype(np.int8).tobytes())
    f.write(offs.astype("<u4").tobytes())
    f.write(b"".join(blobs))
print("head.bin", os.path.getsize(os.path.join(OUT, "head.bin")) / 1e6, "MB |", N, "allowed words")

# ----------------------------------------------------------------------------- bpe.bin (tokenizer)
merges = tj["model"]["merges"]
SLOTS = 65536
table = np.full((SLOTS, 4), 0xFFFF, dtype="<u2")                  # a, b, new id, rank


def h(a, b):
    return ((a * 0x9E3779B1 + b * 0x85EBCA6B) & 0xFFFFFFFF) >> 16


for rank, m in enumerate(merges):
    a_s, b_s = m if isinstance(m, list) else m.split(" ")
    a, b, c = vocab[a_s], vocab[b_s], vocab[a_s + b_s]
    s = h(a, b)
    while table[s, 0] != 0xFFFF:
        s = (s + 1) & (SLOTS - 1)
    table[s] = (a, b, c, rank)
byte_to_id = np.array([vocab.get(b2u[b], 0xFFFF) for b in range(256)], dtype="<u2")
_missing = [b for b in range(256) if byte_to_id[b] == 0xFFFF]
print("bytes without a single-byte token:", len(_missing), "| any printable ASCII or newline missing?",
      [b for b in _missing if 32 <= b < 127 or b == 10])
with open(os.path.join(OUT, "bpe.bin"), "wb") as f:
    f.write(struct.pack("<8s2I", b"SMOLBP01", len(merges), SLOTS))
    f.write(byte_to_id.tobytes()); f.write(table.tobytes())
print("bpe.bin", os.path.getsize(os.path.join(OUT, "bpe.bin")) / 1e6, "MB |", len(merges), "merges")

# ----------------------------------------------------------------------------- round trip: read the files back and test the model
print("\nround trip: rebuilding the weights from model.bin ...", flush=True)
raw = open(os.path.join(OUT, "model.bin"), "rb").read()
(_m, ver, L2, D2, F2, H2, KV2, HD2, V2, G2, B2, rope, eps, n_off, l_off, l_stride, e_off, e_stride, payload, RB2) = \
    struct.unpack("<8s10I2f7I", raw[:struct.calcsize("<8s10I2f7I")])
assert RB2 == RB
assert (L2, D2, F2, H2, KV2, HD2, V2, G2, B2) == (L, D, F, H, KV, HD, V, GROUP, 4)


def deq(buf, o, i):
    """rebuild a whole tensor from its blocks, reading them the way the board does"""
    out = np.empty((o, i), dtype=np.float32); pos = 0
    for r0 in range(0, o, RB):
        nr = min(RB, o - r0); nb_codes = nr * i // 2; ng = nr * (i // GROUP)
        codes = np.frombuffer(buf, dtype=np.uint8, count=nb_codes, offset=pos).reshape(nr, i // 2)
        sc = np.frombuffer(buf, dtype="<f2", count=ng, offset=pos + nb_codes).astype(np.float32).reshape(nr, i // GROUP)
        mn = np.frombuffer(buf, dtype="<f2", count=ng, offset=pos + nb_codes + ng * 2).astype(np.float32).reshape(nr, i // GROUP)
        q = np.empty((nr, i), dtype=np.float32); q[:, 0::2] = codes & 15; q[:, 1::2] = codes >> 4
        out[r0:r0 + nr] = (q.reshape(nr, i // GROUP, GROUP) * sc[..., None] + mn[..., None]).reshape(nr, i)
        pos += nb_codes + ng * 4
    return out


worst = 0.0
with torch.no_grad():
    for li, ly in enumerate(model.model.layers):
        base = l_off + li * l_stride; pos = 0
        for name, get in TENSORS:
            o, i = shapes[name]; size = tensor_bytes(o, i)
            w = deq(raw[base + pos: base + pos + size], o, i); pos += size
            ref = get(ly)
            worst = max(worst, float(np.abs(w - ref.numpy()).max() / np.abs(ref.numpy()).max()))
            ref.copy_(torch.from_numpy(w))
    Er = np.empty((V, D), dtype=np.float32)
    for r in range(V):
        o_ = e_off + r * e_stride
        Er[r] = np.frombuffer(raw, dtype=np.int8, count=D, offset=o_).astype(np.float32) * np.frombuffer(raw, dtype="<f2", count=1, offset=o_ + D).astype(np.float32)
    model.model.embed_tokens.weight.copy_(torch.from_numpy(Er))
print(f"largest weight error after fp16 scales/mins = {worst:.2e} of the weight range")

# teacher-forced word accuracy of the model rebuilt from the files, output limited to the allowed words
train = []
for q, a in load("train_clean_text.jsonl"):
    prompt = f"Instruction: {q}\nResponse:"
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    full = (tok(prompt + " " + a, add_special_tokens=False)["input_ids"] + [tok.eos_token_id])[:512]
    train.append((full, len(p)))
mask = torch.full((V,), float("-inf")); mask[ids] = 0.0
model = model.cuda(); mask = mask.cuda(); hit = 0; n = 0
with torch.no_grad():
    for full, plen in train:
        x = torch.tensor([full]).cuda()
        lg = model(input_ids=x).logits[0, :-1].float() + mask
        pred = lg.argmax(-1)[plen - 1:]; tgt = x[0, plen:]
        hit += (pred == tgt).sum().item(); n += tgt.numel()
print(f"model rebuilt from the exported files: word accuracy on the training answers = {hit / n:.2%} ({hit}/{n})")
json.dump({"layers": L, "hidden": D, "ffn": F, "heads": H, "kv_heads": KV, "group": GROUP, "allowed_words": N,
           "layer_stride": layer_stride, "layers_off": layers_off, "embed_off": embed_off, "embed_stride": embed_stride,
           "files": {fn: os.path.getsize(os.path.join(OUT, fn)) for fn in ("model.bin", "head.bin", "bpe.bin")}},
          open(os.path.join(OUT, "pack_info.json"), "w"), indent=1)

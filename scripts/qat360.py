"""4-bit quantisation-aware distillation of Ambika's smolFem (SmolLM2-360M based).

Teacher = smolFem as it is (bf16). Student = the same model whose layer weights are rounded to 4 bits (groups of GROUP, asymmetric,
straight-through gradient) in every forward pass. Loss = KL(teacher || student) on the next-word distribution at every position of
Ambika's question + answer texts, so the student learns to behave like smolFem, not like the dataset.

usage: python qat360.py [group=64] [epochs=6] [lr=2e-5]
Prints, every 2 epochs: agreement with the teacher's top-1 word, and the word accuracy on the training answers (output limited to the
words that occur in the answers, like the board; the teacher itself scores 71.82%).
"""
import json, math, os, random, re, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

GROUP = int(sys.argv[1]) if len(sys.argv) > 1 else 64
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
LR = float(sys.argv[3]) if len(sys.argv) > 3 else 2e-5
SRC = os.environ.get("SMOLFEM_DIR", "smolFem")   # the full smolFem model folder (Hugging Face format)
DATA = os.environ.get("DATA_DIR", "dataset")   # folder with the cleaned train and valid files
OUT = rf"smolFem360-qat-int4g{GROUP}"
BS, MAXLEN = 2, 320
random.seed(0); torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(SRC)


def load(fn):
    rows = []
    for l in open(os.path.join(DATA, fn), encoding="utf-8"):
        if l.strip():
            t = json.loads(l)["text"]
            m = re.match(r"^Instruction:\s*(.*?)\nResponse:\s*(.*)$", t, re.S)
            rows.append((m.group(1).strip(), m.group(2).strip()))
    return rows


train_qa, valid_qa = load("train_clean_text.jsonl"), load("valid_clean_text.jsonl")


def encode(q, a):
    prompt = f"Instruction: {q}\nResponse:"
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    full = (tok(prompt + " " + a, add_special_tokens=False)["input_ids"] + [tok.eos_token_id])[:MAXLEN]
    return full, len(p)


train = [encode(q, a) for q, a in train_qa]
valid = [encode(q, a) for q, a in valid_qa]
allowed = {tok.eos_token_id}
for q, a in train_qa + valid_qa:
    allowed.update(tok(" " + a, add_special_tokens=False)["input_ids"]); allowed.update(tok(a, add_special_tokens=False)["input_ids"])
ids_allowed = sorted(allowed)
V = len(tok)
amask = torch.full((V,), float("-inf")); amask[ids_allowed] = 0.0; amask = amask.cuda()


def batches(data, shuffle):
    idx = list(range(len(data)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), BS):
        chunk = [data[j] for j in idx[i:i + BS]]
        L = max(len(x[0]) for x in chunk)
        ids = torch.zeros((len(chunk), L), dtype=torch.long); att = torch.zeros((len(chunk), L), dtype=torch.long)
        pl = []
        for k, (x, p) in enumerate(chunk):
            ids[k, :len(x)] = torch.tensor(x); att[k, :len(x)] = 1; pl.append(p)
        yield ids.cuda(), att.cuda(), pl


def fq(w, group):
    o, i = w.shape
    wg = w.reshape(o, i // group, group)
    mn, mx = wg.amin(-1, keepdim=True), wg.amax(-1, keepdim=True)
    scale = (mx - mn).clamp(min=1e-8) / 15
    wq = (torch.round((wg - mn) / scale).clamp(0, 15) * scale + mn).reshape(o, i)
    return w + (wq - w).detach()


class QLinear(nn.Module):
    def __init__(self, lin):
        super().__init__()
        self.weight = lin.weight

    def forward(self, x):
        return F.linear(x, fq(self.weight, GROUP))


teacher = AutoModelForCausalLM.from_pretrained(SRC, dtype=torch.bfloat16).cuda().eval()
for p in teacher.parameters():
    p.requires_grad_(False)
student = AutoModelForCausalLM.from_pretrained(SRC, dtype=torch.float32).cuda()
student.gradient_checkpointing_enable()
student.config.use_cache = False
student.model.embed_tokens.weight.requires_grad_(False)
# the embedding table is deployed as int8 per row: let the student see the same table from the start
with torch.no_grad():
    e = student.model.embed_tokens.weight
    s = e.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 127
    e.copy_(torch.round(e / s).clamp(-127, 127) * s)
for ly in student.model.layers:
    for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
        setattr(ly.self_attn, n, QLinear(getattr(ly.self_attn, n)))
    for n in ("gate_proj", "up_proj", "down_proj"):
        setattr(ly.mlp, n, QLinear(getattr(ly.mlp, n)))
params = [p for p in student.parameters() if p.requires_grad]
print(f"trainable parameters: {sum(p.numel() for p in params) / 1e6:.0f} M | group {GROUP} | {EPOCHS} epochs | lr {LR}", flush=True)
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
steps_total = EPOCHS * math.ceil(len(train) / BS)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.05, 1 - s / steps_total))


def evaluate(data, tag):
    student.eval(); agree = n = hit = nw = 0
    with torch.no_grad():
        for ids, att, pl in batches(data, False):
            ls = student(input_ids=ids, attention_mask=att).logits[:, :-1].float()
            lt = teacher(input_ids=ids, attention_mask=att).logits[:, :-1].float()
            m = att[:, 1:].bool()
            agree += (ls.argmax(-1)[m] == lt.argmax(-1)[m]).sum().item(); n += m.sum().item()
            for k, p in enumerate(pl):
                L = int(att[k].sum())
                pred = (ls[k, p - 1:L - 1] + amask).argmax(-1); tgt = ids[k, p:L]
                hit += (pred == tgt).sum().item(); nw += tgt.numel()
    student.train()
    return agree / n, hit / nw


a, w = evaluate(train, "train")
print(f"before training: agreement with teacher {a:.2%} | word accuracy on training answers {w:.2%}", flush=True)
student.train(); t0 = time.time(); step = 0
for ep in range(1, EPOCHS + 1):
    tot = 0.0; cnt = 0
    for ids, att, pl in batches(train, True):
        with torch.no_grad():
            lt = teacher(input_ids=ids, attention_mask=att).logits[:, :-1].float()
        ls = student(input_ids=ids, attention_mask=att).logits[:, :-1].float()
        m = att[:, 1:].bool()
        loss = F.kl_div(F.log_softmax(ls[m], -1), F.log_softmax(lt[m], -1), log_target=True, reduction="batchmean")
        loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        tot += loss.item(); cnt += 1; step += 1
    msg = f"epoch {ep:2d} | KL {tot / cnt:.4f} | {time.time() - t0:.0f} s"
    if ep % 2 == 0 or ep == EPOCHS:
        a, w = evaluate(train, "train"); av, wv = evaluate(valid, "valid")
        msg += f" | train: agree {a:.2%}, words {w:.2%} | valid: agree {av:.2%}, words {wv:.2%}"
    print(msg, flush=True)

# save as a normal HF model whose layer weights are already the rounded (deployed) values
with torch.no_grad():
    for ly in student.model.layers:
        for mod in (ly.self_attn.q_proj, ly.self_attn.k_proj, ly.self_attn.v_proj, ly.self_attn.o_proj,
                    ly.mlp.gate_proj, ly.mlp.up_proj, ly.mlp.down_proj):
            mod.weight.copy_(fq(mod.weight, GROUP))
state = student.state_dict()
del student, opt, teacher; torch.cuda.empty_cache()
plain = AutoModelForCausalLM.from_pretrained(SRC, dtype=torch.float32)
plain.load_state_dict(state, strict=False)
plain.save_pretrained(OUT, safe_serialization=True); tok.save_pretrained(OUT)
print("saved", OUT, flush=True)

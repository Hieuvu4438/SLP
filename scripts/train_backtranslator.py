"""Pose -> text back-translators for SLP evaluation (B2) and reward (B1).

Two deliberately different architectures so that the reward model and the
evaluator do not share failure modes (see docs/proposal1/05_proposed_method.md, C3):

  --arch mbart : conv/transformer pose encoder -> mBART-large decoder (SOKE's pruned en/zh/de mBART)  [B1 reward]
  --arch qwen  : conv pose encoder -> prefix embeddings for Qwen3-0.6B (MaDiS's pruned LM), LoRA     [B2 evaluator]

Input: SOKE 133-D SMPL-X features (normalised). Trained on the TRAIN split of GT
poses; dev is used for model selection and for the GT ceiling. Never touches test.
Scoring of generated motion: --score_npz name->[T,133] (normalised features).

Run with the MaDiS venv:
  third_party/MaDiS/.venv/bin/python scripts/train_backtranslator.py --arch mbart --out outputs/bt/phoenix_mbart
"""

import argparse
import gzip
import json
import math
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

KEYS = ("smplx_root_pose", "smplx_body_pose", "smplx_lhand_pose",
        "smplx_rhand_pose", "smplx_jaw_pose", "smplx_shape", "smplx_expr")
MBART = "/home/haipd/SLP/third_party/SOKE/deps/mbart-h2s-csl-phoenix"
QWEN = "/home/haipd/SLP/third_party/MaDiS/hf_extract/qwen/Qwen3-0.6B-Base-en-zh-de"


def feats133(p):
    f = np.concatenate([p[k] for k in KEYS], 1)[:, 36:]
    return np.concatenate([f[:, :-20], f[:, -10:]], 1).astype(np.float32)


def load_generic(source, split, mean, std):
    """CSL-Daily / How2Sign via the shared loader (MaDiS crop, How2Sign at 24 fps)."""
    import sys as _s
    _s.path.insert(0, "/home/haipd/SLP/src")
    from slp.data.sign_smplx import LOADERS, madis_crop
    roots = {"csl": "CSL-Daily", "how2sign": "How2Sign"}
    out = []
    for info, path, fps in LOADERS[source](Path(f"/home/haipd/SLP/data/processed/madis/{roots[source]}"), split):
        f = feats133(pickle.load(open(path, "rb")))
        if fps is not None and fps > 24:
            f = f[np.linspace(0, len(f) - 1, int(24 * len(f) / fps), dtype=int)]
        if len(f) >= 8:
            out.append((info["name"], info["text"], madis_crop((f - mean) / std).astype(np.float32)))
    return out


LANG = {"phoenix": "de_DE", "csl": "zh_CN", "how2sign": "en_XX"}
BLEU_TOK = "13a"  # set to "zh" for CSL in main()


def load_phoenix(root, split, mean, std):
    ann = pickle.load(gzip.open(Path(root) / f"phoenix14t.{split}", "rb"))
    out = []
    for a in ann:
        p = Path(root) / "poses_video_level" / f"{a['name']}.pkl"
        if p.exists():
            f = feats133(pickle.load(open(p, "rb")))
            if len(f) >= 8:
                out.append((a["name"], a["text"], (f - mean) / std))
    return out


class PoseEncoder(nn.Module):
    """Part-aware embedding + temporal conv (x2 down) + transformer encoder."""

    def __init__(self, d_out, d=512, layers=4, max_len=512):
        super().__init__()
        self.body = nn.Linear(43, d // 4)
        self.lh = nn.Linear(45, 3 * d // 8)
        self.rh = nn.Linear(45, 3 * d // 8)
        self.conv = nn.Sequential(nn.Conv1d(d, d, 5, 2, 2), nn.GELU(), nn.Conv1d(d, d, 3, 1, 1))
        self.pos = nn.Parameter(torch.zeros(1, max_len, d))
        layer = nn.TransformerEncoderLayer(d, 8, 4 * d, 0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.proj = nn.Linear(d, d_out)

    def forward(self, x, mask):  # x [B,T,133], mask [B,T] True=valid
        h = torch.cat([self.body(torch.cat([x[..., :30], x[..., 120:]], -1)),
                       self.lh(x[..., 30:75]), self.rh(x[..., 75:120])], -1)
        h = self.conv(h.transpose(1, 2)).transpose(1, 2)
        m = mask[:, ::2][:, : h.shape[1]]
        h = h + self.pos[:, : h.shape[1]]
        h = self.enc(h, src_key_padding_mask=~m)
        return self.proj(h), m


class MBartBT(nn.Module):
    def __init__(self, lang="de_DE"):
        super().__init__()
        from transformers import MBartForConditionalGeneration, MBartTokenizer
        import pickle as pk
        self.tok = MBartTokenizer.from_pretrained(MBART, legacy=True)
        self.map = pk.load(open(f"{MBART}/map_ids.pkl", "rb"))  # tokenizer id -> pruned emb id
        self.inv = {v: k for k, v in self.map.items()}
        self.lm = MBartForConditionalGeneration.from_pretrained(MBART)
        self.lm.model.encoder = None  # pose encoder replaces the text encoder
        self.penc = PoseEncoder(self.lm.config.d_model)
        self.lang_code = lang
        self.lang = self.map[self.tok.convert_tokens_to_ids(lang)]

    def _ids(self, texts, device):
        enc = self.tok(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
        ids = enc.input_ids.apply_(lambda t: self.map.get(t, self.map[self.tok.unk_token_id]))
        return ids.to(device), enc.attention_mask.to(device)

    def forward(self, x, mask, texts):
        from transformers.modeling_outputs import BaseModelOutput
        h, m = self.penc(x, mask)
        ids, am = self._ids(texts, x.device)
        dec_in = torch.cat([torch.full_like(ids[:, :1], self.lang), ids[:, :-1]], 1)
        labels = ids.masked_fill(am == 0, -100)
        out = self.lm(encoder_outputs=BaseModelOutput(last_hidden_state=h), attention_mask=m.long(),
                      decoder_input_ids=dec_in, decoder_attention_mask=am, labels=labels)
        return out.loss, out.logits, labels

    @torch.no_grad()
    def generate(self, x, mask, beams=1):  # beam search with external encoder_outputs gave empty hyps
        from transformers.modeling_outputs import BaseModelOutput
        h, m = self.penc(x, mask)
        out = self.lm.generate(encoder_outputs=BaseModelOutput(last_hidden_state=h), attention_mask=m.long(),
                               decoder_start_token_id=self.lang, num_beams=beams, max_new_tokens=64,
                               eos_token_id=self.map[self.tok.eos_token_id],
                               pad_token_id=self.map[self.tok.pad_token_id])
        res = []
        for row in out.tolist():
            toks = [self.inv.get(t, self.tok.unk_token_id) for t in row]
            res.append(self.tok.decode(toks, skip_special_tokens=True).replace(self.lang_code, "").strip())
        return res


class QwenBT(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import AutoTokenizer, Qwen3ForCausalLM
        from peft import LoraConfig, get_peft_model
        import pickle as pk
        self.tok = AutoTokenizer.from_pretrained(QWEN)
        self.map = pk.load(open(f"{QWEN}/map_ids.pkl", "rb"))
        self.inv = {v: k for k, v in self.map.items()}
        lm = Qwen3ForCausalLM.from_pretrained(QWEN, torch_dtype=torch.bfloat16)
        self.lm = get_peft_model(lm, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
        self.penc = PoseEncoder(lm.config.hidden_size)
        self.eos = self.map[self.tok.eos_token_id]

    def _ids(self, texts, device):
        rows = [[self.map.get(t, self.eos) for t in self.tok(s).input_ids[:63]] + [self.eos] for s in texts]
        L = max(len(r) for r in rows)
        ids = torch.full((len(rows), L), self.eos, dtype=torch.long)
        am = torch.zeros((len(rows), L), dtype=torch.long)
        for i, r in enumerate(rows):
            ids[i, : len(r)] = torch.tensor(r); am[i, : len(r)] = 1
        return ids.to(device), am.to(device)

    def forward(self, x, mask, texts):
        h, m = self.penc(x, mask)
        ids, am = self._ids(texts, x.device)
        emb = self.lm.get_input_embeddings()(ids)
        inp = torch.cat([h.to(emb.dtype), emb], 1)
        att = torch.cat([m.long(), am], 1)
        labels = torch.cat([torch.full(m.shape, -100, device=x.device, dtype=torch.long),
                            ids.masked_fill(am == 0, -100)], 1)
        out = self.lm(inputs_embeds=inp, attention_mask=att, labels=labels)
        return out.loss, out.logits[:, h.shape[1] - 1:-1], labels[:, h.shape[1]:]

    @torch.no_grad()
    def generate(self, x, mask, beams=1):
        h, m = self.penc(x, mask)
        emb_layer = self.lm.get_input_embeddings()
        inp, att = h.to(emb_layer.weight.dtype), m.long()
        outs = [[] for _ in range(len(x))]
        done = torch.zeros(len(x), dtype=torch.bool, device=x.device)
        past = None
        cur = inp
        for _ in range(64):  # greedy decoding with KV cache
            o = self.lm(inputs_embeds=cur, attention_mask=att, past_key_values=past, use_cache=True)
            past = o.past_key_values
            nxt = o.logits[:, -1].float().argmax(-1)
            for i, t in enumerate(nxt.tolist()):
                if not done[i]:
                    if t == self.eos:
                        done[i] = True
                    else:
                        outs[i].append(t)
            if done.all():
                break
            cur = emb_layer(nxt[:, None])
            att = torch.cat([att, torch.ones_like(att[:, :1])], 1)
        return [self.tok.decode([self.inv.get(t, self.tok.eos_token_id) for t in r], skip_special_tokens=True).strip()
                for r in outs]


def batches(items, bs, shuffle):
    idx = list(range(len(items)))
    if shuffle:
        random.shuffle(idx)
    for s in range(0, len(idx), bs):
        chunk = [items[i] for i in idx[s:s + bs]]
        T = min(max(len(c[2]) for c in chunk), 400)
        x = torch.zeros(len(chunk), T, 133)
        m = torch.zeros(len(chunk), T, dtype=torch.bool)
        for i, (_, _, f) in enumerate(chunk):
            f = f[:T]
            x[i, : len(f)] = torch.from_numpy(f); m[i, : len(f)] = True
        yield [c[0] for c in chunk], [c[1] for c in chunk], x, m


def seq_logprob(model, x, m, texts):
    """Length-normalised log p(text | pose) per sample (the C3 reward / BT-likelihood metric)."""
    _, logits, labels = model(x, m, texts)
    lp = -F.cross_entropy(logits.float().transpose(1, 2), labels.clamp_min(0), reduction="none")
    valid = labels != -100
    return ((lp * valid).sum(1) / valid.sum(1).clamp_min(1)).tolist()


def evaluate(model, items, dev, bs=32):
    import sacrebleu
    model.eval()
    hyps, refs, lps = [], [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _, texts, x, m in batches(items, bs, False):
            x, m = x.to(dev), m.to(dev)
            hyps += model.generate(x, m)
            refs += texts
            lps += seq_logprob(model, x, m, texts)
    return {"bleu4": sacrebleu.corpus_bleu(hyps, [refs], tokenize=BLEU_TOK).score,
            "chrf": sacrebleu.corpus_chrf(hyps, [refs]).score,
            "bt_logprob": float(np.mean(lps)), "n": len(refs),
            "examples": list(zip(refs[:5], hyps[:5]))}


@torch.no_grad()
def bt_retrieval(model, items, dev, k=99, seed=0, chunk=100):
    """Sentence-discriminative BT: rank of the clip's own sentence among k fixed distractor
    sentences (other dev sentences) by length-normalised log p(text | pose). Returns R@1/R@5/R@10, mean rank."""
    rng = np.random.default_rng(seed)
    texts = [t for _, t, _ in items]
    uniq = sorted(set(texts))
    ranks = []
    model.eval()
    for i, (_, t, f) in enumerate(items):
        pool = [u for u in rng.choice(len(uniq), size=min(k + 1, len(uniq)), replace=False) if uniq[u] != t][:k]
        cands = [t] + [uniq[u] for u in pool]
        x = torch.from_numpy(np.asarray(f[:400], dtype=np.float32))[None].to(dev)
        m = torch.ones(1, x.shape[1], dtype=torch.bool, device=dev)
        lps = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for s0 in range(0, len(cands), chunk):
                c = cands[s0:s0 + chunk]
                lps += seq_logprob(model, x.expand(len(c), -1, -1), m.expand(len(c), -1), c)
        lps = np.array(lps)
        ranks.append(int((lps > lps[0]).sum()) + 1)
    ranks = np.array(ranks)
    return {"bt_R@1": float((ranks <= 1).mean() * 100), "bt_R@5": float((ranks <= 5).mean() * 100),
            "bt_R@10": float((ranks <= 10).mean() * 100), "bt_mean_rank": float(ranks.mean()), "k": k}


def score(a, model, val, dev, out):
    """Back-translate generated / calibration sequences of the dev split with a trained model."""
    model.load_state_dict(torch.load(out / "best.pt", map_location=dev))
    by_name = {Path(n).name: (n, t, f) for n, t, f in val}
    res = {}
    for spec in a.score:
        if spec.startswith("@"):
            lab = spec[1:]
            items = []
            for n, t, f in val:
                f = {"gt": f, "mean_pose": np.zeros_like(f), "gt_reversed": f[::-1].copy()}[lab]
                items.append((n, t, f))
        else:
            lab, path = spec.split("=", 1)
            p = Path(path)
            preds = ({q.stem: pickle.load(open(q, "rb"))["feats_rst"] for q in p.glob("*.pkl")} if p.is_dir()
                     else {k.split("__")[-1]: v for k, v in np.load(p).items() if not k.startswith("coarse::")})
            items = [(by_name[k][0], by_name[k][1], np.asarray(v, dtype=np.float32)) for k, v in preds.items() if k in by_name]
        r = evaluate(model, items, dev)
        if a.retrieval:
            r.update(bt_retrieval(model, items, dev))
        res[lab] = {k: v for k, v in r.items()}
        print(json.dumps({lab: {k: v for k, v in r.items() if k != "examples"}, "example": r["examples"][0]},
                         ensure_ascii=False), flush=True)
    json.dump(res, open(out / (a.score_name or "score.json"), "w"), indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["mbart", "qwen"], required=True)
    ap.add_argument("--source", default="phoenix", choices=["phoenix", "csl", "how2sign"])
    ap.add_argument("--root", default="/home/haipd/SLP/data/processed/madis/Phoenix_2014T")
    ap.add_argument("--stats", default="/home/haipd/SLP/third_party/SOKE/deps/stats")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lm_lr", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--limit", type=int, default=-1, help="debug: subsample train/dev")
    ap.add_argument("--score_name", default=None)
    ap.add_argument("--retrieval", action="store_true", help="score mode: add BT-retrieval R@k (k=99 distractors)")
    ap.add_argument("--score", nargs="*", default=None,
                    help="score mode: label=<MaDiS pred dir | npz> or @gt @mean_pose @gt_reversed; uses <out>/best.pt")
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda"
    torch.backends.cuda.enable_cudnn_sdp(False)  # torch 2.5 cuDNN-SDPA fails with padding masks
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    mean = torch.load(f"{a.stats}/mean.pt", weights_only=True)[36:]
    std = torch.load(f"{a.stats}/std.pt", weights_only=True)[36:]
    mean = torch.cat([mean[:-20], mean[-10:]]).float().numpy()
    std = torch.cat([std[:-20], std[-10:]]).float().numpy()
    global BLEU_TOK
    BLEU_TOK = "zh" if a.source == "csl" else "13a"
    if a.source == "phoenix":
        train = load_phoenix(a.root, "train", mean, std)
        val = load_phoenix(a.root, "dev", mean, std)
    else:
        train = load_generic(a.source, "train", mean, std)
        val = load_generic(a.source, "val", mean, std)
    if a.limit > 0:
        train, val = train[: a.limit], val[: max(a.limit // 8, 16)]
    print(f"train {len(train)} dev {len(val)}", flush=True)

    model = (MBartBT(LANG[a.source]) if a.arch == "mbart" else QwenBT()).to(dev)
    if a.score is not None:
        return score(a, model, val, dev, out)
    enc_params = list(model.penc.parameters())
    lm_params = [p for n, p in model.named_parameters() if not n.startswith("penc") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": a.lr}, {"params": lm_params, "lr": a.lm_lr}],
                            weight_decay=0.01)
    steps = a.epochs * math.ceil(len(train) / a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.lm_lr], total_steps=steps, pct_start=0.05)
    best, log = -1, []
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        tot = 0
        for _, texts, x, m in batches(train, a.bs, True):
            x, m = x.to(dev), m.to(dev)
            x = x + 0.02 * torch.randn_like(x)  # light pose-noise augmentation
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _, _ = model(x, m, texts)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item()
        msg = {"epoch": ep, "train_loss": tot / math.ceil(len(train) / a.bs), "min": (time.time() - t0) / 60}
        if ep % a.eval_every == 0 or ep == a.epochs:
            r = evaluate(model, val, dev)
            msg.update({k: v for k, v in r.items() if k != "examples"})
            msg["example"] = r["examples"][0]
            if r["bleu4"] > best:
                best = r["bleu4"]
                torch.save({k: v for k, v in model.state_dict().items()}, out / "best.pt")
                json.dump(r, open(out / "best_dev.json", "w"), indent=2, ensure_ascii=False)
        log.append(msg)
        print(json.dumps(msg, ensure_ascii=False), flush=True)
    json.dump(log, open(out / "log.json", "w"), indent=2)


if __name__ == "__main__":
    main()

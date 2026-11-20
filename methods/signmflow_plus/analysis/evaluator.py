"""M1: text <-> motion retrieval evaluator (docs/proposal7 04 §1.2, task C10).

A small contrastive model trained on the *train* split only, used to score how well generated
motions match their sentence (R-precision, Frechet distance), the semantic check that DTW and
the broken / saturated back-translation scores do not give.

    # train (GPU if available; a few GPU-minutes on Phoenix)
    python -m methods.signmflow_plus.analysis.evaluator train --data phoenix_smplx
    # calibration on val: GT >> GT of another sentence >> mean pose (required before use)
    python -m methods.signmflow_plus.analysis.evaluator calibrate --data phoenix_smplx

Design (deliberately independent of the generator):
  text   : character 1-3-gram TF-IDF (language-agnostic: German, Chinese, English), fitted on the
           train sentences -> linear -> MLP -> 256-D
  motion : normalised 133-D frames -> 1-D temporal convolutions -> mean+max pooling -> MLP -> 256-D
  loss   : symmetric InfoNCE with a learnable temperature; pairs with the identical sentence are
           removed from the negatives (Phoenix repeats sentences).
Default checkpoint: outputs/signmflow_plus/evaluator/<data>/evaluator.pt (TF-IDF vocabulary included).
"""

from __future__ import annotations

import argparse
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[3]


class MotionEncoder(nn.Module):
    def __init__(self, in_dim=133, hidden=256, out=256, layers=4):
        super().__init__()
        convs, d = [], in_dim
        for i in range(layers):
            convs += [nn.Conv1d(d, hidden, 5, padding=2, stride=2 if i % 2 else 1), nn.GELU()]
            d = hidden
        self.conv = nn.Sequential(*convs)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, out))

    def forward(self, x, mask):
        """x [B,T,133], mask [B,T] True = valid."""
        h = self.conv((x * mask[..., None]).transpose(1, 2)).transpose(1, 2)  # [B,T',H]
        m = F.interpolate(mask[:, None].float(), size=h.shape[1], mode="nearest")[:, 0] > 0
        mean = (h * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp_min(1)
        mx = h.masked_fill(~m[..., None], -1e4).max(1).values
        return self.head(torch.cat([mean, mx], -1))


class TextEncoder(nn.Module):
    def __init__(self, vocab: int, hidden=256, out=256):
        super().__init__()
        self.proj = nn.Linear(vocab, hidden)
        self.head = nn.Sequential(nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, out))

    def forward(self, x):
        return self.head(self.proj(x))


def contrastive_loss(t, m, text_ids, log_scale):
    """Symmetric InfoNCE; other samples with the same sentence are not used as negatives."""
    t, m = F.normalize(t, dim=-1), F.normalize(m, dim=-1)
    logits = log_scale.exp().clamp(max=100) * t @ m.T
    same = text_ids[:, None] == text_ids[None]
    eye = torch.eye(len(t), dtype=torch.bool, device=t.device)
    logits = logits.masked_fill(same & ~eye, float("-inf"))
    target = torch.arange(len(t), device=t.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


def _batch(seqs, max_len=None, rng=None):
    T = max(len(s) for s in seqs) if max_len is None else min(max_len, max(len(s) for s in seqs))
    x = np.zeros((len(seqs), T, seqs[0].shape[-1]), np.float32)
    mask = np.zeros((len(seqs), T), bool)
    for i, s in enumerate(seqs):
        if max_len is not None and len(s) > max_len:  # random crop (training only)
            a = int(rng.integers(0, len(s) - max_len + 1))
            s = s[a:a + max_len]
        x[i, :len(s)] = s
        mask[i, :len(s)] = True
    return torch.from_numpy(x), torch.from_numpy(mask)


class Evaluator:
    def __init__(self, vectorizer, text_enc: TextEncoder, motion_enc: MotionEncoder, device="cpu"):
        self.vec, self.text_enc, self.motion_enc = vectorizer, text_enc.to(device).eval(), motion_enc.to(device).eval()
        self.device = device

    def _tfidf(self, texts):
        x = self.vec.transform(texts).toarray().astype(np.float32)
        return torch.from_numpy(x).to(self.device)

    @torch.no_grad()
    def embed_text(self, texts, bs=256) -> np.ndarray:
        return np.concatenate([F.normalize(self.text_enc(self._tfidf(texts[a:a + bs])), dim=-1).cpu().numpy()
                               for a in range(0, len(texts), bs)])

    @torch.no_grad()
    def embed_motion(self, seqs, bs=64) -> np.ndarray:
        out = []
        for a in range(0, len(seqs), bs):
            x, m = _batch([np.asarray(s, np.float32) for s in seqs[a:a + bs]])
            out.append(F.normalize(self.motion_enc(x.to(self.device), m.to(self.device)), dim=-1).cpu().numpy())
        return np.concatenate(out)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"text": self.text_enc.state_dict(), "motion": self.motion_enc.state_dict(),
                    "vocab": len(self.vec.vocabulary_), "vectorizer": pickle.dumps(self.vec)}, path)

    @classmethod
    def load(cls, path, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        st = torch.load(path, map_location="cpu", weights_only=False)
        te, me = TextEncoder(st["vocab"]), MotionEncoder()
        te.load_state_dict(st["text"])
        me.load_state_dict(st["motion"])
        return cls(pickle.loads(st["vectorizer"]), te, me, device)


def train(texts, seqs, epochs=60, bs=128, lr=1e-3, max_len=200, seed=0, device="cpu", log=print) -> Evaluator:
    from sklearn.feature_extraction.text import TfidfVectorizer

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), sublinear_tf=True, max_features=20000)
    X = vec.fit_transform(texts)
    uniq = {t: i for i, t in enumerate(dict.fromkeys(texts))}
    tid = np.array([uniq[t] for t in texts])
    te, me = TextEncoder(X.shape[1]).to(device), MotionEncoder().to(device)
    log_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07), device=device))
    params = list(te.parameters()) + list(me.parameters()) + [log_scale]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    steps = epochs * (len(texts) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=max(1, steps))
    for ep in range(epochs):
        perm, tot = rng.permutation(len(texts)), 0.0
        for a in range(0, len(perm) - bs + 1, bs):
            idx = perm[a:a + bs]
            x, m = _batch([np.asarray(seqs[i], np.float32) for i in idx], max_len, rng)
            t = torch.from_numpy(X[idx].toarray().astype(np.float32)).to(device)
            loss = contrastive_loss(te(t), me(x.to(device), m.to(device)), torch.from_numpy(tid[idx]).to(device),
                                    log_scale)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach())
        log(f"evaluator epoch {ep + 1}/{epochs} loss {tot / max(1, len(perm) // bs):.4f}")
    return Evaluator(vec, te, me, device)


def calibrate(ev: Evaluator, texts, gts, seed=0) -> dict:
    """R-precision of GT, GT of another sentence (shuffled) and the mean pose on val."""
    from slp.evaluation.metrics import frechet_distance, r_precision

    rng = np.random.default_rng(seed)
    t = ev.embed_text(texts)
    gt = ev.embed_motion(gts)
    perm = rng.permutation(len(gts))
    shuffled = ev.embed_motion([gts[i] for i in perm])
    mean_pose = ev.embed_motion([np.zeros_like(g) for g in gts])
    res = {"gt": r_precision(t, gt), "gt_other_sentence": r_precision(t, shuffled),
           "mean_pose": r_precision(t, mean_pose)}
    res["fid_mean_pose_to_gt"] = frechet_distance(mean_pose, gt)
    res["separates_gt_from_shuffled"] = res["gt"]["R@1"] >= 3 * max(res["gt_other_sentence"]["R@1"], 1e-9)
    return res


def _split(data, which):
    from slp import config as config_lib
    from slp.data import load_dataset

    d = load_dataset(config_lib.compose("signmflow_plus", data))
    sp = {"train": d.train, "val": d.val}[which]  # never the test split
    return [x["text"] for x in sp.inputs], [np.asarray(sp.targets[i], np.float32) for i in range(len(sp.inputs))]


def main(argv=None):
    sys.path.insert(0, str(ROOT / "src"))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("action", choices=["train", "calibrate"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    ckpt = Path(a.ckpt or ROOT / "outputs/signmflow_plus/evaluator" / a.data / "evaluator.pt")
    if a.action == "train":
        texts, seqs = _split(a.data, "train")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        ev = train(texts, seqs, epochs=a.epochs, seed=a.seed, device=dev)
        ev.save(ckpt)
        print(f"-> {ckpt}")
    else:
        import json

        texts, gts = _split(a.data, "val")
        res = calibrate(Evaluator.load(ckpt), texts, gts, a.seed)
        print(json.dumps(res, indent=1))
        json.dump(res, open(ckpt.with_name("calibration_val.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

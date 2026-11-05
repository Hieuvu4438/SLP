"""Independent back-translation with released Uni-Sign pose-only SLT models (ICLR'25).

Run in the Uni-Sign venv:
  /home/haipd/Sign_Representations/.venv-unisign/bin/python scripts/unisign_bt_eval.py \
      --ckpt third_party/unisign_ckpt/openasl_pose_only_slt.pth --dataset OpenASL \
      --kp_dir outputs/cocowb/how2sign_val --labels gt mean_pose gt_reversed madis [--real_kp_dir DIR]

Inputs:
- <kp_dir>/<label>.npz: {name: [T,133,3]} from scripts/smplx_to_cocowb.py;
- <kp_dir>/texts.pkl: references;
- optional real RTMPose keypoints (<real_kp_dir>/<name>.pkl, Uni-Sign format) as the
  "real-video" ceiling.

The Uni-Sign code is imported read-only from the user's checkout. Metrics come from
Uni-Sign's own `translation_performance` (sacreBLEU 13a, ROUGE-L), with its CSL
character-level tokenisation. Independence: the OpenASL model never saw How2Sign;
CSL-Daily / How2Sign models saw the same train sentences but only real-video
RTMPose keypoints (no SMPL-X).
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

UNISIGN = "/home/haipd/Sign_Representations/third_party/Uni-Sign"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True, choices=["CSL_Daily", "How2Sign", "OpenASL"])
    ap.add_argument("--kp_dir", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--real_kp_dir", default=None)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kp_dir = Path(a.kp_dir).resolve(); out = Path(a.out).resolve(); ckpt = Path(a.ckpt).resolve()
    real_dir = Path(a.real_kp_dir).resolve() if a.real_kp_dir else None
    os.chdir(UNISIGN); sys.path.insert(0, UNISIGN)
    from types import SimpleNamespace
    from models import Uni_Sign
    from pose_preprocessing import load_part_kp
    from SLRT_metrics import sableu  # Uni-Sign's bundled sacreBLEU (13a); ROUGE pkg not installed in that venv

    # Uni-Sign defaults (utils.get_args_parser) for the fields the model reads; avoids the deepspeed import
    args = SimpleNamespace(dataset=a.dataset, hidden_dim=256, label_smoothing=0.2, rgb_support=False,
                           mt5_path=f"{UNISIGN}/pretrained_weight/mt5-base", language="")
    model = Uni_Sign(args=args)
    print(model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"], strict=True))
    model = model.cuda().eval()
    texts = pickle.load(open(kp_dir / "texts.pkl", "rb"))

    def to_parts(arr):  # arr [T,133,3] -> Uni-Sign part dict
        idx = np.linspace(0, len(arr) - 1, min(len(arr), a.max_len), dtype=int)
        sk = [arr[i:i + 1, :, :2] for i in idx]
        cf = [arr[i:i + 1, :, 2] for i in idx]
        return load_part_kp(sk, cf, force_ok=True)

    def run(samples):
        preds, refs = [], []
        for s in range(0, len(samples), a.bs):
            chunk = samples[s:s + a.bs]
            parts = [to_parts(x) for _, x in chunk]
            src = {}
            L = max(len(p["body"]) for p in parts)
            for key in parts[0]:
                src[key] = torch.stack([torch.cat([p[key], p[key][-1:].expand(L - len(p[key]), -1, -1)]) for p in parts]).cuda()
            lens = torch.tensor([len(p["body"]) for p in parts])
            src["attention_mask"] = (torch.arange(L)[None] < lens[:, None]).long().cuda()
            src["src_length_batch"] = lens
            with torch.no_grad():
                enc = model.encode(src)
                o = model.generate(enc, max_new_tokens=100, num_beams=a.beams)
            preds += model.mt5_tokenizer.batch_decode(o, skip_special_tokens=True)
            refs += [texts[n] for n, _ in chunk]
        if a.dataset == "CSL_Daily":
            preds = [" ".join(list(r.replace(" ", "").replace("\n", ""))) for r in preds]
            refs = [" ".join(list(r.replace("，", ",").replace("？", "?").replace(" ", ""))) for r in refs]
        bleu = sableu(references=refs, hypotheses=preds, tokenizer="13a")
        return {**{k: float(v) for k, v in bleu.items()}, "n": len(refs),
                "names": [n for n, _ in samples], "preds": preds, "refs": refs,
                "examples": list(zip(refs[:3], preds[:3]))}

    results = {}
    names_common = None
    for lab in a.labels:
        z = np.load(kp_dir / f"{lab}.npz")
        names_common = set(z.files) if names_common is None else names_common & set(z.files)
    if real_dir is not None:
        names_common = {n for n in names_common if (real_dir / f"{n}.pkl").exists()}
    names = sorted(names_common)[: a.limit] if a.limit > 0 else sorted(names_common)
    print("clips", len(names), flush=True)
    if real_dir is not None:
        samples = []
        for n in names:
            r = pickle.load(open(real_dir / f"{n}.pkl", "rb"))
            frames = []
            for k_, s_ in zip(r["keypoints"], r["scores"]):  # per frame: first person, or zero-confidence
                k_, s_ = np.asarray(k_, dtype=np.float32), np.asarray(s_, dtype=np.float32)
                if k_.ndim == 3 and len(k_) > 0 and k_.shape[1:] == (133, 2):
                    frames.append(np.concatenate([k_[0], s_.reshape(-1, 133)[0][:, None]], -1))
                else:
                    frames.append(np.zeros((133, 3), dtype=np.float32))
            samples.append((n, np.stack(frames)))
        results["real_rtmpose"] = run(samples)
        print("real_rtmpose", {k: v for k, v in results["real_rtmpose"].items() if k not in ("examples", "preds", "refs", "names")}, flush=True)
    for lab in a.labels:
        z = np.load(kp_dir / f"{lab}.npz")
        results[lab] = run([(n, z[n]) for n in names])
        print(lab, {k: v for k, v in results[lab].items() if k not in ("examples", "preds", "refs", "names")}, flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

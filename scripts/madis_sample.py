"""Stochastic N-sample generation from the released MaDiS (for FLARE C3: best-of-N / RAFT).

The released `MaskedDiffusionLanguageModel.generate` is greedy (argmax, body-only
confidence). Here it is replaced at runtime (third_party is not modified) by a variant that:
- keeps MaDiS's UTC grid schedule (every 4th → every 2nd → all positions) and 25 steps;
- samples each part token from softmax(logits / τ);
- ranks unmasking by the mean of the three part confidences (body, lhand, rhand).
With --temp 0 and --conf body it reduces to the released greedy decoder.

  cd third_party/MaDiS && .venv/bin/python /home/haipd/SLP/scripts/madis_sample.py \
      --cfg /home/haipd/SLP/configs/madis/madis_how2sign.dev.yaml --split val --n 4 --temp 1.0 \
      --out /home/haipd/SLP/outputs/madis_samples/how2sign_val_t1
Output: <out>/s<k>/<name>.pkl with feats_rst (normalised 133-D) and tokens, as in madis_generate.py.
"""

import argparse
import pickle
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")


INIT = {"codes": None, "keep": 0.5, "rng": None}  # retrieval-initialised mode (batch size 1)


def make_sampler(temp, conf_mode):
    from mGPT.archs import masked_diffusion as md

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, max_length=100, **_):
        B = input_ids.shape[0]
        dev = input_ids.device
        body = torch.full((B, max_length), self.mask_idx, dtype=torch.long, device=dev)
        lhand, rhand = body.clone(), body.clone()
        if INIT["codes"] is not None:  # C2: start from retrieved-clip tokens, re-mask a (1-keep) fraction
            assert B == 1, "retrieval-initialised mode needs batch size 1"
            c = torch.as_tensor(INIT["codes"][:max_length], device=dev)
            L = len(c)
            for arr, part, col in ((body, "body", 0), (lhand, "lhand", 1), (rhand, "rhand", 2)):
                ids = torch.as_tensor(self.id_dict[part], device=dev)
                arr[0, :L] = ids[c[:, col]]
                arr[0, L:] = self.eos_idx
            remask = torch.from_numpy(INIT["rng"].random(L) >= INIT["keep"]).to(dev)
            for arr in (body, lhand, rhand):
                arr[0, :L][remask] = self.mask_idx
        pos = torch.arange(max_length, device=dev)
        grids = [(pos % 4 == 0).expand(B, -1), (pos % 2 == 0).expand(B, -1), torch.ones(B, max_length, dtype=torch.bool, device=dev)]
        ts = torch.linspace(1, 1e-5, md.DIFFUSION_STEPS + 1, device=dev)
        for step in range(md.DIFFUSION_STEPS):
            if step <= int((1.0 - md.UTC_PIVOTS[0]) * md.DIFFUSION_STEPS):
                grid = grids[0]
            elif step <= int((1.0 - md.UTC_PIVOTS[1]) * md.DIFFUSION_STEPS):
                grid = grids[1]
            else:
                grid = grids[2]
            masked = body == self.mask_idx
            if not masked.any():
                break
            lb, ll, lr, _ = self._get_logits(input_ids, torch.stack([body, lhand, rhand], -1), attention_mask)
            preds, confs = [], []
            for lg in (lb[masked], ll[masked], lr[masked]):
                lg = lg.float()
                if temp > 0:
                    probs = F.softmax(lg / temp, -1)
                    p = torch.multinomial(probs, 1)[:, 0]
                else:
                    probs = F.softmax(lg, -1)
                    p = lg.argmax(-1)
                preds.append(p)
                confs.append(F.softmax(lg, -1).gather(-1, p[:, None])[:, 0])
            conf = confs[0] if conf_mode == "body" else torch.stack(confs).mean(0)
            conf = conf.masked_fill(~grid[masked], float("-inf")).reshape(B, -1)
            per = masked.sum() // B
            k = int(per * (1 - ts[step + 1] / ts[step])) if step < md.DIFFUSION_STEPS - 1 else int(per)
            if k == 0:
                continue
            sel = torch.topk(conf, k).indices + torch.arange(0, masked.sum(), per, device=dev)[:, None]
            for arr, p in zip((body, lhand, rhand), preds):
                nxt = torch.full_like(p, self.mask_idx)
                nxt[sel] = p[sel]
                arr[masked] = nxt
        for i in range(B):
            eos = (body[i] == self.eos_idx).nonzero(as_tuple=False)
            if eos.numel():
                e = int(eos[0, 0])
                body[i, e:] = lhand[i, e:] = rhand[i, e:] = self.eos_idx
        return {"outputs_re": body, "outputs_hand": lhand, "outputs_rhand": rhand}

    md.MaskedDiffusionLanguageModel.generate = generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--conf", choices=["body", "mean"], default="mean")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init_npz", default=None, help="retrieved motion {split__name: feats} to initialise from (C2)")
    ap.add_argument("--keep", type=float, default=0.5, help="fraction of retrieved tokens kept unmasked")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    make_sampler(a.temp, a.conf)
    sys.argv = [sys.argv[0], "--cfg", a.cfg, "--nodebug", f"TEST.SPLIT={a.split}", f"TEST.BATCH_SIZE={a.bs}"]
    from mGPT.config import parse_args
    from mGPT.data.build_data import build_data
    from mGPT.models.build_model import build_model
    from mGPT.utils.load_checkpoint import load_pretrained
    from torch.utils.data import DataLoader
    cfg = parse_args(phase="test")
    dm = build_data(cfg, phase="test")
    model = build_model(cfg, dm)
    load_pretrained(cfg, model, None, phase="test")
    model = model.cuda().eval()
    ds = dm.test_dataset
    ds.tasks = ds.tasks[:1]
    if a.limit > 0:
        ds.samples = ds.samples[: a.limit]
    init = None
    if a.init_npz:
        import numpy as np
        z = np.load(a.init_npz)
        init = {k.split("__")[-1]: z[k] for k in z.files}
        a.bs = 1
        INIT["keep"] = a.keep
        INIT["rng"] = np.random.default_rng(a.seed)

    def encode_codes(f):
        x = torch.from_numpy(f[: (len(f) // 4) * 4])[None].cuda()
        cb, _ = model.vae.encode(torch.cat([x[..., :30], x[..., 120:]], -1))
        cl, _ = model.hand_vae.encode(x[..., 30:75])
        cr, _ = model.rhand_vae.encode(x[..., 75:120])
        return torch.stack([cb[0], cl[0], cr[0]], -1).cpu().numpy()
    for k in range(a.n):
        torch.manual_seed(a.seed + k)
        out = Path(a.out) / f"s{k}"
        out.mkdir(parents=True, exist_ok=True)
        dl = DataLoader(ds, batch_size=a.bs, shuffle=False, num_workers=4, collate_fn=dm.dataloader_options["collate_fn"])
        with torch.no_grad():
            for batch in dl:
                if all((out / f"{Path(nm).name}.pkl").exists() for nm in batch["name"]):
                    continue
                for key in ("motion", "m_tokens"):
                    batch[key] = batch[key].cuda()
                if init is not None:
                    INIT["codes"] = encode_codes(init[Path(batch["name"][0]).name])
                r = model.generate_forward(batch)
                for i, nm in enumerate(batch["name"]):
                    L = int(r["lengths_rst"][i])
                    with open(out / f"{Path(nm).name}.pkl", "wb") as h:
                        pickle.dump({"feats_rst": r["m_rst"][i, :L].cpu().numpy(), "text": batch["text"][i]}, h)
        print("sample", k, "done", flush=True)


if __name__ == "__main__":
    main()

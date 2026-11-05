"""Generate MaDiS outputs for any split (incl. TRAIN) with one instruction template.

Needed for FLARE C1 v1 ("self-generated" refiner inputs) and C3 (RAFT / GRPO data).
Uses the released code paths unchanged (datamodule, MaDiS.generate_forward); only the
train-split template multiplication is disabled (dataset.tasks[:1]).

  cd third_party/MaDiS && .venv/bin/python /home/haipd/SLP/scripts/madis_generate.py \
      --cfg /home/haipd/SLP/configs/madis/madis_phoenix.dev.yaml --split train \
      --out /home/haipd/SLP/outputs/madis_gen/phoenix_train
Output: <out>/<name>.pkl with feats_rst (normalised 133-D), body/lhand/rhand tokens, text.
"""

import argparse
import pickle
import sys
from pathlib import Path

import torch

sys.path.insert(0, ".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=-1)
    a = ap.parse_args()
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
    dl = DataLoader(ds, batch_size=a.bs, shuffle=False, num_workers=4, collate_fn=dm.dataloader_options["collate_fn"])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    n = 0
    with torch.no_grad():
        for batch in dl:
            todo = [i for i, nm in enumerate(batch["name"]) if not (out / f"{Path(nm).name}.pkl").exists()]
            if not todo:
                continue
            for k in ("motion", "m_tokens"):
                batch[k] = batch[k].cuda()
            r = model.generate_forward(batch)
            for i, nm in enumerate(batch["name"]):
                L = int(r["lengths_rst"][i])
                with open(out / f"{Path(nm).name}.pkl", "wb") as h:
                    pickle.dump({"feats_rst": r["m_rst"][i, :L].cpu().numpy(),
                                 "body": r["body_tokens"][i].cpu().numpy(),
                                 "lhand": r["lhand_tokens"][i].cpu().numpy(),
                                 "rhand": r["rhand_tokens"][i].cpu().numpy(),
                                 "text": batch["text"][i]}, h)
            n += len(batch["name"])
            if n % 400 < a.bs:
                print("generated", n, flush=True)
    print("done", n, flush=True)


if __name__ == "__main__":
    main()

---
status: exploring   # exploring | promising | final | abandoned
created: 2026-11-24
---

# SignRAE: MoRAE representation autoencoders for sign language production

Text-to-sign (Phoenix-2014T, CSL-Daily, How2Sign; SOKE SMPL-X 133-D) generated in the compact
latent of a self-supervised **Sign-JEPA** encoder, following
[MoRAE](https://arxiv.org/abs/2607.29180) (`third_party/MoRAE`, inference-only release). The full
design, the paper-vs-ours decisions, the experiment plan and the risks are in
[PLAN.md](PLAN.md) (Vietnamese).

## Hypothesis

A latent learned by *predicting* hidden joint trajectories keeps cross-joint coordination
(handshape, arm configuration) on its high-variance directions and pushes joint-specific residuals
(including SMPL-X fitting jitter) into the low-variance tail. A flow transformer recovers
high-variance directions most accurately. So SignRAE should generate more coherent hands and arms
than the same DiT on a reconstruction latent or on raw poses. Paper evidence (HumanML3D FID):
prediction 0.0253 vs AE 0.0349 vs raw 0.25.

## Method

1. **Sign-JEPA** (`jepa.py`): 8×768 RoPE encoder, 6×384 predictor, and an EMA teacher
   (momentum 0.996→1).
   - Input is the 133-D pose plus per-group velocity.
   - Masking hides 30% of the 42 groups (41 joints plus expression) across all frames. A masked
     group can be a joint, a kinematic chain (a finger, an arm, the head) or a whole hand.
   - Loss: smooth-L1 between the predictor output and LN(teacher).
2. **Compact latent** (`autoencoder.py`): one 3072-wide attention layer turns
   H = LN(Σ last 4 teacher layers) into a diagonal Gaussian with d = 64.
   - A 4×512 RoPE decoder maps it straight back to the pose.
   - Loss: MSE + 1e-4 KL. The checkpoint with the lowest validation MSE is kept.
   - Controls: `latent.kind` = `ae` / `vae` / `raw`.
3. **Latent flow** (`dit.py`, `flow.py`): DiT-M (12×512, adaLN-zero on time + pooled mBART +
   language) plus token cross-attention to frozen SOKE mBART.
   - Linear-path velocity MSE in per-channel standardised coordinates, 10% text dropout, EMA 0.9999.
   - A duration head predicts log T.
   - Sampling: RK4 with 5 steps and CFG.

Running all three stages on a single dataset is self-contained (`fit`). The recommended route is
to pretrain stages 1–2 once on the **union of the three train splits** (`pretrain.py`), then
train only the flow for each dataset (`method.init_ae`).

## How to run (from the SLP root)

```bash
# 1-2. Self-supervised stages on the union of train splits (no text; val only for AE selection)
python -m methods.morae_slp.pretrain --stage jepa \
  --datasets phoenix_smplx,csl_smplx,how2sign_smplx_soke --out outputs/morae_slp/pretrain/jepa_union
python -m methods.morae_slp.pretrain --stage ae \
  --datasets phoenix_smplx,csl_smplx,how2sign_smplx_soke --out outputs/morae_slp/pretrain/ae_union_jepa_d64 \
  --init-jepa outputs/morae_slp/pretrain/jepa_union/jepa.pt
# Representation gate (spectrum, C/P loadings, reconstruction ceiling, gloss probe)
python -m methods.morae_slp.diagnostics --data phoenix_smplx \
  --ae outputs/morae_slp/pretrain/ae_union_jepa_d64/ae.pt --spectrum --recon 519 --probe --out g1.json
# 3. Flow on one dataset, scored on dev
python scripts/run.py --method morae_slp --data phoenix_smplx --name morae_phoenix --seeds 0 \
  method.init_ae=outputs/morae_slp/pretrain/ae_union_jepa_d64/ae.pt
# Sampler selection on dev (guidance, steps, noise temperature)
python -m methods.morae_slp.sweep --run outputs/runs/morae_phoenix/morae_slp/<run_id> \
  --cfg 1.5,2.5,3.5,4.5,6.5 --steps 5,8 --temperature 1.0,0.8
```

`configs/experiments/morae_slp/run.sh {g0,g1,g2,sweep,ablations,scale}` scripts the phases of
PLAN.md §7. It runs one process at a time, seed 0, on dev.

- **Other datasets:** use `--data csl_smplx` or `--data how2sign_smplx_soke` (the SOKE How2Sign
  protocol).
- **CPU smoke run:** `method.device=cpu method.debug_limit=8 method.{jepa,ae,flow}.updates=2`
  with small widths. `method.text.backend=bytes` avoids loading mBART.
- **Tuning budget:** every arm (jepa / ae / raw latent) gets the same sampler grid
  {w ∈ 1.5…6.5, steps ∈ {5, 8}, τ ∈ {1.0, 0.8, 0.6}} and the same number of updates.

## Outputs

- `checkpoints/jepa.pt` (when stage 1 runs): the Sign-JEPA online encoder, predictor and teacher.
- `checkpoints/ae.pt`: encoder + compressor + decoder + flow statistics for this train split.
  It can be reused with `method.init_ae`.
- `checkpoints/model.pt`: the EMA generator (DiT, duration head) + decoder + statistics. It is
  restored with `Method.load(run_dir / "checkpoints")`. `model_stepXXXXXX.pt` are the optional
  `flow.snapshot_steps`.
- `predictions.npz` contains normalised `[T, 133]` arrays by clip name. `metrics.json` holds the
  SOKE DTW-JPE / DTW-PA-JPE and kinematic ratios from the shared evaluator.
  `metrics_history.jsonl` holds the stage losses (`jepa/val_target_std` monitors collapse).

## Verification

- `pytest tests/morae_slp` covers:
  - group and unit partitions; masks hide whole units;
  - velocity input and padding invariance of every encoder and decoder;
  - JEPA masking, gradient flow and EMA;
  - all four latent kinds;
  - samplers against the Gaussian reference flow of the paper's Eq. (S7);
  - DiT zero-init and CFG null text;
  - fit → predict → export → reload, and reuse via `init_ae` with locked durations;
  - sampler knobs; diagnostics.
- The shared helpers `latent_spectrum`, `directional_loadings` and `multilabel_average_precision`
  in `slp.evaluation.metrics` have their own tests.
- CPU smoke runs (2 updates per stage, tiny widths) on real data passed through every entry point:
  - `fit`/`predict` + SOKE scoring on Phoenix with the real mBART;
  - `pretrain` on the 56,429-clip union of the three train splits;
  - `sweep`;
  - `diagnostics`.
- An independent code review (2026-11-24) found no serious bugs. Its four minor findings were fixed,
  each with a regression test:
  - `part_mse` now equals the plain MSE when all part weights are equal;
  - the duration head runs in FP32;
  - the flow refuses an uncalibrated `init_ae` when `recalibrate=false`;
  - teacher features are summed in FP32.
- First full-size run finished 2026-11-25 (see Results); flow val loss bottomed at ~26k updates and rose afterwards (overfitting on Phoenix).

## Results

| Experiment | Data | Seeds | JPE body / hand | PA body(ua) / hand | speed ratio b / h | Run dir |
|---|---|---|---|---|---|---|
| morae_phoenix_main = v1 control (jepa d64, flow 60k last EMA, rk4-5, w=4.5) | Phoenix dev (519) | 0 | 5.55 / 7.35 | 4.89 / 1.50 | 1.02 / 1.09 | outputs/runs/morae_phoenix_main/morae_slp/seed0_20261125-062634-648119 |

## Notes / log

- 2026-11-25: **v1 diagnosis and v2 plan** in [PLAN_V2.md](PLAN_V2.md) (Vietnamese).
  - The v1 flow memorised the Phoenix sentences. On train clips it reaches JPE 2.24 / 3.26. On dev, the
    latent flow loss *with* text is worse than with null text (1.31 vs 0.94 at 60k; +0.011 gain at 20k).
  - DTW-JPE rewards shrinkage toward the mean.
    - A static Phoenix-mean pose scores body 4.76 on 256 dev clips.
    - The replica moves at 0.66 of GT speed.
    - At a matched shrinkage (w = 1, τ = 0.3), v1 is about level with the replica: 5.19 / 6.87 · 4.57 / 1.36
      vs 5.11 / 6.84 · 4.53 / 1.35.
  - v2 (`python -m methods.morae_slp.job --plan v2`):
    - pretrain the flow on the 56k-sentence union with a fine-tuned mBART, text-token dropout and
      best-val selection, then fine-tune per dataset;
    - sweeps report a single-sample (natural) and a K-sample mean (point estimate) operating point.
  - New keys: `init_flow`, `text.finetune`, `text.lr_scale`, `flow.token_drop`, `sampler.ensemble`.
    New sweep flags: `--ensemble --data --lengths --subset --device`. All defaults reproduce v1.

- 2026-11-24: created. MoRAE ships inference code only, so the training code is ours, written
  from Table S14.
  - The released checkpoint uses DiT-XL, a two-stage decoder and Euler 24 steps. We follow the
    paper's main configuration instead: DiT-M, a single-stage decoder and RK4-5.
  - Paper-silent choices are listed in PLAN.md §2.3.

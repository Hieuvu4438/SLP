# Third-party notices and provenance

The implementation under `src/cprflow/` was written for the supplied CPR-Flow implementation specification. No SOKE/SignFlow generator, evaluator, fitting dataset, licensed SMPL-X asset, or upstream pretrained signing-generator weights are vendored into this package. Architecture/equation references are distinct from copied implementation code.

Actual resolved revisions and per-file SHA256 values are in `sources.lock.json`. The local audit downloaded reference source snapshots into ignored `/source_snapshots/`; these snapshots are not part of the distributable package or source archive of this implementation.

| Reference / dependency | Role | License / conditions checked in the locked source |
|---|---|---|
| [SignSparK](https://github.com/JianHe0628/SignSparK) | Read DATA.md, LMDB schema, row6D/native-left convention, downloader and visualization assembly; independently written adapter | Apache-2.0 at root. Preserve source dataset/model terms separately. |
| [MARDM](https://github.com/neu-3dvi/MARDM) | Read continuous temporal AE implementation pattern; AE layers here follow the supplied exact specification | MIT, `LICENSE.txt`. No HumanML3D data/evaluator included. |
| [DiT](https://github.com/facebookresearch/DiT) | Read adaLN-Zero and zero-output initialization pattern | Locked root `LICENSE.txt`: CC BY-NC4.0. No image-tokenization implementation copied. |
| [SMPL-X](https://github.com/vchoutas/smplx) | Runtime dependency; matrix-input `SMPLXLayer`, LBS and native joint names | Custom non-commercial scientific research license. Library/model licenses apply. Model files must be acquired separately from [the official site](https://smpl-x.is.tue.mpg.de/). |
| [google/mt5-base](https://huggingface.co/google/mt5-base) | Frozen encoder/tokenizer; resolved HF revisions in lock | Model card/license and Transformers dependency terms apply; weights are not distributed here. |
| [SignSparK pose release](https://huggingface.co/datasets/LionelLow/SignSparK_data) | Initial three-corpus reconstructed pose schema/source | Dataset card records source dataset restrictions; access record is required. Pose release does not grant rights to the source corpora. |

Source-corpus access: [CSL-Daily](https://ustc-slr.github.io/datasets/2021_csl_daily/), [How2Sign](https://how2sign.github.io/), [PHOENIX-2014T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/). No corpus was downloaded during this implementation session.

Mathematical references: [Flow Matching](https://arxiv.org/abs/2210.02747), [Rectified Flow](https://arxiv.org/abs/2209.03003), [DiT](https://arxiv.org/abs/2212.09748), [DDPM](https://arxiv.org/abs/2006.11239), [DDIM](https://arxiv.org/abs/2010.02502), and the supplied proposal/specification. Relation co-generation and semantic protection remain research hypotheses; implementing them is not a novelty or linguistic-quality certification.

SOKE and SignFlow are not implementation dependencies. No code is imported from other `methods/` folders. Meta `flow_matching` is not required or bundled, and is not represented here as MIT-licensed.

Dependency inventory is the actual environment snapshot in `requirements.lock.txt`; package ranges are in `pyproject.toml`. Each dependency retains its own license. This file does not select a license for the user's new implementation or authorize redistribution of restricted assets.

# data/

- `raw/` - original downloaded data, **never modified** (gitignored).
- `processed/` - outputs of preprocessing scripts (gitignored, reproducible).

Document every dataset here: source URL, version/date, license, download
command, preprocessing command, and checksum.

| Dataset | Source | Version | License | Download / preprocess |
|---|---|---|---|---|
| Phoenix-2014T SMPL-X (SOKE fittings) | SOKE project page → Google Drive `1nVaMRx4z53OiZaG9EKh_JpBY6oBmqm1-` (`phoenix_poses.zip`, 2.0 GB) | SOKE release, fetched ≤2026-11-03 | research only (RWTH-PHOENIX-2014-T terms) | raw: `data/raw/soke/Phoenix_2014T/{train,dev,test}/<name>/imagesXXXX.pkl` + `phoenix14t.{train,dev,test}`; pack: `python scripts/madis_build_phoenix.py` → `data/processed/madis/Phoenix_2014T/poses_video_level/<split>/<name>.pkl` (7096/519/642) |
| CSL-Daily SMPL-X (SOKE fittings) | Dropbox `scl/fo/3lwcdq60q8yg3foj6ulh0` → `CSL-Daily-Fittings.zip` (2,887,684,111 B) | SOKE release 2024-11-25 | research only (CSL-Daily terms) | raw: `data/raw/soke/_downloads/{CSL-Daily-Fittings.zip,csl_fit/}` (inner `csl-daily_pose.zip`, 2.48 M frame pkl; `csl_mean/std.pt` are CSL-only stats, differ from SOKE `deps/stats`); pack (streams zip, no extraction): `python scripts/madis_build_csl.py` → `data/processed/madis/CSL-Daily/poses_video_level/<name>.pkl`. Split files identical to `data/raw/soke/CSL-Daily/csl_clean.*` (minus `src` key) |
| How2Sign SMPL-X (SOKE / Neural-Sign-Actors fittings) | Dropbox `scl/fo/nc4qxrwqz1ze2b4dy5wce` → `Neural-Sign-Actors.zip` (7,250,855,933 B) | SOKE release | research only (How2Sign CC BY-NC 4.0) | raw: `data/raw/soke/_downloads/Neural-Sign-Actors.zip`; packing script TBD |
| MaDiS pre-extracted sign tokens | HF `2000zrl/MaDiS/TOKENS_h2s_csl_phoenix.zip` | 2026 release | research only | `third_party/MaDiS/hf_extract/tokens/` (phoenix 7092/519/642, csl 20652, how2sign 34709) |

### Loaders (`src/slp/data/sign_smplx.py`)

`phoenix_smplx`, `csl_smplx`, `how2sign_smplx` (configs in `configs/data/`) read the packed
data above with the **official splits** (val = Phoenix `dev`). Sizes: Phoenix 7,092 / 519 / 642;
CSL-Daily 18,399 / 1,077 / 1,176; How2Sign 30,965 / 1,724 / 2,318. Targets are normalised
133-D features (SOKE `deps/stats`) with the MaDiS crop rule; How2Sign is subsampled to 24 fps.

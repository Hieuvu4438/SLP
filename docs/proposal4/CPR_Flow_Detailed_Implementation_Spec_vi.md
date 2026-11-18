# CPR-Flow: đặc tả implementation cho AI coding agent

**Phiên bản:** implementation-spec v1.0, 10/11/2026. **Phạm vi:** CSL-Daily, How2Sign, PHOENIX-2014T. **Deliverable yêu cầu từ coding agent:** một repository PyTorch có thể preprocess, train, resume, sample và evaluate; không chỉ notebook/demo. Tài liệu này tự chứa specification; không cần đọc báo cáo cũ để bắt đầu.

Đây là thiết kế được chốt để implement và kiểm chứng, **không phải code đã chạy hoặc kết quả benchmark đã có**. Các lệnh `python -m cprflow...` dưới đây là **CLI phải được coding agent tạo**, không phải lệnh đã tồn tại upstream. Paths trong code blocks là relative với root của repository mới. Số updates và weights là defaults của thiết kế, không phải hyperparameters đã tune.

## 0. Chỉ thị giao việc cho coding agent

> Implement tuần tự milestones M0–M9 trong tài liệu. Đọc specification trước khi sửa code. Tạo `IMPLEMENTATION_STATUS.md` với checklist, kết quả commands, failures và artifact paths. Không thay flow objective, motion profile, dataset split hoặc inference protocol một cách âm thầm. Hoàn tất data/geometry tests, AE và plain-CFM baseline trước relational/semantic extensions. Không claim trained/reproduced khi chỉ chạy unit tests. Không thay dữ liệu thiếu bằng synthetic data rồi báo training thật. Dùng synthetic fixtures chỉ cho tests, có nhãn `synthetic=true`.

Nếu thiếu assets, ghi đúng missing path, nguồn tải và milestone bị chặn; vẫn hoàn thiện các modules không phụ thuộc asset đó. Không cần xin lại xác nhận cho các lựa chọn đã được chốt ở đây. Không tự gửi email, xin quyền dataset hoặc đăng tải model/dữ liệu. Không tải corpus ngoài ba dataset đã chọn để “cải thiện” kết quả.

### 0.1 Quyết định được khóa

| Hạng mục | Quyết định |
| --- | --- |
| Training chính | Joint multilingual trên ba train splits; language ID riêng CSL/ASL/DGS |
| Controls | Ba monolingual runs và matched diffusion/CFM/relational variants |
| Task | Text-to-3D-motion offline; predicted duration ở inference |
| Input conditioning | Text, sign-language ID, output length; không gloss/keyframes/target relation |
| Motion profile | SignSparK-derived 296D: body60 + left90 + right90 + face56 |
| Hand convention | Native SMPL-X left/right; **không flip trái** trong canonical representation |
| Text encoder | Frozen `google/mt5-base`, encoder only, 768D token features, revision pinned |
| Motion AE | Continuous deterministic part AE, temporal stride2, latent128/part |
| Generator | Typed-stream DiT, hidden384, 8 blocks, 6 heads, FFN1536 |
| Main objective | Linear conditional flow matching, time0=noise, time1=data |
| Relation extension | 33D proxy cùng temporal stride2, co-generated, không oracle |
| First complete trainable version | Plain CFM + duration, không teacher/guard dependency |
| Full research version | Relation losses, lag loss, calibrated teacher và guard |
| Default FPS controlled track | 30, chỉ sau khi xác minh feature sampling FPS |
| Evaluation | Mỗi dataset riêng; no cross-language pooled BLEU; predicted/oracle length tách biệt |

### 0.2 Điều chỉnh kỹ thuật so với proposal

Tài liệu này đóng các chỗ proposal còn để mở: canonical order, native-left policy, row-based rot6D, exact AE layers, token positions/masks, per-clip loss reduction, dataset-balanced stats/sampling, DDP reductions và guard rollback. Không share hand encoder bằng reflection trong v1. `decoder_cross_part_fusion` dùng một MLP cụ thể ở latent resolution. Auxiliary losses được tính định kỳ với hệ số bù tần suất được chỉ định. Không dùng KL/VQ, OT, multiscale flow, LoRA, RL hoặc one-step distillation trong training mặc định.

## 1. Nguồn nào dùng cho phần nào?

Các mã `[Rxx]`, `[Dxx]`, `[Pxx]`, `[Txx]` trỏ tới đường dẫn đầy đủ ở cuối tệp. Nguồn có trạng thái kiểm tra tại ngày nêu trên; coding agent phải lưu revision thật vào lock trước khi tái sử dụng.

| Module | Nguồn cần đọc | Cách dùng |
| --- | --- | --- |
| LMDB adapter | R01: `DATA.md`, `pose_datasets_lmdb.py`, `tools/download_data.py` | Schema/format; viết adapter không truncate utterance |
| Native SMPL-X assembly | R01: `tools/visualize.py`; R03: `SMPLXLayer`, joint names | Kiểm tra mapping; **không copy đường SciPy/NumPy vào differentiable losses** |
| Continuous AE | R02: `models/AE.py` | Tham khảo residual temporal AE; profile/layers trong spec này là authoritative |
| Flow equations | P01, P02 | Tự implement path đơn giản; không cần flow library để chạy baseline |
| DiT conditioning | P03, P04; R01 model chỉ để đối chiếu | adaLN-Zero và attention; không chuyển nguyên model action generation |
| Relation co-generation | P05 | Tiền lệ từ HOI, adaptation signing phải đánh giá lại |
| Semantic representation | P06–P08 | Cơ chế tham khảo, không coi encoder ảnh/action là sign oracle |
| DDPM baseline | P09, P10 | Objective/scheduler/sampler riêng được khóa ở mục 15 |
| Training framework | T01–T06 | DDP, AMP, SDPA masks, mT5, Hub snapshots |

Code release SignSparK được kiểm tra có Apache-2.0 ở root; MARDM có MIT ở root. SMPL-X/code/model assets có điều kiện riêng; xem license tương ứng. SOKE/SignFlow không phải base để phát hành fork sửa đổi mặc định do license ND trong audit trước. Không cần chúng để chạy code mới. Source dataset terms vẫn áp dụng cho reconstructed pose release. [R01–R04, D01–D04]

## 2. Repository bắt buộc

```text
cpr-flow/
  pyproject.toml
  requirements.lock.txt
  README.md
  IMPLEMENTATION_STATUS.md
  THIRD_PARTY_NOTICES.md
  configs/
    base.yaml
    data/{joint3,csl,how2sign,phoenix}.yaml
    model/{ae,cfm,relational,diffusion,teacher}.yaml
    runs/{smoke,ae_joint3,duration_joint3,teacher_joint3,baseline_joint3,baseline_cont_joint3,relation_joint3,auxonly_joint3,capacity_joint3,diffusion_joint3,full_joint3}.yaml
  assets/                 # model assets ignored by git
  data/                   # corpus/metadata/caches ignored by git
  runs/                   # outputs ignored by git
  src/cprflow/
    __init__.py
    config.py
    types.py
    data/{sources,manifest,lmdb_adapter,resample,stats,dataset,sampler,collate,cache}.py
    geometry/{rotations,body_model,joint_maps,relations,temporal}.py
    models/{ae,text,duration,embeddings,attention,dit,teacher}.py
    objectives/{reductions,ae,flow,geometry,lag,semantic,diffusion}.py
    training/{engine,distributed,precision,checkpoint,ema,logging}.py
    sampling/{ode,cfg,guard}.py
    evaluation/{geometry,dtw,retrieval,backtranslation,efficiency,aggregate}.py
    cli/{audit_sources,prepare,fit_stats,train,cache,sample,evaluate,render,profile}.py
  scripts/{run_stage,download_authorized_data}.sh
  tests/{test_rotations,test_schema,test_resample,test_ae_padding,test_relations,
         test_flow,test_attention_masks,test_gradients,test_ddp,test_resume,test_no_oracle}.py
```

Tên CLI public là `python -m cprflow.cli.<module>`. Mọi file phải import được từ editable package, không sửa `sys.path` theo notebook. Config loader dùng dataclasses/Pydantic hoặc strict schema; unknown key là error. Không dùng một file config chứa flags không được code đọc.

`runs/<run_id>/` chứa resolved config, sources lock, data manifest hash, environment, git commit/dirty diff, train/dev logs, checkpoints, dev samples và final metrics. Không commit weights/raw video/SMPL-X models vào git.

## 3. M0: environment và sources lock

### 3.1 Environment

Khởi đầu Python3.10 hoặc3.11, PyTorch có CUDA phù hợp driver máy lab. Upstream SignSparK hiện hướng dẫn Python3.10/Torch2.10/CUDA12.8; đó là setting upstream, không phải yêu cầu bắt buộc của repo mới. Coding agent kiểm tra `nvidia-smi`, CUDA runtime, BF16 support, SDPA và SMPLXLayer trước khi pin environment thực tế. Không ghi “tested” cho một version chưa chạy.

Dependency groups:

- Core: torch, numpy, scipy **offline resampling/metrics only**, PyYAML, tqdm, safetensors, lmdb, transformers, sentencepiece, huggingface_hub, smplx.
- Evaluation/render: scipy, pandas, imageio/ffmpeg; pyrender/trimesh hoặc một renderer đã cài thành công. Rendering không là dependency của sampling parameters.
- Dev: pytest, jsonschema, ruff; logging dùng JSONL/TensorBoard. W&B optional, disabled mặc định.

Chọn đúng build torch trước khi `pip freeze`; không trộn CPU/CUDA wheels. Baseline không bắt buộc xformers, flash-attn hoặc PyTorch3D compilation. Viết rotation conversion thuần Torch với tests. SMPL-X assets chỉ download theo license của chủ sở hữu.

```bash
python -m venv .venv
source .venv/bin/activate
# Install the torch CUDA wheel appropriate for this machine first.
pip install -e '.[dev,eval]'
python -m cprflow.cli.audit_sources --config configs/base.yaml --resolve-lock
python -m pytest tests/test_rotations.py tests/test_flow.py -q
```

Các bước cài package phải do agent ghi chính xác sau khi tạo `pyproject.toml`; comment chọn Torch ở trên không được thay bằng “latest” tự động ở mọi run.

### 3.2 Lock và provenance

Tạo `sources.lock.json` chứa repo URL, resolved Git commit, SHA256 của các source files tham khảo, HF dataset revision, mT5 model revision, tokenizer revision, dependency lock hash và asset hashes. Resolve branch một lần, sau đó `snapshot_download(revision=<resolved_revision>)`. Upstream downloader chưa pin revision; wrapper mới phải pin. Không giả SHA trong tài liệu là revision tồn tại; `git cat-file -e <sha>^{commit}` hoặc Hub API phải xác nhận.

Record `SMPLX_NEUTRAL` file hash, `num_expression_coeffs=50`, `num_betas=10`, `use_face_contour=false`, `joint_mapper=null`. Đây là configuration của v1; không tự rút expression xuống10 để tránh lỗi shape.

## 4. Dữ liệu và quyền truy cập

### 4.1 Dataset routing

| Canonical ID | Sign language / text | Pose release name prefix | Nguồn gốc |
| --- | --- | --- | --- |
| `csl_daily` | CSL / Chinese | `CSL-Daily_reopt_` | D01 |
| `how2sign` | ASL / English | `How2Sign_reopt_` | D02 |
| `phoenix2014t` | DGS / German | `PHOENIX-2014T_reopt_` | D03 |

Initial pipeline dùng reconstructed LMDBs từ D04/R01, không refit cả ba video corpora từ đầu. Chỉ download/train khi lab có quyền theo dataset terms. Dataset card của reconstructed release không thay quyền source corpus. Không dùng BOBSL/CSL-News hoặc released SignSparK generator weights trong controlled training.

Public release có thể re-optimize khác paper; đây là **new controlled track**, không gán lại số literature làm kết quả tái lập. Dữ liệu phải cùng version cho mọi baseline. Đếm actual clips sau import; đừng hardcode counts từ paper vì fitted release có thể lọc clips.

### 4.2 Download interface phải implement

Wrapper `scripts/download_authorized_data.sh` gọi Hub snapshot với repo `LionelLow/SignSparK_data`, `repo_type=dataset`, pinned revision và allow patterns của đúng ba corpora. Nguồn upstream download script ở R01. Giữ original bytes read-only sau download.

```text
data/raw_pose/<revision>/
  train/CSL-Daily_reopt_train.lmdb/data.mdb
  dev/CSL-Daily_reopt_dev.lmdb/data.mdb
  test/CSL-Daily_reopt_test.lmdb/data.mdb
  train/How2Sign_reopt_train.lmdb/data.mdb
  ...
  test/PHOENIX-2014T_reopt_test.lmdb/data.mdb
data/manifests/{train,dev,test}.jsonl
data/processed/<preprocess_hash>/<dataset>/<split>/<clip_id>.npz
data/stats/<stats_hash>/{motion,latent,relation,activity}.npz
data/cache/<cache_hash>/{text,latent,relations}/
```

Không substring-match paths khi hai revision cùng nằm trong thư mục; manifest ghi full exact source path. Không tự rename `validation` thành `test`; chỉ map source dev→dev. Duy trì original clip ID, corpus ID và source split.

## 5. LMDB adapter và canonical schema

### 5.1 Source schema đã kiểm tra

`__meta__` chứa clip index; mỗi clip là NPZ bytes. Fields chính: `translation`, `language`, `gloss`, `segment`, `left_features`, `right_features`, `body_features`, `face_features`. Source `gloss/segment` chỉ được giữ trong provenance/diagnostic; không đưa vào model input main. [R01]

Adapter đọc metadata của source đã xác minh, lazy-open LMDB **trong từng worker**, không giữ open handle qua fork. Readonly, max_readers đủ cho world size × workers; configurable readahead. Missing key/schema mismatch là error/quarantine có reason; không trả `None` rồi silently drop trong collate.

### 5.2 Canonical order chính thức

| Slice `x[..., :]` | Kích thước | Nguồn |
| --- | ---: | --- |
| `[0:60]` | 60 | body upper10 |
| `[60:150]` | 90 | native left-hand |
| `[150:240]` | 90 | native right-hand |
| `[240:246]` | 6 | jaw rot6D |
| `[246:296]` | 50 | expression coefficients |

**Canonical order khác upstream `full` concat**, vốn ghép left/right/body/face. Viết functions named `source_to_canonical` và `canonical_to_source`; never reuse upstream indices vào canonical tensor.

Expected source body shape `[T,126]` là21×6, select `[66:126]`. Nếu source body đã `[T,60]`, chỉ nhận khi source profile khai báo upper10 rõ. Shape ≥126 nhưng lớn hơn126 không được lấy `[:,66:]` mù quáng; cần source-profile adapter explicit. Hands `[T,>=90]` lấy90 đầu chỉ khi source card/profile xác nhận trailing fields không phải rotations cần thiết. Face phải `[T,56]`; profile133 hoặc face10 là nguồn khác, không pad thành profile296.

All arrays float32, finite, aligned cùng T trước resampling. `translation` decode string scalar/[1], Unicode NFC + collapse whitespace; giữ case và punctuation, không dịch sang English. Language IDs fixed: `CSL=0, ASL=1, DGS=2`; xác minh source language tag, không suy sign language từ ngôn ngữ text.

### 5.3 Record/manifest contract

```python
@dataclass
class MotionRecord:
    uid: str                 # dataset:source_clip_id, globally unique
    dataset_id: str
    sign_language_id: int
    spoken_language: str
    source_clip_id: str
    source_video_id: str | None
    signer_id: str | None
    split: str               # train/dev/test
    text: str
    motion: np.ndarray       # [T,296], physical parameter representation
    fps: float               # processed effective sampling rate
    source_fps: float        # verified feature sampling FPS, not guessed video FPS
    source_frames: int
    timestamps: np.ndarray   # [T], seconds
    preprocessing_hash: str
    asset_profile_hash: str
```

Manifest thêm source LMDB path/revision/file hash, feature profile, mapping version, text hash, quality flags, access-record reference và path của processed NPZ. `source_video_id` chưa biết để null; không gán clip ID làm video ID rồi tuyên bố đã deduplicate source videos. Reliability optional/null; không bịa confidence.

Persist NPZ numeric arrays, strings trong JSONL; không object arrays trong processed data. `np.load(...,allow_pickle=False)` cho processed files. Original metadata pickle chỉ xử lý source release tin cậy với hashes đã lưu; không ingest arbitrary pickle từ internet.

## 6. Frame rate, duration và sequence coverage

### 6.1 Xác minh sampling trước khi dùng velocity

PHOENIX raw video25fps theo D03; raw How2Sign/CSL thường30fps, nhưng **raw-video FPS không chứng minh pose LMDB FPS**. R01 schema không có FPS bắt buộc. `audit_sources` phải xác nhận bằng metadata preprocessing release hoặc đối chiếu clip frame/timestamps với video/annotations gốc. Tạo `source_timing.json` theo revision/dataset; ghi evidence URL/file và verification method. Nếu chưa xác định được effective pose FPS, dừng main preprocessing của nguồn đó; vẫn làm unit tests và source parser. Không dùng một giả định chưa xác minh để báo velocity m/s hoặc duration seconds.

Sau xác nhận, controlled track resample về30fps. Endpoint-preserving grid: duration `(T_source-1)/fps_source`; output grid đều `j/30` tới không vượt duration, thêm/điều chỉnh terminal sample theo một rule đã version. V1 chốt `T_out=round(duration*30)+1`, `timestamps=linspace(0,duration,T_out)`; lưu effective `dt=duration/(T_out-1)`, không giả mọi clip có đúng1/30 nếu rounding. Motion derivative và positional encoding dùng timestamps này. Trong inference target dt=1/30.

Rotations resample SLERP shortest arc sau quaternion sign continuity; coefficients linear interpolation. Không interpolate trực tiếp6D rồi coi là SLERP. Không random time-stretch/mirror ở v1. Nếu downsample source FPS cao hơn target, thêm anti-alias preprocess đã version, test high-speed loss; ba dataset không tự tạo thêm source FPS bằng cách đo độ dài tensor.

### 6.2 Length policy

Không truncate source motion tới224/300 frames như một vài upstream demo mà giữ nguyên toàn câu. Training full-utterance v1 tối đa1024 processed frames; clips dài hơn nằm trong `long_train_exclusions.jsonl`, báo coverage theo dataset và nguyên nhân. Không giấu chúng hoặc random crop text-mismatched. Test không loại dài theo kết quả model: sample batch1 tới2048 frames; nếu hơn2048 hoặc OOM sau retry batch1, ghi explicit failure/coverage, không cấp GT length rồi truncate.

Monolingual/joint controls dùng cùng included clip sets. Báo cả full-test failure coverage và supported-subset metrics. Hướng long-context generation là task sau, không tự chunk và claim original sentence-level metrics.

T<=1 không có derivative: quarantine train production, giữ report diagnostic. Empty text, nonfinite arrays, unknown layout, misaligned T hoặc missing modality phải có quarantine record và counts. Trong full296 track không tự fill missing face bằng zero; representation thiếu face cần run profile riêng.

## 7. Rotations, handedness và differentiable geometry

### 7.1 Row-based 6D

R01 loader dùng hai **hàng** đầu của3×3 rotation. Với `a1=d6[:3], a2=d6[3:]`:

```python
b1 = normalize(a1)
u2 = a2 - dot(b1, a2) * b1
b2 = normalize(u2)
b3 = cross(b1, b2)
R = stack([b1, b2, b3], dim=-2)   # rows, not columns
d6 = R[..., :2, :].reshape(..., 6)
```

Identity6D là `[1,0,0,0,1,0]`, **không phải sáu số0**. Physical FK áp dụng standard rotation matrices theo SMPL-X; row representation không có nghĩa đổi toàn bộ FK convention sang row-vector multiplication.

Safe decode: nếu norm a1<1e-6 chọn e1; nếu orthogonalized a2 degenerate, chọn basis e_k ít aligned nhất với b1 rồi project/normalize. Implement bằng Torch operations/where; log degenerate-rate. Ground-truth degeneracy là data error. Generated degeneracy được fallback deterministic nhưng phải report; không để fallback che mô hình collapse. Loss/geometry chạy FP32. Geodesic rotation distance dùng robust atan2 của skew norm và trace, tránh acos derivative vô hạn ở identity. Clamp/floor phải được unit-test tại0/pi.

### 7.2 Native hands

Canonical left là native SMPL-X-left trong LMDB, không `flip_left_hand_features`. V1 có hai AE encoders riêng, không cần chuyển về right-hand frame. Chỉ khi đọc **upstream generated hand-stream outputs** có right-hand convention mới dùng explicit adapter với conjugation như upstream; ghi convention vào manifest. Không flip lại native raw data khi render. Render both hand poses với known asymmetric synthetic pose để kiểm tra.

### 7.3 Body pose assembly

SMPL-X `body_pose`21 joints, không gồm global root. Mười body joints được predict ở body-pose indices11..20:

| Body-pose index | Name | Base SMPL-X joint index |
| ---: | --- | ---: |
| 11 | neck | 12 |
| 12 | left_collar | 13 |
| 13 | right_collar | 14 |
| 14 | head | 15 |
| 15 | left_shoulder | 16 |
| 16 | right_shoulder | 17 |
| 17 | left_elbow | 18 |
| 18 | right_elbow | 19 |
| 19 | left_wrist | 20 |
| 20 | right_wrist | 21 |

Build21 matrices initialized identity, insert predicted upper10. Global root identity, translation0, beta neutral0, eyes identity, jaw from face[:6], expression face[6:56]. `use_pca=False`, `flat_hand_mean=True`, expression50, no joint mapper, no face contour. Không decode zero6D cho lower11 rồi overwrite NaNs sau đó.

Profile này không tái tạo global locomotion/lower body/spine motion đầy đủ. Không dùng nó để claim body-space fidelity ngoài các DOF được giữ. Side-by-side native fitting render là G0 bắt buộc. [R01, R03]

### 7.4 Differentiable backend

Dùng `smplx.SMPLXLayer` với rotation-matrix inputs, `return_verts` khi cần fingertips/face landmarks. Backend này gọi LBS với `pose2rot=False`. Chỉ `requires_grad_(False)` cho model buffers/parameters; **không `no_grad` cho predicted motion đi vào loss/guard**. Không `.cpu().numpy()`, scipy Rotation hoặc detached axis-angle trong đường backward. SMPLXLayer shapes:

```text
body_pose:       [N,21,3,3]
left_hand_pose:  [N,15,3,3]
right_hand_pose: [N,15,3,3]
jaw_pose:        [N,1,3,3]
global_orient:   [N,1,3,3]
expression:     [N,50]
betas:          [N,10]
```

Chunk valid frames theo `fk_chunk_frames`128 (start value) để giảm peak memory; **không detach giữa chunks**. Ground-truth joints/relations precompute offline. Prediction geometry có thể checkpoint theo frame chunks để trade compute/RAM. Guard không dùng fast nondifferentiable renderer.

Base hand rotations order theo source/SMPL-X là index,middle,pinky,ring,thumb, mỗi finger3 joints; không đổi thành thumb-first vì thói quen. Fingertips lấy vertex-joint selector có names `left_thumb/index/middle/ring/pinky` và phải kiểm tra output mapping ở R03. Không lấy `index3` làm fingertip. Face landmarks chọn by names/version, không chép các magic indices từ body-only skeleton.

## 8. Relation33: định nghĩa đủ để code

### 8.1 Coordinate frame

Geometry từ neutral-shape FK. `o=(J_neck+J_pelvis)/2`. `ex=normalize(J_right_shoulder-J_left_shoulder)`. `ey=normalize(v-dot(v,ex)*ex)` với `v=J_neck-J_pelvis`. `ez=cross(ex,ey)`. `C=stack([ex,ey,ez],dim=-1)` là **columns** của body frame trong world; local vector là `C.T @ (p-o)`. Đây là frame transform, độc lập row-based rot6D serialization. `s=||right_shoulder-left_shoulder||`, assert s>0.05m và finite theo neutral asset.

Anchors theo thứ tự cố định: jaw, o, left_shoulder, right_shoulder. Mỗi hand point set theo thứ tự wrist, thumb_tip,index_tip,middle_tip,ring_tip,pinky_tip. Palm normal native world: `normalize(cross(index1-wrist,pinky1-wrist))`; mirror anatomy giữ sign native, không sửa normal chỉ để hai tay trông “giống”. Transform normal bằng C.T. Test mapping/index round trip.

### 8.2 Ordering

| Slice | Nội dung |
| --- | --- |
| 0:3 / 3:6 | left/right wrist local positions chia s |
| 6:9 | `(right_wrist-left_wrist)` local chia s |
| 9:12 / 12:15 | left/right palm normals local |
| 15:19 | min distance left-hand point set tới4anchors, chia s |
| 19:23 | tương tự right-hand |
| 23:24 | min pairwise distance giữa2 hand point sets, chia s |
| 24:33 | soft proximity của distances15:24, cùng order |

`proximity=sigmoid((0.08-d)/0.02)` trong shoulder-width units. Native left/right labels không thay đổi. Đây là **joint/landmark proximity**, không là surface contact GT. Relation không có labels mouth grammar hoặc phonology do LLM tạo.

### 8.3 Latent rate và normalization

Frame-level `R=relation33(X)` có `[T,33]`. `pool2` lấy arithmetic mean của frames(2l,2l+1), frame cuối odd chỉ dùng một frame. `R_low=pool2(R)` dàiceil(T/2), rồi normalize bằng train-only mean/std. Normals sau mean có thể norm<1: đó là averaged feature hợp lệ, **không renormalize riêng ở một nhánh**. Cycle/target dùng cùng operator `A(X)=norm_R(pool2(relation33(FK(X))))`.

Không tính relation từ average pose rồi coi tương đương average relation. Loss/eval phải dùng đúng A. Không clip noisy/predicted relation về[0,1] trong ODE. R_head có thể ngoài physical range; ghi diagnostic invalid-rate và tái tính final metrics từ decoded geometry.

## 9. Statistics, masks và caches

### 9.1 Dataset-balanced statistics

Joint3 stats dùng train-only **clip-balanced, dataset-balanced** moments: trong mỗi clip mean qua valid frames, average clips trong dataset, rồi average ba dataset với weight1/3. Variance tính qua second moment theo cùng weights; `std=sqrt(max(E[x²]-mean²,0)).clamp_min(0.01)`. Không average sample stds. Ghi phương pháp và counts. Bốn part motion được norm per-channel; relation33 riêng; latent stats riêng sau AE frozen. Monolingual control vẫn dùng joint3 stats và AE nếu mục tiêu cô lập generator transfer; fully monolingual AE là ablation khác.

Activity/velocity scale là train RMS theo cùng sampling weights, có floor. Không tính normalization từ dev/test hoặc mỗi inference sample. Statistics hash gắn với included train manifest + preprocessing + asset + AE checkpoint nếu latent.

### 9.2 Tensor contract

```text
MotionBatch:
    x: FloatTensor[B,Tmax,296]         # normalized parameters
    lengths: LongTensor[B]            # physical valid frames
    frame_mask: BoolTensor[B,Tmax]    # True=valid
    timestamps: FloatTensor[B,Tmax]
    language_id: LongTensor[B]
    dataset_id: list[str]
    text: list[str]
    uid: list[str]
LatentBatch:
    z: FloatTensor[B,4,Lmax,128]
    r: FloatTensor[B,Lmax,33]          # only relational model
    latent_lengths: LongTensor[B]     # ceil(lengths/2)
    latent_mask: BoolTensor[B,Lmax]
    token_times: FloatTensor[B,Lmax]   # mean timestamp of each pair
TextBatch:
    hidden: FloatTensor[B,Smax,768]
    mask: BoolTensor[B,Smax]
```

Outside AE convolution internals, invalid positions là0 trong normalized representation. Loss means exclude pads. Mask bool semantics được wrap thành một helper; PyTorch SDPA uses True=allowed, `MultiheadAttention.key_padding_mask` uses True=ignore. Không truyền cùng mask trực tiếp cho cả hai. All-masked rows phải cho output0, không NaN. [T01]

### 9.3 Cache keys và invalidation

- Processed motion: source revision/UID + preprocess version + FPS policy + convention/mapping.
- Text: normalized text + sign-language tag + tokenizer/model revision + tokenization config; teacher embedding cache thêm teacher hash.
- Latent: processed motion hash + AE EMA checkpoint hash + latent stats hash.
- Relations: processed motion hash + SMPL-X asset/profile + relation33 version + relation stats hash.

Không reuse latent cache sau AE fine-tune. Cache dùng per-clip safetensors/NPZ numeric hoặc sharded indexed format, không đẩy toàn bộ corpus lên VRAM. DataLoader workers CPU only, pinned memory optional, workers/rank khởi đầu4 và tune theo I/O.

## 10. Dataset sampler và loss reduction

### 10.1 Joint3

Default `q(dataset)=1/3` cho ba corpora; đây là quyết định implementation, không lấy sampling ratio của checkpoint upstream. Mỗi **logical optimizer batch** chọn một dataset bằng deterministic RNG(seed,optimizer_step), rồi chọn length bucket theo số included clips trong dataset đó; sample64 clips cho flow,128 cho AE. Whole batch cùng dataset giúp batching và negative sampling cùng ngôn ngữ. Không round-robin theo kích thước corpus khiến How2Sign chi phối. Log actual samples/frames theo corpus mỗi1k steps.

Global list IDs được tạo trước khi shard cho ranks; không mỗi rank sample một dataset riêng. Bucket boundaries128/256/512/1024. Sample without replacement trong batch nếu đủ clips; nếu bucket nhỏ hơn global batch, replacement được cho phép nhưng đánh dấu duplicate IDs và đưa vào multi-positive mask. RNG resume phải phục hồi chính xác next batch.

Nếu muốn mixture `q~sqrt(N_dataset)`, thêm run riêng `joint3_sqrt`, không sửa ratio của baseline giữa chừng. Dev/test duyệt tuần tự toàn bộ từng dataset, không balanced sampling evaluation.

### 10.2 Reduction chuẩn

Mọi basic reconstruction/FM loss là **mean valid elements của từng clip**, sau đó average clips; không mean tất cả frames khiến clip dài có trọng số lớn. Cho stream s:

```python
def per_clip_mse(pred, target, mask):
    # pred [B,L,D], mask [B,L], True valid
    err = (pred.float() - target.float()).square()
    numer = (err * mask[..., None]).sum((1, 2))
    denom = mask.sum(1).clamp_min(1) * pred.shape[-1]
    return numer / denom     # [B]; reduce later over logical batch
```

Main FM motion loss = average của4 stream MSEs; relation loss có hệ số riêng0.25, để thêm relation không thay weight motion. Các weights cố định, không phụ thuộc endpoint confidence. Losses không có label/cặp derivative hợp lệ ở clip nào được đặt0 và report coverage; không để `mean(empty)`.

DDP mặc định average gradients qua W ranks. Với sum of per-clip terms và global count N, local loss phải là `W*local_sum/N`; all-reduce denominator detached. Microbatch accumulation dùng denominator logical batch **toàn cục**, không chia tiếp cho accumulation_steps một lần nữa. Có regression test so gradients single-GPU logical batch với simulated shards/microbatches. Mean loss logging dùng all-reduce numerator/denominator, không chỉ rank0 local loss.

## 11. Continuous motion autoencoder

### 11.1 Architecture exact

Bốn part encoders/decoders độc lập: body60, LH90, RH90, face56. Input/output layout public `[B,T,D]`; Conv1d internal `[B,C,T]`.

Encoder một part:

1. Conv1d(D,128,k3,s1,p1) + SiLU.
2. Hai ResBlocks128, dilation1 rồi2.
3. Conv1d(128,256,k3,s2,p1): output ceil(T/2).
4. Hai ResBlocks256, dilation1 rồi2.
5. Conv1d(256,128,k1): latent.

ResBlockC: channel-only LayerNorm trên `[B,T,C]` → SiLU → Conv1d(C,C,k3,dilation=d,padding=d) → channel LayerNorm → SiLU → Conv1d(C,C,k1) → residual add. Không BatchNorm/temporal GroupNorm. Norm không tổng hợp padding/time dimensions.

Decoder: latent fusion → Conv1d(128,256,k3,s1,p1) → hai ResBlocks256 d2,d1 → nearest-neighbor upsample×2 → Conv1d(256,128,k3,p1) → hai ResBlocks128 d2,d1 → Conv1d(128,D,k1), crop về original T. Convolution boundaries dùng replicate padding đúng true length, không pad zero chung rồi giả valid boundary bất biến.

Fusion dùng concat bốn latents cùng timestamp `[B,L,512]` → LN512 → Linear512→256 → SiLU → Linear256→512, split4×128, residual add mỗi part. Fusion thuộc decoder; encoder cache vẫn giữ bốn latents gốc. Initialize fusion cuối0. Tất cả variants dùng cùng AE/fusion checkpoint.

### 11.2 Padding invariance bắt buộc

MVP correctness path: trim từng clip tới trueT, encode/decode theo clip hoặc group cùng trueT, rồi pad latents/output cho batch. Vì không BatchNorm, kết quả independent. Có thể optimize sau bằng masked replicate-padding tại **mọi** temporal layer, nhưng phải pass `output(prefix alone)==output(prefix in longer padded batch)` và gradient tests. Không chấp nhận “mask loss là đủ”: convolution có thể nhìn thấy pad.

Initialize output biases để clean physical rotations gần identity sau inverse normalization; expression gần train mean. Generated degenerate6D fallback-rate phải log. Không dùng sigmoid/tanh toàn bộ motion output để giới hạn expression/rotations.

### 11.3 AE objectives

`L_AE=L_param +0.5 L_rot +1.0 L_FK +0.1 L_vel`.

- `L_param`: Huber(delta1) trên normalized channels, mean clip/part; không cho90D hand tự lấn56D face.
- `L_rot`: mean geodesic angle giữa rotations, chia pi; gồm body10, mỗi hand15, jaw1, average bốn groups.
- `L_FK`: body10 global positions + hand21 global positions + hand20 relative-to-wrist positions; divide shoulder width. V1 weights group body1, globalLH1, globalRH1, localLH2, localRH2, normalize tổngweights7. Face position loss chưa mặc định vì basis/landmark mapping cần audit; jaw/expression vẫn có param/rotation losses.
- `L_vel`: target residual của same FK groups và expression velocities, dùng true timestamps; scale bằng fixed train RMS của group, floor1e-3. Chỉ valid adjacent frame pairs, average clip/group. Không minimize speed/acceleration tuyệt đối.

Để stageAE vừa RAM, FK/rotation auxiliaries trên **cùng một tập frames stride2 được dịch offset0/1 theo optimizer_step** cho cả target/prediction; velocity trên consecutive sampled pairs với dt thật. Dense validation đo full frames. Đây là fixed objective sampling policy, ghi config; đừng ngầm subsample khác nhau giữa variants. Nếu đủ RAM có dense training run riêng, không trộn bảng.

Loss reduction của L_param vẫn dense. Không có KL, VQ commitment, adversarial hoặc semantic loss trong default AE. StageAE max200k updates, effective batch128, AdamW lr2e-4, weight_decay0.01, warmup5k, cosine đến lr_min2e-5, gradclip1.0. Freeze best-dev EMA AE để huấn luyện flow; không vừa train AE vừa cache latent.

AE checkpoint selection: macro-average normalized dev reconstruction loss qua3datasets, cùng weights ở trên. Report từng dataset và từng part, finite-gradient và degenerate-rate. Kết quả human contrast ceiling chưa có thì chỉ ghi engineering gate, không giả xác nhận3pp linguistic retention.

## 12. Text encoder và duration model

### 12.1 mT5

Load `MT5EncoderModel.from_pretrained('google/mt5-base',revision=...)`, frozen, eval mode; tokenizer cùng revision. Encode string `'<CSL> '+text` bằng literal tag nếu không thêm vocabulary mới; các tag không cần một token độc lập vì language ID còn có embedding riêng. Chốt default dùng prefix literal `[CSL]`, `[ASL]`, `[DGS]`; không resize frozen embeddings bằng token mới random.

Max text tokens512. Nếu vượt, fail/quarantine train với coverage, không truncate âm thầm; eval ghi unsupported_text_length. Batch pad right, use tokenizer attention_mask. Cache token hidden768D fp16/bf16 trên disk, train DiT projection768→384. Text features cache phải ở eval mode để dropout không làm cache phụ thuộc lượt chạy. [R05, T03]

CFG drop toàn clip text condition với p0.1. Dùng một learnable null token384D với text maskTrue, giữ language ID/output length. Do đó “unconditional” nghĩa unconditional-on-text, **không** unconditional-on-language/duration. Không dùng empty/all-false mask gây NaN. Null token/projected text đặt trong generator, không trong mT5 frozen cache.

### 12.2 Duration network

Concat masked mean token768 với language embedding32 → LayerNorm800 → Linear800→1024 → SiLU → dropout0.1 → Linear1024→2. First input dim800, không768. Heads `mu` và `rho`; `sigma=clamp(softplus(rho)+0.05,max=2.0)`. Target là `log(T_processed)`, full log-normal NLL gồm `log T` nếu báo likelihood.

Training AdamW lr1e-4, batch256 token-cache examples, max30k updates, early stop patience5 validations mỗi1k, same dataset-balanced sampler. Inference median `exp(mu)`, round nearest even frames theo `2*floor((median+1)/2)`, clamp `[2,2048]`; log preclamp and clipping_rate. `duration_seconds=(T_hat-1)/30`. Chọn checkpoint bằng macro normalized absolute log-length error trên dev, báo MAE frames/seconds theo dataset.

Generator training conditioning GT T là bình thường vì target sequence có T đó. Main inference/evaluation dùng duration prediction; oracle T ở CLI riêng. Không thêm noise cho training T khiến tensor target không tương ứng length.

## 13. Typed-stream DiT

### 13.1 Shapes và tokenization

Motion pack `z[B,4,L,128]`. Plain baseline P4. Relational model thêm `r[B,L,33]`, P5. Project mỗi part128→384 và relation33→384; part embeddings gồm body/LH/RH/face/relation. **Không concat feature33 với128 rồi nhét chung một linear128.** Working hidden `[B,P,L,384]`.

Temporal positional embedding là sinusoidal384D từ token timestamp seconds, frequencies log-spaced như standard sinusoidal; nhân thời gian30 để index scale xấp xỉ frame. Position của token l là mean timestamps của frames2l,2l+1. Length/time/language conditioning: sinusoidal embedding của flow t256 → MLP256→384→384, plus embedding language3×384, plus MLP(log(T),log(T/30))2→384→384. Tổng three vectors qua SiLU. Dtype và time convention kiểm tra bằng tests t0/t1.

### 13.2 Mỗi block theo thứ tự exact

1. Temporal self-attention theo từng stream: reshape `[B*P,L,H]`, noncausal, valid padding mask.
2. Local cross-part attention: query tại mỗi `(p,l)`, keys tất cả parts ở latent offsets -4..+4; tương đương ±8 original frames. Mask out-of-range và padding.
3. Chỉ blocks2/4/6/8: query mỗi motion/relation token tới P global part tokens, được mean-pool valid temporal positions của hidden hiện tại. Global tokens chỉ context, không là extra state output.
4. Text cross-attention: tất cả valid part-time queries tới projected mT5/null token.
5. FFN Linear384→1536 → GELU → Linear1536→384.

Mỗi sublayer có pre-LN không affine và adaLN time/lang/length scale/shift/gate. Modulation output3H mỗi active sublayer; gate zero-init. Self-attention6 heads, head_dim64, separate projections cho temporal/cross-part/global/text, shared across part types trong block. Attention dropout0.0; FFN dropout0.0 mặc định. Final LN + per-stream output head128 hoặc33, weight/bias zero-init theo DiT practice.

Sau mỗi residual step, zero invalid queries. Không materialize dense `[B,P*L,P*L]` mask để giả window efficiency; implement gather-based cross-part windows `[B,L,P,9P]` hoặc equivalent unfold. Temporal attention full L×L; profile memory, global part context không thay cho tính toán này.

### 13.3 B1 → C0 warm start

Reuse AE/text/duration, DiT common blocks và4motion in/out heads. Thêm relation projection/head/type embedding. Trong2k continuation steps đầu, attention từ relation keys vào motion queries ramp `alpha_R:0→1`; alpha0 mask relation keys khỏi motion cross-part/global attention để output B1 không bị đổi chỉ vì softmax thêm keys. Relation queries vẫn đọc motion để học. Với0<alpha<1 dùng log(alpha) attention bias; at0 dùng bool mask. Gate này không áp dụng temporal attention riêng từng stream.

Test alpha0: copied B1 vs C0 motion outputs bằng nhau ở eval/FP32 với same z/t/text, tolerance1e-5. Không yêu cầu C0 relation output bằngB1 vì B1 không có. B3 extra-capacity control cũng có cùng warm-up policy. Ramp là engineering stabilization, không novelty.

## 14. CFM objectives và training step

### 14.1 Time/path

`t~U(0,1)` shape[B], một t dùng chung tất cả streams trong clip; `eps` độc lập theo component, zero padded positions. `U_t=(1-t)*eps+t*U1`, target `U1-eps`. U1 motion là **normalized frozen-AE latents**; relation là normalized A(X). Không corrupt text/language/length. Không project noisy motion latent/relations thành physical rotations.

`L_FM=(1/4)sum_s MSE_clip(vz_s,z1_s-eps_s) +0.25*MSE_clip(vr,r1-eps_r)` khi P5. P4 không có relation term. Hệ số này cố định; confidence chỉ dùng cho optional auxiliary labels, không trọng số FM endpoint-dependent. No OT v1.

### 14.2 Endpoint và geometry

`U1_hat=U_t+(1-t)*v_hat`; denorm latent → differentiable frozen decoder → physical x_hat. Define A(X) như8.3. Auxiliary per-example:

- `L_geom = Huber(A(x_hat),A(x_gt)) + L_FK(x_hat,x_gt) +0.1 L_vel(x_hat,x_gt)`.
- `L_cycle = Huber(r1_hat,A(x_hat))`; không detach một nhánh theo mặc định. Ground-truth supervision của FM/geom ngăn two branches tự nhất quán nhưng cùng sai.
- `L_lag`: mục14.3.
- `L_sem`: mục17; disabled tới khi teacher gate đạt.

`L_total = L_FM + ramp*(0.1 L_geom +0.05 L_cycle +0.01 L_lag +0.05 L_sem)`.

Aux chỉ xét t>=0.5, và semantic aux thêm điều kiện text không bị CFG-drop. Terms invalid đặt0 trong vector[B], average trên logicalB. Trong continuation chỉ evaluate aux mỗi4 optimizer steps, multiply **4** vào auxiliary loss ở steps đó để giữ expected weight tương đương objective đã định nghĩa. Không vừa multiply4 vừa tune coefficient theo convention không bù mà ghi cùng tên. FM tính mọi step. Logs có raw aux và effective weighted loss.

Nội dung ramp: continuation steps0..19999 chỉ joint FM; từ20k đến40k ramp0→1;40k..100k full weights. Teacher không đạt thì semantic coefficient forced0, run gắn `semantic_enabled=false`, không ghi “full CPR-Flow”.

### 14.3 Lag objective exact

Groups: left hand activity norm của20finger relative-to-wrist + global wrist velocity; right tương tự; head activity từ geodesic angular velocity của head rotation; face activity RMS velocity của normalized expression50 và jaw angular velocity. Group RMS scales fixed train-only. Pair list `(LH,RH),(LH,head),(RH,head),(LH,face),(RH,face)`.

Activity trước mỗi sample dùng derivatives / actual dt, align derivative tại midpoint timestamps. Window32 consecutive activity samples, hop16, shifts k=-15..15. Pearson correlation trên overlap; target-valid windows yêu cầu ít nhất32 activities và variance mỗi signal>1e-4 sau scaling. Prediction variance floor1e-4 trong denominator; **không loại window chỉ vì model dự đoán hằng số**, tránh model trốn loss. Softmax(corr/0.1), `KL(P_target||P_pred)`, clamp probs1e-8, mean windows/pairs/clip. `a_q[f+k]` nên q trễ3frames peak k=+3. Với clip ngắn không có valid windows: differentiable0 + coverage log.

### 14.4 Train-step pseudocode

```python
def flow_microstep(batch, global_clip_count, world_size, optimizer_step):
    # Logical-batch denominator known before splitting into microbatches.
    z1, r1, masks, text, lang, lengths = load_frozen_targets(batch)
    t = sample_uniform_per_clip(batch, rng=training_rng)
    eps_z = masked_randn_like(z1, masks)
    zt = (1 - t[:, None, None, None]) * eps_z + t[:, None, None, None] * z1
    rt, target_r = prepare_relation_path_if_enabled(r1, t, masks)
    pred = ddp_model(zt, rt, t, text, lang, lengths, masks)
    per_clip = motion_fm_per_clip(pred.z, z1 - eps_z, masks)
    if relation_enabled:
        per_clip = per_clip + 0.25 * relation_fm_per_clip(pred.r, target_r, masks)
    eligible = (t >= 0.5)
    if relation_enabled and optimizer_step % 4 == 0 and eligible.any():
        # Helpers select eligible clips, decode with grad, then scatter losses
        # into a [B_micro] vector whose ineligible entries are exactly zero.
        xhat, rhat = decode_eligible_endpoints_with_grad(zt, rt, pred, t, masks, eligible)
        aux = scatter_gated_aux_vector(xhat, rhat, batch, t, teacher, masks, eligible)
        per_clip = per_clip + 4.0 * aux_ramp(optimizer_step) * aux
    # DDP averages gradients over world_size; do not divide by accumulation again.
    loss = world_size * per_clip.sum() / global_clip_count
    backward_with_amp(loss)
```

Đây là control-flow contract; coding agent phải implement các functions, không để wrappers `pass`. Khi microbatch không có eligible aux sample, không gọi FK/teacher vô ích; vẫn tạo zero gắn graph khi cần. Freeze AE/teacher parameters nhưng duy trì autograd theo xhat. Unscale gradients trước grad clip; step scheduler/EMA chỉ khi optimizer update không bị AMP overflow bỏ qua.

## 15. Matched diffusion baseline

Để so objective, B0 dùng cùng AE/text/DiT P4, duration và training manifest. Forward noise schedule trong **baseline riêng** dùng DDPM convention k0=data,kK=noise, K1000. `alpha_bar(k)=cos²(((k/K+0.008)/1.008)*pi/2)` normalized alpha_bar(0)=1; derive beta clip[1e-5,0.999]. Network time embedding nhận `k/K`, output epsilon. Loss unweighted per-clip epsilon MSE, average4parts. Không reuse CFM noise→data formula bằng cách đổi tên t.

Sampler DDIM eta0, network time indices descendingK..1 rồi endpoint0, steps{16,32,50}; timestep indices được lưu trong result config, không count1000 training steps như NFE. Clean estimate `(z_k-sqrt(1-ab_k)*eps_hat)/sqrt(ab_k)`. Không clip continuous latent theo[-1,1]. Dùng official derivation P10, test deterministic DDIM và endpoint formula. B0 có thể dùng separate velocity/x0 parameterization ablations nếu budget đủ, ghi rõ, không gọi một epsilon/cosine run duy nhất là mọi diffusion models.

Tune B0 vàB1 bằng cùng dev budget, báo quality-cost frontier. DDIM vs Euler khác sampler nên kết luận thuộc configured system; không là theorem FM superiority. B0 không có relation/guard trong primary objective comparison. Trong primary B0/B1-cont comparison, cả hai dùng parent200k + EMA/reset-optimizer continuation100k cùng schedule; uninterrupted300k chỉ là reference bổ sung.

**Index contract authoritative:** dùng arrays0..K với alpha_bar[0]=1; train k uniform1..K. Tính betas từ cosine cumulative ratios rồi clip và recompute cumulative products. Sampling chọn NFE indices descendingK..1, append0 như endpoint, network không được gọi tại k0. Implementation chỉ dùng contract0..K, có test không off-by-one.

## 16. Teacher: module đầy đủ nhưng không chặn baseline

### 16.1 Teacher architecture

Input clean normalized canonical parts. Mỗi part linear D→256, sinusoidal positions, temporal stride2 bằng mean-pool input embeddings trên pair frames. Thêm learned mask-indicator projection nếu train masked reconstruction. Hai temporal Transformer encoder blocks/part, heads4, FFN1024, dropout0.1. Không đọc generated relation head. Mean-pool valid temporal features → MLP256→256 → L2-normalized part embedding. Global concat4×256 → MLP1024→512→256, L2-normalized. Text side masked mean frozen mT5 token768 + language embedding32 → MLP800→512→256, L2-normalized.

Temporal teacher features decode về normalized input parameters qua linear256→D và repeat-interleave2/crop để masked reconstruction. Input mask spans2–10 frames, mask probability0.3; positive motion/text embedding dùng clean pass, masked reconstruction dùng separate corrupted pass. Không làm regression của face từ hands như nhiệm vụ bắt buộc.

Global symmetric multi-positive InfoNCE temperature0.07 +0.1 mean per-part masked Huber reconstruction. Positives là exact normalized text cùng sign language hoặc một equivalence annotation đã được người ký hiệu xác nhận; không tự coi mọi “synonym” là cùng sign realization. Duplicate positives in batch được numerator mask. In-batch negatives mặc định trong cùng language; cross-language negatives không được để teacher chỉ học phân biệt corpus.

Teacher train-only, AdamW1e-4, effective batch64, max100k updates, choose macro dev retrieval recall@1 + masked reconstruction report. Với DDP, contrastive embeddings all-gather phải gradient-aware; test so một global batch trên single process. Nếu dùng gradient accumulation để đạt64 thì không được gọi64 là negative pool khi thực tế mỗi microbatch chỉ8. V1 có thể dùng microbatch negatives với poolsize ghi rõ và giữ cùng setting giữa runs, hoặc implement cross-batch frozen queue; chọn mặc định **logical batch64 có differentiable gather/gradient-cache implementation đã test**. Khi chưa thực hiện được, chạy teacher single-device batch nhỏ hơn và label `teacher_negative_pool=<actual>`, không báo tương đương64.

### 16.2 Semantic auxiliary

Freeze teacher params, retain gradient through teacher input xhat. Motion query `g(xhat)` so với frozen text keys `h(y_j)` của toàn logical training batch; multi-positive numerator, temperature0.07. Chỉ dùng samples t>=0.5 và text condition không drop; denominator có same-language keys. Precache text keys để microbatch gradient accumulation không đổi negative pool. Nếu không có eligible negatives, semantic term0 + coverage log.

Loss này là weak transcript alignment. Cho phép run `relation_semantic_aux` để nghiên cứu, nhưng default first training đặt coefficient0 cho tới teacher contrast sensitivity được kiểm tra. Không gọi default teacher “certified semantic model”.

### 16.3 Calibration artifact

`teacher_calibration.json` phải có teacher hash, asset/preprocess hash, dev annotation split, sensitivity/false alarm theo handshape/location/orientation/NMF, threshold selection, sample counts và trust budgets. Proposed gate sensitivity>=0.8, false-positive<=0.1 trên annotated dev contrasts, không phải standard ngôn ngữ. Đủ sample/power chưa có thì `approved=false`.

Phân biệt meaning-changing contrasts với style-preserving positives phải do người ký hiệu/linguist xác nhận. Automated perturbation không tự là label đổi nghĩa. Không dùng test contrasts để đặt budgets. Nếu thiếu annotations, agent vẫn implement teacher/guard/tests, để guard disabled; không tạo file calibration giả với `approved=true`.

## 17. Euler sampler, CFG và guarded correction

### 17.1 Plain inference contract

API main không nhận `MotionBatch`, chỉ nhận request chứa text, sign language, seed và optional user-specified duration. Default user_duration=null → duration model. Oracle mode nằm trong evaluator riêng, không trong default API.

```python
@dataclass
class GenerationRequest:
    text: str
    sign_language: str
    seed: int
    duration_seconds: float | None = None   # explicit user control, labeled

@dataclass
class GenerationResult:
    canonical_motion: np.ndarray          # [T_hat,296], physical parameters
    fps: float
    predicted_frames: int
    duration_source: str                   # predicted/user/oracle_diagnostic
    seed: int
    metadata: dict
```

Predict T_hat, L=ceil(T_hat/2), initialize four latent noises and optionally relation noise. Time grid linspace0..1 NFE+1, default16 Euler calls at t_j=j/16,j0..15; no need call network at1. Update `U += dt*v`. Mask pads after every step. Decode denormalized final z, never directly treat normalized latent as motion coefficients. Relation output chỉ plan; final evaluation recompute A from motion.

### 17.2 CFG

`v_guided=v_null+scale*(v_text-v_null)`; scale1 chỉ một conditional forward, không gọi null branch rồi tính lãng phí. scale>1 hai forward-equivalents dù combinebatch. Same language, length, z/t ở hai branches. CFG scale sweep1/1.5/2/3 trên dev, default1. Log ODE solver steps, vector-field calls, batch-equivalent forwards và latency riêng.

Nếu Heun:8steps≈16field calls; sửa time endpoint argument nhất quán. Chỉ chọn final sampler từ dev. Không làm output smoothing mặc định; smoothing ablation có thể phá fingerspelling, phải đo riêng.

### 17.3 Guard algorithm

Chỉ enable khi valid calibration `approved=true` và hashes khớp. Freeze generated relation plan R0, original z0/x0, teacher eval-mode. Energy `E=Huber(A(D(z)),R0)`; penetration weight0 v1. Không reshape distance features thành actual mesh collision flags. Teacher score `S(x,y)=cos(global_motion(x),text_embedding(y))`; part distances `1-cos(P_part(x),P_part(x0))` cho LH/RH/face.

Hai accepted-attempt iterations tối đa, mỗi iteration3backtracking proposals; base step0.05/RMS(grad), halve0.5^attempt. Zero padding in delta. Candidate accept iff all:

1. Finite motion, finite teacher/energy, valid rotation decoding.
2. E(candidate) < E(current)-1e-6 (normalized energy units).
3. S(candidate,y) >= S(original,y)-epsilon_sem.
4. Mỗi part cosine distance so original <= epsilon_part.
5. Latent residual RMS <=0.2 và per-part physical output finite; residual trust là thêm engineering bound của spec v1.

Epsilon khởi đầu0.01 chỉ là scale để calibrate; calibration file phải cung cấp values thật. Confidence gate có thể là score threshold được calibrated, không một sigmoid bất kỳ của cosine. Original confidence dưới threshold → return original và reason. Không proposal nào pass → keep current; constraints luôn so original để không drift. Gradient zero/tiny → stop. Log proposed/accepted steps, Δgeometry, Δsemantic, per-part shifts, skip reason, forward/backward/FK costs.

Không giảm loss bằng sửa R0 theo geometry hiện tại. Không tìm nearest GT pose ở test. Không tự chọn seed/candidate bằng test JPE. Bản guard không giữ probability distribution FM một cách chính xác; báo trước/sau cùng seeds và matched cost.

## 18. Training engine và reproducibility

### 18.1 Optimizer/precision

AdamW, weight_decay0.01; không decay bias/LayerNorm parameters. AE lr2e-4, generator1e-4, teacher/duration1e-4. Warmup5k + cosine floor0.1×initial lr trừ duration early-stop. Clip global grad norm1.0 **sau unscale**, all parameters của một model. BF16 autocast network nếu supported; FP16 với GradScaler fallback; CPU testsFP32. Rotation/FK/variance/probability logs/teacher norms trong FP32. Matmul TF32 setting ghi trong run metadata.

EMA decay0.9999 sau mỗi successful optimizer step; EMA includes trainable model weights và buffers cần thiết, không cập nhật frozen mT5/AE bằng EMA generator. Save raw và EMA. Nếu overflow skip step thì scheduler, EMA, optimizer_step đều không tiến, batch stream advancement policy ghi rõ replay same logical batch để resume deterministic.

### 18.2 Checkpoint content

Checkpoint gồm model, optimizer, scheduler, scaler, EMA, successful_optimizer_step, total_batches_seen, sampler RNG/cursor, rank RNG states (Python/NumPy/TorchCPU/CUDA), resolved config hash, AE/text/duration/teacher/stats/assets/source manifest hashes. Atomic write temporary file → rename, rank0 write; barrier sau completion nếu consumers dùng checkpoint ngay.

Resume strict: mismatch representation/stats/AE/text/feature mapping/world size cần explicit migration/reset flag. Default strict resume chỉ same world size; `weights-only` là fork run mới, không gọi exact resume. Checkpoint test resume tiếp20updates so uninterrupted20 với determinism mode, tolerance phù hợp backend; record nếu CUDA kernels không bitwise deterministic.

### 18.3 Validation/checkpoint selection

- Every1k successful steps: deterministic dev loss panel mỗi dataset64clips, fixed IDs/noise/t; log raw parts và weighted terms.
- Every5k: generate fixed dev prompts20/dataset, seeds101, predicted length,16Euler, same avatar/model assets.
- Generator best checkpoint chọn macro-average **DTW-JPE trên global hand21 positions** qua3datasets trong panel, with duration-error report. Relation head không tham gia score. Đây là proxy selection rule, không semantic-quality certificate.
- Save last every5k và best, giữ ít nhất milestones50k/100k/200k/300k để audit; disk budget configurable, không xóa best hoặc only-resume checkpoint.
- Final selected checkpoint rerun full dev để confirm rồi mở full test một lần theo protocol; không tune bằng test.

AE chọn theo mục11; duration theo12; teacher theo16. Không chọn best generator bằng train loss khi dev samples collapse. Log clear `selection_metric` và `selection_subset_hash`; nếu dev panel dùng ít prompts, nhãn “panel-selected” trong report.

### 18.4 Mandatory logging

step, elapsed time, lr, successful/skipped updates, gradnorm; total/FM motion perpart/FM relation/eachaux/raw/effective losses; t hist, CFG-drop ratio; lengths, corpus sampling counts, padding fraction; data wait/time per update; allocated/reservedVRAM; degenerate6D rate; predicted length clipping; teacher/lag coverage; sample/call seeds. Metrics JSONL machine-readable, no only screenshots of dashboards.

### 18.5 Hardware profiles

| Profile | Flow microbatch/rank | Accumulation | Global batch |
| --- | ---: | ---: | ---: |
| 1 GPU,24–48GB development | 1 | 64 | 64 |
| 4 GPUs,48–80GB starting point | 2 | 8 | 64 |
| AE one GPU | 1 | 128 | 128 |
| AE four GPUs | 4 | 8 | 128 |

Đây là **starting profiles chưa đo**. Chạy profile100warmup+200timed updates, per length bucket. Nếu OOM, giảm microbatch và tăng accumulation giữ global batch. Không giảm L/num_expression/hand joints để “fit” mà không đổi experiment profile. Lower computational preset hidden256/layers 6 là `small` control riêng, không override main hidden384 âm thầm. Không ép full confirmatory runs khi data/AE gate chưa đạt.

## 19. Stages và thứ tự train trên ba dataset

| Stage | Run ID | Training data | Updates / dependency |
| --- | --- | --- | --- |
| Smoke | `smoke_real48` | 16train clips/corpus, complete short utterances | AE1k + CFM1k, chỉ engineering |
| AE | `ae_joint3_s17` | Joint3 train, balanced | Max200k; geometry G0 trước |
| Duration | `duration_joint3_s17` | Joint3 train text/length | Max30k, fixed encoder |
| Cache | `cache_joint3` | Train/dev/test encoded riêng | AE/text/stats hashes cố định; không học stats từ dev/test |
| B1-pre | `cfm_joint3_parent_s17` | Joint3 train | 200k parent; source của mọi fork |
| B1-cont | `cfm_joint3_s17` | Joint3 train | Parent EMA@200k +100k với optimizer reset |
| C0 | `relation_joint3_s17` | Joint3 train | B1@200k +100k continuation |
| B2 | `auxonly_joint3_s17` | Joint3 train | B1@200k +100k; no relation head, geom/lag available |
| B3 | `capacity_joint3_s17` | Joint3 train | Match params/FLOPs as measured, same schedule |
| B0 | `diffusion_joint3_s17` | Joint3 train | Same300k budget, tuned DDIM |
| Mono controls | `cfm_<dataset>_sharedAE_s17` | Một train corpus/run | Shared joint3 AE/stats/text;300k, label shared-AE clearly |
| Teacher | `teacher_joint3_s17` | Joint3 train only |100k + calibration dev |
| Full | `full_joint3_s17` | Joint3 train only | Separate semantic-aux continuation + guard calibration; compare equal total budgets |

Trong bảng này G0 là data/geometry audit M1–M2. Default delivery priority tới C0 + evaluation trên ba datasets. B0/B2/B3/mono là confirmatory controls sau khi baseline stable. Teacher/guard không chặn các stages này. Replicate key B1/C0/full với seeds42,123 sau screening; không chạy mọi ablation×mọi seed trước khi biết hệ thống hoạt động.

Full run nếu thêm continuation **phải có no-semantic control cùng số added updates**; không so400k với300k rồi quy gain cho semantic objective. Weights của teacher/AE/stats dùng chung cho generator-seed comparison, báo rõ đây không đo variance của toàn pipeline.

## 20. Config hoàn chỉnh cần hỗ trợ

File mẫu dưới đây là config defaults của implementation mới; schema phải validate đầy đủ và ghi resolved copies trong run. Paths cần map vào máy lab, asset/revision values phải resolve M0, không giả đã có.

```yaml
spec_version: cprflow-impl-v1
seed: 17
run:
  id: cfm_joint3_parent_s17
  out_dir: runs
  task: flow
  variant: plain_cfm
  strict_reproducibility: true
  synthetic: false
paths:
  source_pose: data/raw_pose
  manifests: data/manifests
  processed: data/processed
  cache: data/cache
  stats: data/stats
  smplx_model: assets/smplx/SMPLX_NEUTRAL.npz
  sources_lock: sources.lock.json
  ae_checkpoint: runs/ae_joint3_s17/checkpoints/best_ema.pt
  duration_checkpoint: runs/duration_joint3_s17/checkpoints/best.pt
  teacher_checkpoint: null
  teacher_calibration: null
data:
  datasets: [csl_daily, how2sign, phoenix2014t]
  q_dataset: [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]
  sign_language_ids: {CSL: 0, ASL: 1, DGS: 2}
  profile: canonical296_nativeleft_row6d_v1
  dims: {body: 60, left: 90, right: 90, face: 56}
  require_verified_feature_fps: true
  target_fps: 30.0
  resample_grid: endpoint_linspace_v1
  max_train_frames: 1024
  max_generation_frames: 2048
  bucket_boundaries: [128, 256, 512, 1024]
  truncate_motion: false
  mirror_augmentation: false
  random_time_stretch: false
  workers_per_rank: 4
  pin_memory: true
stats:
  method: equal_dataset_equal_clip_moments
  fit_split: train
  std_floor: 0.01
geometry:
  backend: smplx_layer_matrix
  num_betas: 10
  num_expression: 50
  flat_hand_mean: true
  joint_mapper: null
  face_contour: false
  root: identity
  translation: zero
  shape: neutral_zero
  fk_chunk_frames: 128
  epsilon: 0.000001
ae:
  latent_dim_per_part: 128
  stride: 2
  channels: [128, 256]
  residual_dilations: [1, 2]
  norm: channel_layernorm
  boundary: true_length_replicate
  decoder_fusion: residual_mlp512_256_512
  kl_weight: 0.0
  weights: {param: 1.0, rotation: 0.5, fk: 1.0, velocity: 0.1}
  local_hand_fk_weight: 2.0
  auxiliary_frame_stride: 2
text:
  model_id: google/mt5-base
  use_encoder_only: true
  frozen: true
  hidden_dim: 768
  max_tokens: 512
  prefix_by_sign_language: true
  condition_dropout: 0.1
duration:
  distribution: log_normal_frames
  input_dim: 800
  hidden_dim: 1024
  sigma_floor: 0.05
  sigma_max: 2.0
  round_frames_to: 2
model:
  hidden_dim: 384
  blocks: 8
  heads: 6
  ffn_dim: 1536
  dropout: 0.0
  temporal_attention: full_per_stream
  local_cross_part_radius_latent: 4
  global_part_context_every: 2
  modulation: adaln_zero
relation:
  enabled: false
  dim: 33
  pooling: valid_pair_mean
  threshold: 0.08
  softness: 0.02
  warmup_key_gate_updates: 2000
flow:
  path: linear_noise_to_data
  t_distribution: uniform
  fm_motion_weight: 1.0
  fm_relation_weight: 0.25
  coupling: independent
loss:
  reduction: global_mean_of_per_clip_valid_means
  geom: 0.0
  cycle: 0.0
  lag: 0.0
  semantic: 0.0
  auxiliary_every_updates: 4
  auxiliary_frequency_compensation: 4.0
  auxiliary_time_min: 0.5
  auxiliary_ramp_start: 20000
  auxiliary_ramp_end: 40000
  lag_window: 32
  lag_hop: 16
  lag_max_offset: 15
  lag_temperature: 0.1
  lag_variance_floor: 0.0001
train:
  max_updates: 200000
  effective_batch: 64
  microbatch_per_rank: 1
  accumulate: 64
  optimizer: AdamW
  lr: 0.0001
  weight_decay: 0.01
  warmup_updates: 5000
  lr_floor_ratio: 0.1
  grad_clip: 1.0
  precision: bf16_if_supported
  ema_decay: 0.9999
  dev_loss_every: 1000
  dev_generation_every: 5000
  checkpoint_every: 5000
  selection: macro_dev_hand_global_dtw_jpe
distributed:
  backend: nccl
  find_unused_parameters: false
  strict_resume_world_size: true
sample:
  solver: euler
  nfe: 16
  cfg: 1.0
  duration_source: predicted
  oracle_inputs: false
guard:
  enabled: false
  max_steps: 2
  max_backtracking: 3
  step_size: 0.05
  max_latent_delta_rms: 0.2
  energy_improvement_min: 0.000001
  penetration_weight: 0.0
eval:
  split: dev
  per_dataset: true
  sample_seeds: [1234, 1235, 1236, 1237, 1238]
  recompute_relations_from_geometry: true
  best_of_n_with_gt: false
  backtranslation_checkpoint: null
  official_metrics_adapter: null
```

Overrides có kiểm soát:

- AE run: `task=ae`, lr2e-4, max200k, effective_batch128 và accumulation phù hợp.
- Relation run: `variant=relational_cfm`, relation.enabled=true, geom0.1/cycle0.05/lag0.01, warm-start path B1@200k, train.max_updates=100000 **continuation-relative**; save both parent_global_step và continuation_step. Semantic0/guardfalse mặc định.
- Full run: teacher paths populated, approved calibration, semantic0.05/guardtrue; no-semantic control cùng extra updates.
-4GPU: micro2,accumulate8,effective64; validator require `micro*accumulate*world==effective_batch`.

YAML example không bao gồm mọi model-specific key của teacher/diffusion; các dataclasses của chúng phải nhận defaults từ mục15–16, export đầy đủ vào resolved config. Tuyệt đối không để một config trông như full-run nhưng loss.semantic0/guardfalse mà report “full”.

## 21. CLI và runbook AI agent phải tạo

Các commands sau là **acceptance interface của repo mới**, chỉ chạy được sau khi milestones tương ứng được implement. `--set dotted.key=value` parse theo schema, không arbitrary eval.

### 21.1 Prepare/audit

```bash
python -m cprflow.cli.audit_sources --config configs/base.yaml --resolve-lock
python -m cprflow.cli.prepare --config configs/data/joint3.yaml --phase import
python -m cprflow.cli.prepare --config configs/data/joint3.yaml --phase verify-timing
python -m cprflow.cli.prepare --config configs/data/joint3.yaml --phase canonicalize
python -m cprflow.cli.prepare --config configs/data/joint3.yaml --phase audit-geometry --samples-per-dataset 64
python -m cprflow.cli.fit_stats --config configs/data/joint3.yaml --kind motion-relation --split train
```

Mỗi phase idempotent theo source/preprocess hash. Nếu file đã đúng hash skip; nếu khác version ghi thư mục mới, không overwrite raw. `verify-timing` không tự đoán source FPS khi thiếu evidence. `audit-geometry` export joints/parameters và contact sheets/video tham chiếu, không chỉ `assert shape`.

### 21.2 AE, text và duration

```bash
python -m cprflow.cli.train --config configs/runs/smoke.yaml --task ae
python -m cprflow.cli.train --config configs/runs/ae_joint3.yaml --task ae
python -m cprflow.cli.cache --config configs/runs/ae_joint3.yaml --kind text --splits train dev test
python -m cprflow.cli.train --config configs/runs/duration_joint3.yaml --task duration
python -m cprflow.cli.fit_stats --config configs/runs/ae_joint3.yaml --kind latent --split train
python -m cprflow.cli.cache --config configs/runs/ae_joint3.yaml --kind latent-relation --splits train dev test
```

Encode dev/test bằng **frozen train-fit** preprocessing/statistics là hợp lệ; không optimize bất kỳ weights/statistics nào bằng chúng. Cache mode train/test metadata không cho phép vô tình sample test trong train loader.

### 21.3 Baseline và relational

```bash
python -m cprflow.cli.train --config configs/runs/baseline_joint3.yaml --task flow
torchrun --standalone --nproc_per_node=4 -m cprflow.cli.train \
  --config configs/runs/baseline_joint3.yaml --task flow \
  --set train.microbatch_per_rank=2 --set train.accumulate=8
python -m cprflow.cli.train --config configs/runs/baseline_cont_joint3.yaml --task flow \
  --warm-start runs/cfm_joint3_parent_s17/checkpoints/step_200000_ema.pt
python -m cprflow.cli.train --config configs/runs/relation_joint3.yaml --task flow \
  --warm-start runs/cfm_joint3_parent_s17/checkpoints/step_200000_ema.pt
```

Exact resume cần raw optimizer/model/RNG state tại200k; weights-only warm-start là một fork mới. CLI `--warm-start` nêu trên khởi tạo weights-only từ EMA, reset optimizer **cho mọi continuation controls** nếu chọn setting này. `--resume` khác `--warm-start`: resume phải giữ optimizer/RNG. Đừng khởi tạo C0 từEMA còn B2/B3 từraw mà không ghi khác biệt. Authoritative default cho controlled fork: **EMA@200k + reset optimizer/scheduler cho B1-cont/B2/B3/C0**, cùng100k continuation, warmup2k rồi cosine; full parent path lưu provenance. Uninterrupted B1@300k là reference bổ sung, không thay B1-cont trong fork comparison. Default base config chạy parent200k rồi các fork100k; tổng exposure300k/run. Không chạy thêm uninterrupted300k như một dependency bắt buộc.

### 21.4 Generation và evaluation

```bash
python -m cprflow.cli.sample --checkpoint runs/cfm_joint3_s17/checkpoints/best_ema.pt \
  --text-file prompts/asl_demo.jsonl --output runs/demo --nfe 16 --cfg 1.0
python -m cprflow.cli.evaluate --checkpoint runs/cfm_joint3_s17/checkpoints/best_ema.pt \
  --dataset csl_daily --split dev --duration-source predicted
python -m cprflow.cli.evaluate --checkpoint runs/cfm_joint3_s17/checkpoints/best_ema.pt \
  --dataset how2sign --split dev --duration-source predicted
python -m cprflow.cli.evaluate --checkpoint runs/cfm_joint3_s17/checkpoints/best_ema.pt \
  --dataset phoenix2014t --split dev --duration-source predicted
python -m cprflow.cli.evaluate --checkpoint runs/cfm_joint3_s17/checkpoints/best_ema.pt \
  --dataset csl_daily --split dev --duration-source oracle --diagnostic-only
python -m cprflow.cli.profile --config configs/runs/baseline_joint3.yaml --warmup 100 --updates 200
```

Prompts JSONL cần `{request_id,text,sign_language,seed}`. Request file không chứa target motion/keyframes/gloss. Optional user duration phải gắn `duration_source=user` trong result; không gộp với predicted-duration score.

Mỗi request có `torch.Generator` riêng theo seed; tạo noise đúng true shape trước batch padding. Không dùng Python `hash()` làm stable UID seed. Batch grouping không được đổi noise của một request.

Sampling output mỗi UID/seed: NPZ motion296 physical, timestamps, optional recomputed joints; metadata JSON có checkpoint+stats+assets hashes, NFE/CFG/duration source/costs; optional renderedMP4. Raw latent/R_head để debug nằm ở subdir riêng. Không output pickle bắt người dùng deserialize arbitrary code để xem motion.

## 22. Evaluation implementation

### 22.1 Sets và protocols

Mỗi dataset có metrics JSON riêng, không gộp English/Chinese/German vào BLEU. Joint3 summary chỉ macro-average các metric đã cùng units/implementation. Prefix names tách:

- `controlled_v1_*`: implementation ở đây, cùng pose release/asset/evaluator.
- `official_<paper>_*`: chỉ dùng khi đã đọc/port official evaluator và xác minh mapping/units/normalization.
- `oracle_*`: diagnostics, tuyệt đối không xuất vào main comparison table.

Không gọi một metric homemade là số SOKE/SignSparK-reproduction chỉ vì cùng tên DTW-JPE. Official back-translator có thể đòi representation khác296; adapter phải được test bằng real fit samples và cùng recognizer cho mọi model. Thiếu code/checkpoint → `metric_status=unavailable`, không trả0 hoặc dùng generated caption LLM thay BLEU.

### 22.2 Geometry metrics

All positions từ same neutral SMPL-XLayer, units metres; display mm/cm bằng explicit conversion factor. Body group10 upper joints, hand group mỗi21 gồm wrist+15 joints+5tips. FK helper export names, không indices ngầm. Global-hand JPE và relative-to-wrist hand20 JPE phải báo riêng. Không lấy PA-hand tốt để suy ra đúng signing-space location.

Variable predicted length: exact DTW cho frame-level group positions, cost(i,j)=mean joint Euclidean distance. Allowed steps(1,0),(0,1),(1,1), full path từ(0,0) tới(Tpred-1,Tgt-1), objective min cumulative cost. Report **path-average** cost trên selected min-sum path, cùng path length. Tie-break diagonal→vertical→horizontal deterministic. Complexity O(Tpred*Tgt), float32/64 CPU; dùng rolling cost và backpointers nếu cần. Banding là different protocol, disabled mặc định.

Unaligned DTW-JPE primary geometry; PA variant nếu implement, dùng per-frame Kabsch proper rotation det+1, no reflection; similarity-scale variant có tên riêng `pa_similarity`, rigid-only `pa_rigid`. Align group trước DTW theo documented algorithm; không mix scale/no-scale. Khi đối chiếu literature phải match chính xác protocol nguồn. DTW reconstruction AE có frame alignment nên dùng direct no-DTW JPE và rotation errors làm primary; thêmDTW chỉdiagnostic.

Relation error tính **A(decoded motion)** so A(reference fit), với temporal alignment mapping từ same decoded positions, không so R_head với R_target làm primary. Report components position/normal/distance/proximity riêng trước normalized composite. Physical contact F1/penetration unavailable nếu chưa có mesh labels/validated implementation; proxy threshold không tự trở thành contact annotation.

### 22.3 Temporal/distributional/semantic

- Duration relative error `abs(Tpred-Tgt)/Tgt`, MAE frames/seconds, clipped/failure rates.
- Head/hand angular velocity, high-frequency energy attenuation và lag error trên valid target windows; report native vs resampled reference để thấy fitting/resampling ceiling.
- Sample5 seeds/prompt; primary automatic score average seeds, không best seed theoGT. Conditional diversity trên same text và valid semantic subset khi có annotation.
- Retrieval/FID chỉ dùng sign-specific encoder đã freeze và công bố training split/feature version; FID fixed sample count và real-vs-real/reference-fit baseline. Hệ teacher dùng train model không phải independent evaluator; luôn báo hạn chế này. Không dùng HumanML3D evaluator vào296D bằng padding/truncate.
- BT adapter có real-fit/AE/model sanity suite trước: text shuffle, reverse time, zero face, strong smoothing. Nếu không nhạy với NMF thì không dùng nó để claim NMF quality.

### 22.4 Metrics JSON

```json
{
  "protocol": "controlled_v1",
  "dataset": "csl_daily",
  "split": "dev",
  "run_id": "cfm_joint3_s17",
  "checkpoint_sha256": "<resolved actual hash>",
  "manifest_sha256": "<resolved actual hash>",
  "duration_source": "predicted",
  "seeds": [1234, 1235, 1236, 1237, 1238],
  "units": "metres",
  "num_requested": null,
  "num_successful": null,
  "coverage": null,
  "metrics": {},
  "unavailable_metrics": [],
  "failure_reasons": {},
  "oracle": false
}
```

Nulls/hashes trên chỉ là schema example, **không là run result**. Actual report phải có realcounts/hashvalues. Aggregate script reject mixed duration_source, fitting/profile, metric versions hoặc oracle/nonoracle trong cùng main table. Report bootstrap perprompt/sourcevideo khi biếtsourcevideo, training-seed variability riêng. Không coi5samplings của1model như5independent trainingruns.

## 23. Tests và acceptance criteria

Các tests sau kiểm tra hành vi có thể làm hỏng nghiên cứu; không chỉ test class name/shape tồn tại.

| Test | Fixture / assertions |
| --- | --- |
| Rotation identity | Identity6D ↔ matrix; det1, RRᵀI; zero6D fallback finite/logged |
| Rot6D convention | Known90-degree matrix, first2rows round trip; transpose bug phải fail |
| SO(3) edge gradients | Identity, small angle, nearpi geodesic losses/backprop finite |
| Native-left | Asymmetric known pose; no accidental doubleflip; source↔canonical roundtrip |
| Body mapping | Lower11 identity, upper10 indices exact; jaw independent head;50expr retained |
| Schema |126→60 body slicing; reject unknown layout/face10; mismatched T/quarantine |
| Resampling | Known angular trajectory0→90SLERP, endpoint/duration preserved, coefficient interpolation |
| Dataset leakage | Manifest train/dev/test disjoint UID; no fit_stats on nontrain; cache writes no weights update |
| AE padding | Same true clip with different padding/otherbatchclips same valid output and gradients |
| Relation33 | Dimension/order; rigid global-transform invariance; left/right swap changes appropriate slices |
| Pool2 | Odd length handling; A includes pool-after-relation; cycle aligns identical timestamps |
| CFM path | t0epsilon,t1target; finite-difference derivative U1-eps; exact endpoint reconstruction with targetv |
| FM reduction | Appending huge errors in padding leaves loss unchanged; longclip not overweighted |
| Attention masks | Pad perturbation cannot change valid outputs; all-masked query finite/zero |
| Decoder autograd | FK→predicted latent grad nonzero/finite, frozen AE weights gradNone |
| Lag | Second signal delayed3→peak+3; static target window excluded, staticprediction not excluded |
| CFG | scale1conditionalexact; null branch keeps lang/length, no empty-mask NaN |
| Warm-start | C0 alphaR0 copies B1 motion outputs ≤1e-5FP32 |
| DDP/accum | Compare globalbatch gradients to shards/accum; no factorW or factoraccum error |
| DDIM indexing | alpha_bar0=1; k sampled1..K; deterministic eta0; no division0/network callk0 |
| Resume |20steps resumed vs uninterrupted same nextIDs/LR/EMA and compatible weights |
| No oracle | Main generation signature forbids target fields; evaluator spies cannot pass GTlen/keyframes |
| Guard | Crafted candidate reducesgeometry but breaks semanticbudget → reject; rollback exact; baseline unmodified |
| Provenance | Changing AE/stats/pose revision invalidates relevant caches/checkpoints |

CPU synthetic tests không cần body assets ngoại trừ mapping/FK. Tests với licensed assets gắn marker `requires_body_model`, skip với exact missing-path reason, không report passed. DDP tests2CPU gloo có thể kiểm tra reduction;2GPU NCCL integration thêm khi cóGPU. Library SMPLXLayer matrices/gradients phải có asset-backed test trướctraining geometry.

Engineering gates:

1. **M0:** import package, parse config, dependency/source lock có hashes thật; unit math tests pass.
2. **M1–M2:** import9LMDBs (3datasets×3splits), audit/quarantine/coverage report; native render64clips/corpus; differentiable geometry tests pass.
3. **M3:** AE48realclips smoke loss giảm, gradients finite; validation renders không jointflip/facecollapse do mapping; fullAE checkpoint+reconstructionreport.
4. **M4:** textcache768D, duration metrics và latentstats trainonly; no stalecache.
5. **M5:** CFM smoke and real run resume/sampling works; no-oracle test; same prompt+seed deterministic in eval mode.
6. **M6:** C0 warmstart, losses nonzero as expected, relation geometry error computed independently; baseline/fork budget matched.
7. **M7:** teacher/calibration/guard interfaces+tests; annotations thiếu → feature-only/disabled đúng nhãn.
8. **M8–M9:** separate metrics for3datasets, failurecoverage, efficiency profile và comparison tables reproducible.

Loss giảm trên smoke không chứng minh ký hiệu đúng. Chỉ đánh dấu linguistic gate sau đánh giá fluent signers/contrasts; cấm suy kết luận đó từ unit tests.

## 24. Experiment matrix cần xuất

| Row | Training / inference | Câu hỏi |
| --- | --- | --- |
| B0 | SharedAE + diffusion300k, DDIM tuned | Objective/sampler baseline |
| B1-pre | SharedAE + plainCFM200k | Parent warmstart, không so với300k như matched result |
| B1-cont | B1-pre EMA resetoptimizer +100k | Main continuation control |
| B2 | Samefork + geom/lag loss, noRstream | Có cần co-generated relations không? |
| B3 | Samefork + extra learnedtokens/capacity, no semanticrelationlabels | Capacity/attention-token confound |
| C0 | Samefork + Rstream + geom/cycle/lag | Relational contribution |
| C1 | C0 + unrestricted geometriccorrection | Geometry/semantic trade-off |
| C2 | C0 + part-feature guard | Featurepreservation control |
| C3 | Calibrated semanticfull +guard | Main full hypothesis khi cólabels |
| Mono×3 | SamejointAE, text/stats, generator trainonecorpus | Transfer of generator training only |
| OracleT | Same checkpoint, testGTduration | Upperbound, diagnosticonly |
| OracleR | Samecheckpoint, GTrelationplan | Upperbound, diagnosticonly |

B3 implement extraP5 latent stream33D cùng attention schedule nhưng target sampled random fixed noise/control signal **không chứa motion relation**; tune hidden/FFN nếu cần để report params/FLOPs gầnC0. Nó là negative/control run, không production. Chỉ thêm untrained dummyparameters để match paramcount là không đủ match compute. Keep compute-report with exact tensor counts.

Human confirmatory endpoint: comprehension/minimalpair accuracy, geometry co-primary. Proposed noninferiority margin3pp cần pilot/poweranalysis chốt trước test. C0/C3 geometry gain nhưng comprehension giảm vượtmargin → bác bỏ mainclaim. Metric thiếu/NMFfittingyếu → giới hạnclaim, không đổi endpoint sau khi xem kết quả.

## 25. Failure handling và những lỗi cấm

| Vấn đề | Hành động bắt buộc |
| --- | --- |
| Thiếu sourceFPS | Chặn timed preprocessing cho corpus, ghi evidence cần có; không guess |
| Thiếu face50/basis phù hợp | Chặn full296track; có thể chạy explicit manual-onlydebug, không cùng report |
| SMPLX npz không đủexpression50 | Dùng correctasset; không truncate xuống10 |
| Empty aux mask | Zero differentiableterm, log0coverage; không NaN và không skiplogicalstep |
| OOM | Lower microbatch/raiseaccum/checkpoint/FKchunk; giữglobalbatch+utterancecoverage |
| Generatedinvalid6D | Deterministicfallback + diagnosticcount; không replacebằngGT |
| Unknownofficialmetric | Mark unavailable; không inventBTcheckpoint/score |
| Cachehashmismatch | Rebuildderivedcache; không ignorewarning |
| DDP rankdatafailure | Coordinatedfail/retry, log UID; không mộtrankcontinue gâydeadlock |
| Guardcalibrationthiếu | Guarddisabled và không fullsemanticclaim |
| TestOOM/longinput | Failureledger+coverage; không dropkhôngbáo |

Không copy nguyên upstream cropping, handflip hoặc SciPy renderer conversion vào training chỉ vì demo chạy. Không “fix” phonological differences bằng post-smoothing. Không dùng extra corpus, pretrained generator rộng hơn hoặc GT anchors mà vẫn giữ nhãn controlledtext-only. Không ghi fakebenchmarknumbers/examplehashes vào actualmetrics.

## 26. Definition of Done của coding agent

Repository được coi implementation-complete khi:

- [ ] M0–M6 modules/CLI/tests implemented và artifacts thực đã được ghi hoặc blockreason cụ thể.
- [ ] M7 teacher/guard đã có code/interfaces/tests; gate thiếuannotations được xử lý đúng, không mockapproval.
- [ ] Dataset import/checkpoint sampling/evaluation dùng đúng3corpora, native conventions và per-datasetprovenance.
- [ ] `README.md` có exact install/run commands đã chạy, asset downloadlinks và hardwareprofile đo được.
- [ ] `IMPLEMENTATION_STATUS.md` phân biệt implemented/tested/realdata-run/trained/evaluated/blocked; mỗi hàng trỏlog.
- [ ] Một optimizerresume integration và sample output thực có thể reproduce.
- [ ] Confirmatory training results chỉ có khi thực sự chạy; nếu budgetkhôngcho phép, ghi pending, không đưa sốđẹp placeholder.
- [ ] `THIRD_PARTY_NOTICES.md` nêu filesadapted, licenses và citations; không redistributelicensedassets/rawdata.
- [ ] Experimentalrunner rejectOracle/nonoracle/profilemix và báo coverage/failures.

“Implementation-complete” không đồng nghĩa “SOTAvalidated”. Người thực hiện có thể hoàn thành code trước khi main training/human study xong, nhưng phải gắn trạng thái trung thực.

## 27. Nguồn đầy đủ cho AI coding agent

Không cần đọc lại mọi paper trước M0; ưu tiên R01/R03/T01/T03 rồi P01/P03. Các links là nguồn sơ cấp hoặc official implementation/docs. File-level hashes của những source đã đọc được ghi ở mục28; chúng nhận diện snapshot tham khảo, không thay việc checkout/pin dependencies trên máy lab.

### 27.1 Repositories và files

- **[R01] SignSparK, official code/data schema** — [repository](https://github.com/JianHe0628/SignSparK), [DATA.md](https://github.com/JianHe0628/SignSparK/blob/main/DATA.md), [LMDB loader/rot6D/hand conventions](https://github.com/JianHe0628/SignSparK/blob/main/signspark/pose_datasets_lmdb.py), [downloader](https://github.com/JianHe0628/SignSparK/blob/main/tools/download_data.py), [SMPL-X visualization assembly](https://github.com/JianHe0628/SignSparK/blob/main/tools/visualize.py), [flow path](https://github.com/JianHe0628/SignSparK/blob/main/signspark/rflow.py), [sampler](https://github.com/JianHe0628/SignSparK/blob/main/signspark/sampling_rflow.py), [LICENSE](https://github.com/JianHe0628/SignSparK/blob/main/LICENSE). Scope: parser/format/reference; method mới không phải exact SignSparK reproduction.
- **[R02] MARDM, official continuousAE/transport implementation** — [repository](https://github.com/neu-3dvi/MARDM), [AE.py](https://github.com/neu-3dvi/MARDM/blob/main/models/AE.py), [model](https://github.com/neu-3dvi/MARDM/blob/main/models/MARDM.py), [transport](https://github.com/neu-3dvi/MARDM/tree/main/diffusions/transport), [AE training](https://github.com/neu-3dvi/MARDM/blob/main/train_AE.py), [LICENSE.txt](https://github.com/neu-3dvi/MARDM/blob/main/LICENSE.txt). Scope: implementation patterns; không dùng HumanML3D data/evaluator cho signing.
- **[R03] SMPL-X, official library** — [repository](https://github.com/vchoutas/smplx), [SMPLXLayer/body_models.py](https://github.com/vchoutas/smplx/blob/main/smplx/body_models.py), [joint_names.py](https://github.com/vchoutas/smplx/blob/main/smplx/joint_names.py), [vertex_joint_selector.py](https://github.com/vchoutas/smplx/blob/main/smplx/vertex_joint_selector.py), [LBS](https://github.com/vchoutas/smplx/blob/main/smplx/lbs.py), [official model download/license](https://smpl-x.is.tue.mpg.de/). Scope: differentiable body geometry và mapping. Assets không bundled.
- **[R04] SOKE, reference baseline/fittings** — [repository](https://github.com/2000ZRL/SOKE), [project/data links](https://2000zrl.github.io/soke/), [license](https://github.com/2000ZRL/SOKE/blob/main/license.txt). Không cần fork để chạy repo mới; đọc terms trước adapt/distribute.
- **[R05] mT5-base model** — [model card/files](https://huggingface.co/google/mt5-base), [config.json](https://huggingface.co/google/mt5-base/blob/main/config.json). `d_model=768` trong config đã kiểm tra; pin weights/tokenizer revision cùng nhau.
- **[R06] Official backtranslation release** — [SignSparK_BT](https://github.com/JianHe0628/SignSparK_BT), [model hub](https://huggingface.co/LionelLow/SignSparK_BT). Source R01 README trỏ đến release; coding agent phải audit actual schema/checkpoint trước khi dùng. Không assume BT đọc 296D trực tiếp.
- **[R07] DiT official implementation** — [repository](https://github.com/facebookresearch/DiT). Architecture pattern source; kiểm tra license và không copy image patch tokenization như motion tokens.
- **[R08] Meta flow_matching** — [repository](https://github.com/facebookresearch/flow_matching). Optional reference paths/solvers, không required mặc định. Audit trước ghi CC BY-NC license, không gọi là MIT.

### 27.2 Datasets / pose release

- **[D01] CSL-Daily official dataset/access** — [USTC page](https://ustc-slr.github.io/datasets/2021_csl_daily/). Sourcepaper: [Improving Sign Language Translation with Monolingual Data by Sign Back-Translation, CVPR2021](https://openaccess.thecvf.com/content/CVPR2021/html/Zhou_Improving_Sign_Language_Translation_With_Monolingual_Data_by_Sign_Back-Translation_CVPR_2021_paper.html).
- **[D02] How2Sign official dataset** — [website/download metadata](https://how2sign.github.io/), [CVPR2021paper](https://openaccess.thecvf.com/content/CVPR2021/html/Duarte_How2Sign_A_Large-Scale_Multimodal_Dataset_for_Continuous_American_Sign_Language_CVPR_2021_paper.html). Toàn corpus không phải Panoptic 3D MoCap.
- **[D03] PHOENIX-2014T official dataset** — [RWTHpage](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/), [Neural Sign Language Translation,CVPR2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Camgoz_Neural_Sign_Language_CVPR_2018_paper.html). Không nhầm PHOENIX-2014 và PHOENIX-2014T.
- **[D04] Reconstructed source for v1** — [LionelLow/SignSparK_data](https://huggingface.co/datasets/LionelLow/SignSparK_data), [dataset card raw](https://huggingface.co/datasets/LionelLow/SignSparK_data/raw/main/README.md), [files](https://huggingface.co/datasets/LionelLow/SignSparK_data/tree/main). Card ghi source-dataset terms và license other; resolve/pin revision, không tải generator weights nhầm thành data.

### 27.3 Papers: objective/architecture/evaluation

- **[P01] Flow Matching for Generative Modeling**, ICLR2023 — [paper](https://arxiv.org/abs/2210.02747). Conditional path, velocity target và training equivalence trước auxiliary guidance.
- **[P02] Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport**, TMLR2024 — [paper](https://arxiv.org/abs/2302.00482). Optional OT theory; default implementation không OT.
- **[P03] Scalable Diffusion Models with Transformers**, ICCV2023 — [paper](https://arxiv.org/abs/2212.09748). adaLN-Zero/block conditioning.
- **[P04] SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers**,2024 — [paper](https://arxiv.org/abs/2401.08740). Phân biệt backbone/path/objective/sampler.
- **[P05] JointHOI: Jointly Generating Contact Maps Enhances Hand Object Interaction Generation**,2026 — [paper](https://arxiv.org/abs/2607.01768). Tiền lệ joint motion–relation generation; không chứng minh sign semantics.
- **[P06] Representation Alignment for Generation**, ICLR2025 — [paper](https://arxiv.org/abs/2410.06940). Frozen representation targets và gradient through noisy features.
- **[P07] SignCLIP**, EMNLP2024 — [paper](https://aclanthology.org/2024.emnlp-main.518/). Sign-specific cross-modal alignment; không mặc định là pose teacher đã calibrated.
- **[P08] Targeted Linguistic Analysis of Sign Language Models with Minimal Translation Pairs**,2026 — [paper](https://arxiv.org/abs/2604.27232). Contrastive evaluation và blind spots về non-manual features.
- **[P09] Denoising Diffusion Probabilistic Models**,NeurIPS2020 — [paper](https://arxiv.org/abs/2006.11239).
- **[P10] Denoising Diffusion Implicit Models**,ICLR2021 — [paper](https://arxiv.org/abs/2010.02502).
- **[P11] SignSparK**,ECCV2026 — [version4](https://arxiv.org/abs/2603.10446v4). Data/profile/conditioning regimes phải phân biệt.
- **[P12] SOKE**,ICCV2025 — [version3](https://arxiv.org/abs/2411.17799v3). Baseline 3D/tokenizer/retrieval.
- **[P13] M3T**,BMVC2026 — [version2](https://arxiv.org/abs/2603.23617v2). Facial fitting bottleneck; không pad face10 thành50/108.
- **[P14] SignNMFlow**,2026 — [paper](https://arxiv.org/abs/2609.33339). Manifold-path baseline là extension khác latent CFM.
- **[P15] MaDiS**,2026 — [version2](https://arxiv.org/abs/2601.19577v2). Discrete diffusion/evaluation context, không bắt buộc implementation.
- **[P16] BLANC**,2026 — [paper](https://arxiv.org/abs/2610.24109). Multirate/adaptive-block prior; spec v1 này không implement adaptive blocks.
- **[P17] SeRV**,2026 — [version2](https://arxiv.org/abs/2609.05742v2). Semantic tokenization prior; không là dependency CFM.
- **[P18] MARDM**,CVPR2025 — [paper](https://openaccess.thecvf.com/content/CVPR2025/html/Meng_Rethinking_Diffusion_for_Text-Driven_Human_Motion_Generation_Redundant_Representations_Evaluation_CVPR_2025_paper.html). Representation/evaluation confounds.
- **[P19] MotionHiFlow**,CVPR2026 — [paper](https://arxiv.org/abs/2604.23264). Optional multiscale control sau baseline.

### 27.4 Official engineering docs

- **[T01] PyTorch scaled dot-product attention/mask semantics** — [SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html), [MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html).
- **[T02] PyTorch distributed training** — [DDPdesignnote](https://docs.pytorch.org/docs/stable/notes/ddp.html), [DDPAPI](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html), [distributedoverview](https://docs.pytorch.org/docs/stable/distributed.html).
- **[T03] Transformers mT5** — [modeldocumentation](https://huggingface.co/docs/transformers/model_doc/mt5).
- **[T04] PyTorch AMP/gradient accumulation** — [AMPexamples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html), [AMPAPI](https://docs.pytorch.org/docs/stable/amp.html).
- **[T05] HuggingFace Hub reproducible downloads** — [downloadguide](https://huggingface.co/docs/huggingface_hub/guides/download), [snapshot_downloadAPI](https://huggingface.co/docs/huggingface_hub/package_reference/file_download).
- **[T06] PyTorch reproducibility/checkpoints** — [randomnessnotes](https://docs.pytorch.org/docs/stable/notes/randomness.html), [saving/loading](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).

Stable documentation có thể redirect version mới. Khi implement, đối chiếu đúng version đã pin; không copy args từ API mới vào version cũ mà không test.

## 28. Snapshot source evidence và trạng thái của specification

Các file ở bảng dưới đã đọc qua raw official URLs trong phiên này; SHA256 xác định nội dung snapshot. Không bao gồm raw corpus, weights hoặc license-gated body model. Coding agent lưu source lock thực tế của mình và giải thích nếu upstream đã đổi.

| Snapshot | URL | SHA256 của file đã đọc |
| --- | --- | --- |
| `spark_loader` | [official file](https://raw.githubusercontent.com/JianHe0628/SignSparK/main/signspark/pose_datasets_lmdb.py) | `440cd965409898a44f365567b0e0990fd133908c70a7bc39513703e189a81092` |
| `spark_data` | [official file](https://raw.githubusercontent.com/JianHe0628/SignSparK/main/DATA.md) | `bd26f64571898aa7c91ef04116e37ded13745021f212cc5cf1966cad4dd46f2f` |
| `spark_download` | [official file](https://raw.githubusercontent.com/JianHe0628/SignSparK/main/tools/download_data.py) | `174b82279cf1734a60b1936a24c616ba510a9307a1e5bca8280c538022875088` |
| `spark_vis` | [official file](https://raw.githubusercontent.com/JianHe0628/SignSparK/main/tools/visualize.py) | `2a7af5a5cab4fea70e2bfec7e39ed6edddc78aabbc314631ffe5128ae577d0a1` |
| `smplx_names` | [official file](https://raw.githubusercontent.com/vchoutas/smplx/main/smplx/joint_names.py) | `1515994aaf8fb2539f8c1059bfd7c08e623d5d4f6921223ce3d6c2a482bcbb8c` |
| `smplx_models` | [official file](https://raw.githubusercontent.com/vchoutas/smplx/main/smplx/body_models.py) | `1f394b22bd7da6bc7dd2abb2186884d4c233299532aeb0693ab5f9ff40563f8d` |
| `mt5_config` | [official file](https://huggingface.co/google/mt5-base/resolve/main/config.json) | `7ce360bdccf5b665dbb8c32f6912f1d8fd394aa553c133863283a16503d6d550` |
| `mardm_ae` | [official file](https://raw.githubusercontent.com/neu-3dvi/MARDM/main/models/AE.py) | `d592ce1e38388a1bf1297bf1c4ecc2726bd61373743d9be3e16c645470f33eda` |
| `spark_data_card` | [official file](https://huggingface.co/datasets/LionelLow/SignSparK_data/raw/main/README.md) | `d3cf1b962d78f86eaf56b8541151e92e16484c898fd7648a9a02d88d4f45b27c` |

Các chi tiết source facts đã kiểm tra gồm: source full concat order; body slicing11×6; row6D conversion; native-left storage; face jaw6+expression50; SMPLXLayer nhận matrices/differentiable LBS; mT5-base hidden768. **Chưa kiểm tra trên actual LMDB clips:** pose feature FPS, exact split counts, reconstructed facial-basis compatibility và joint output order của asset máy lab. Những điều này thuộc M1–M2, không phải facts đã được giả định đúng.

Tệp này là yêu cầu implementation và nguồn tham chiếu. Không có training, dataset download, human study hoặc SOTA validation được thực hiện khi tạo specification. Synthetic math/schema checks chỉ xác nhận tính nhất quán của tài liệu; actual runnable training tests thuộc coding agent sau khi code/assets sẵn sàng.

## 29. Kết quả kiểm tra tài liệu trước bàn giao

Đã parse năm Python code blocks bằng AST, một YAML config và một JSON schema example; đối chiếu37 reference IDs và9 source-file snapshots. Đã chạy các kiểm tra toán học độc lập bằng NumPy: CFM endpoints/derivative/endpoint estimate; row-rot6D round trip; per-clip thay vì token-weighted reduction; hệ số mean-gradient DDP; DDPM/DDIM timestep indexing và ideal-denoiser endpoint.

Các kiểm tra này **không import hoặc chạy implementation PyTorch của model**, vì model đó là sản phẩm coding agent phải tạo tiếp theo. Chưa có actual LMDB scan, CUDA-memory benchmark, SMPL-X asset-backed gradient test hoặc training run. M0–M9 không được đánh dấu hoàn thành chỉ vì specification đã qua kiểm tra nội dung.

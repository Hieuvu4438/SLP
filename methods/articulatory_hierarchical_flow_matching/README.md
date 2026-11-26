# Articulatory Hierarchical Flow Matching

status: exploring

Gloss-free text-to-pose generation in the normalized 133-D SMPL-X representation. The first flow generates low temporal frequencies and fast directions estimated from signing geometry. The second flow fills the orthogonal complement. Both stages attend to the sentence on the output frame grid. See the derivation and PHOENIX evidence in [the design document](THIET_KE_FLOW_CAU_AM.md).

The [PHOENIX research roadmap](HUONG_CAI_TIEN_SAU_BASELINE_PHOENIX.md) reports the seed-0 results, coarse and residual interventions, back-translation checks, and follow-up experiments.

Use the roadmap recipe for new PHOENIX runs. It uses one process per method, a
broadcast-held train monitor for the pose-to-text teacher, and a sealed checkpoint
selection before scoring held-out dev. The sentence and word–segment losses are
optional and act on the decoded coarse endpoint. The roadmap also describes
planner, geometry, refinement, rotation-chart, and train-only retrieval controls.

```bash
bash configs/experiments/articulatory_hierarchical_flow_matching/phoenix.sh roadmap --help
```

## Run

Use the project's SOKE SMPL-X rig, pose statistics and local mBART assets. The default run is PHOENIX-2014T dev.

```bash
pip install -e '.[articulatory-hierarchical-flow-matching]'
python scripts/run.py --method articulatory_hierarchical_flow_matching \
  --data phoenix_smplx --name articulatory_phoenix --seeds 0,1,2
```

The method also accepts `csl_smplx`, `how2sign_smplx`, and `how2sign_smplx_soke`.
The training checkpoint stores the train-fitted whitening transform, directions, stage scales,
descriptor scales, manifest hash and FK asset hashes. The runner writes
`checkpoints/model.pt`, EMA snapshots at the configured steps and `predictions.npz`.
Set `method.subspace.fk_device=cpu` to fit the descriptor basis on CPU. The byte
encoder is available for synthetic tests via `method.text.backend=bytes`.

## Comparisons

| Override | Effect |
|---|---|
| `method.subspace.mode=temporal` | Temporal-only hierarchy with the nearest integer number of coefficients to the articulatory budget. |
| `method.subspace.mode=parts` | Same low band for every joint, with the extra degrees of freedom reserved for the hands. The hand subspace is expressed in the same whitened space as the default model. |
| `method.subspace.mode=pca` or `random` | Replace the learned fast directions with PCA or a fixed-seed random orthonormal basis. |
| `method.subspace.mode=single` | One flow in the same whitened space with the full sampling budget. |
| `method.subspace.unconstrained=true` | The second flow can revise the whole pose. |
| `method.flow.energy_weight=0.1` | Predict four intervals of fast motion energy from text and feed the prediction into the first flow. This also works with the single-flow control. |
| `method.flow.endpoint_fk_weight=0.02` | Supervise the estimated clean endpoint with wrist and fingertip positions. |
| `method.sampler.lengths_from_npz=PATH` | Lock frame counts to a common archive of text-predicted durations. |

Tune snapshots using generated geometry on separate dev broadcasts. The dev recipe can also
export matching-duration, text-swap and component-oracle predictions for diagnostics.
The reported text-to-pose protocol uses durations predicted from text. Run snapshot selection with
an explicit run directory. [PHOENIX recipe](../../configs/experiments/articulatory_hierarchical_flow_matching/phoenix.sh)

```bash
bash configs/experiments/articulatory_hierarchical_flow_matching/phoenix.sh dev PATH_TO_RUN --diagnostic-clips 24
```

For a matched comparison, pass the same `--lengths-from` archive and
`--descriptor-from` articulatory checkpoint to each dev run. The first supplies common
text-predicted lengths and the second supplies the train scales for common-path geometry.
The script writes the selected checkpoint and its hash, tune/held-out IDs, scores and
predictions into `dev_selection/`.

## Verification

The tests cover complementary projectors, padding, a one-frame clip, exact sampling with
oracle velocity, train-only calibration, the optional energy head and checkpoint reload.
A short PHOENIX run verifies SMPL-X, mBART, CUDA BF16 and the shared evaluator over
the dev split. Smaller CSL-Daily and How2Sign checks exercise the same train and predict path.
Benchmark-quality training follows the separate experiment protocol in the design document.

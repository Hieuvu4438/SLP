# SLP multi-rate flow matching

status: exploring

Proposal 8 implements offline text-to-3D ASL generation over a fixed 44-joint rig.
The cascade predicts duration, samples coarse body rotations and samples full-rate
wrists, fingers and shoulder/elbow corrections. The anchor planner receives frozen
text states and duration. The MVP covers manual signing and head/neck rotations.

The canonical pipeline and its research protocols use `python -m slp.cli`.
`Method.fit` also accepts canonical train and dev_tune datasets for Python callers.

## CPU sanity check

```bash
pip install -e ".[dev,slp-multirate]"
python -m slp.cli smoke --device cpu --out outputs/slp_multirate_p0
python -m pytest -q tests/slp_multirate
```

The smoke command generates an artificial fixture and runs every stage through
20 updates with checkpoint/resume, followed by text-only inference and synthetic
oracle and text-only evaluation. Each stage publishes config, versions, hashes
and metrics. Synthetic evaluation checks interfaces and finite outputs.

The staged real-data workflow and review are documented in
[implementation review](implementation_review.md).

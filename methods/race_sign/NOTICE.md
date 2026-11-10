# Sources and attribution

RACE modules in this folder implement the research design in proposal3. Mathematical CFM and SO(3) operations are implemented here with independent analytic/reference checks; a public paper formulation does not imply a reproduction of that paper's full system.

Runtime baseline dependencies are imported from the author's source rather than copied into RACE:

- SOKE: https://github.com/2000ZRL/SOKE — VQ encoders/decoders, codebooks, mBART multi-head token generation and token mapping.
- MaDiS: https://github.com/2000ZRL/MaDiS — optional released masked-diffusion/Qwen base and author tokenizer. Model checkpoints: https://huggingface.co/2000zrl/MaDiS (author model card lists CC-BY-NC-ND-4.0). Preserve source-model/dataset terms; no weights are redistributed in this folder.
- SMPL-X: https://github.com/vchoutas/smplx — licensed parametric body implementation and separately obtained model assets. No body-model data are bundled.

Design sources:

- DisCoRD, https://arxiv.org/abs/2411.19527 — direct continuous flow decoding from discrete tokens and learned frame-specific conditions.
- SIGNER, https://arxiv.org/abs/2506.07460 — inter-part information exchange and temporal-grounding considerations.
- SALAD, https://arxiv.org/abs/2503.13836 — skeleton/temporal/text separation; its HumanML3D skeleton implementation is not reused as a signing skeleton.
- Flow Matching, https://arxiv.org/abs/2210.02747 — linear conditional probability paths and velocity regression.
- SOKE, https://arxiv.org/abs/2411.17799; MaDiS, https://arxiv.org/abs/2601.19577 — original representations, native generation and pretrained components.

The selector is the proposal's expected-improvement experiment, not an implementation claimed to be novel/validated by these authors. No code or results from other `methods/` implementations are imported. Public-source checkpoint tensors are strictly selected by their module prefixes; historical training dataset objects are discarded by a restricted metadata loader and are not used as experiment evidence.

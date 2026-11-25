"""Text-only generation and explicitly separated development interventions."""

import time

import torch

from .bridge import bridge
from .data import LANGUAGES, Request, digest, pack133, phase_grid, resample
from .relations import GROUPS, dense_relations
from .rotations import identity_like, normalize
from .source import UNIFORM, donor_for, sample_source, solve_ik


def corpus_for(request):
    languages = (request.source_language, request.target_sign_language)
    for corpus, pair in LANGUAGES.items():
        if pair == languages:
            return corpus
    raise ValueError(f"Unsupported language pair: {languages}")


def conditioning(arm, relation, stats, cfg):
    if arm in ("A-paper", "A-matched"):
        return None
    if arm == "A-zero":
        return torch.zeros_like(relation)
    normalized = (relation - stats["mean"].to(relation)) / stats["scale"].to(relation)
    for name in cfg["source"].get("ablate_relation_groups", []):
        normalized[..., GROUPS[name]] = 0
    return normalized


def build_center(arm, plan, rest, cfg, fk, scale, *, donor_bank=None, request=None, oracle=None):
    if arm in UNIFORM:
        return None, {}
    r = plan["relation"]
    if arm == "D":
        return plan["direct"], {}
    if arm == "E-rest":
        return rest.to(r)[None].expand(len(r), -1, -1), {}
    if arm == "E-shuffle":
        donor, center = donor_for(donor_bank, request, plan["length"])
        return center.to(r), {"donor_id": donor}
    if arm not in ("C", "F-oracle-dev", "F-both", "F-oracle-length"):
        raise ValueError(f"Unknown source arm: {arm}")
    target, valid = oracle if oracle is not None else (r, None)
    center, history, status = solve_ik(
        target,
        rest,
        cfg["ik"],
        fk,
        scale,
        cfg["planner"]["loss_weights"],
        valid,
        cfg["source"].get("ablate_relation_groups", []),
    )
    return center, dict(energy=history, **status)


def plan_for(planner, req, cap, head=None, length=None):
    device = next(planner.parameters()).device
    lengths = torch.tensor([length], device=device) if length is not None else None
    out = planner([req], cap, lengths)
    valid = out["valid"][0]
    plan = {"length": int(out["length"][0]), "relation": out["relation"][0, valid]}
    if head is not None:
        plan["direct"] = head(out["hidden"][0, valid])
    return plan


def replan(
    model,
    q0,
    expression,
    conditions,
    relation,
    nfe,
    correction,
    *,
    debug=False,
    sign_enabled=True,
    frame_valid=None,
):
    if nfe < 1:
        raise ValueError("NFE must be positive")
    q = q0
    valid = (
        torch.ones(q.shape[:2], device=q.device, dtype=torch.bool)
        if frame_valid is None
        else frame_valid
    )
    states = []
    for step in range(nfe):
        t = q.new_full((len(q), 1, 1, 1), step / nfe)
        s = (step + 1) / nfe
        end, end_e = model.predict_target(
            q, expression, t, conditions, relation, valid, sign_enabled
        )
        if step + 1 == nfe:
            q, expression = end, end_e
        else:
            q = normalize(bridge(q0, end, torch.full_like(t, s), correction))
            expression = expression + ((s - step / nfe) / (1 - step / nfe)) * (end_e - expression)
        if debug:
            states.append(pack133(q, expression).cpu())
    return q, expression, states


class Sampler:
    def __init__(
        self,
        cfg,
        planner,
        generator,
        stats,
        correction,
        fk,
        arm,
        *,
        head=None,
        donor_bank=None,
        version=None,
    ):
        self.cfg, self.planner, self.model = cfg, planner.eval(), generator.eval()
        self.stats, self.correction, self.fk = stats, correction, fk
        if correction is not None:
            correction.eval()
        self.arm, self.head, self.donors = arm, head, donor_bank
        self.version = version or digest(cfg)

    def generate(self, request: Request, nfe=None, debug=False):
        if self.arm.startswith("F-"):
            raise ValueError("Oracle arms require the explicit dev diagnostic entry point")
        return self._sample(request, nfe, debug)

    def _prepare_request(self, request, dev_clip=None):
        corpus = corpus_for(request)
        cap = self.stats["cap"][corpus]
        device = next(self.model.parameters()).device
        start = _clock(device)
        plan = plan_for(
            self.planner,
            request,
            cap,
            self.head,
            len(dev_clip.q) if self.arm == "F-oracle-length" else None,
        )
        plan_end = _clock(device)
        n, relations = plan["length"], plan["relation"]
        phase = phase_grid(n, device=device, dtype=relations.dtype)
        oracle = None
        if dev_clip is not None:
            from .pipeline import label_at

            sparse = phase_grid(
                n, self.cfg["planner"]["stride_frames"], device=device, dtype=relations.dtype
            )
            oracle = label_at(dev_clip, sparse, self.fk, self.stats["rest"])
            if self.arm == "F-both":
                relations = oracle[0]
        center, diagnostics = build_center(
            self.arm,
            plan,
            self.stats["rest"],
            self.cfg,
            self.fk,
            self.stats["scale"],
            donor_bank=self.donors,
            request=request,
            oracle=oracle,
        )
        if center is not None:
            center = resample(center, phase, quaternion=True)
        r = dense_relations(relations, phase)
        q0, e0 = sample_source(
            self.arm, center, n, self.cfg["source"], request, corpus, device=device, dtype=r.dtype
        )
        return {
            "q0": q0,
            "e0": e0,
            "relation": r,
            "center": center,
            "meta": {
                "fps": self.stats["fps"][corpus],
                "T_hat": n,
                "sample_id": request.sample_id,
                "target_sign_language": request.target_sign_language,
                "seed": request.seed,
                "version": self.version,
                "length_cap": cap,
                "ik_calls": int(self.arm == "C" or self.arm.startswith("F-")),
                "plan_seconds": plan_end - start,
                "source_seconds": _clock(device) - plan_end,
                "source_diagnostics": diagnostics,
            },
        }

    def _predict(self, requests, items, nfe, debug=False):
        from torch.nn.utils.rnn import pad_sequence

        if nfe is None:
            nfe = self.cfg["evaluation"]["nfe_main"]
        q0, e0, r = [
            pad_sequence([item[name] for item in items], batch_first=True)
            for name in ("q0", "e0", "relation")
        ]
        lengths = torch.tensor([item["meta"]["T_hat"] for item in items], device=q0.device)
        valid = torch.arange(q0.shape[1], device=q0.device)[None] < lengths[:, None]
        q0 = torch.where(valid[..., None, None], q0, identity_like(q0))
        condition = conditioning(self.arm, r, self.stats, self.cfg)
        before_text = _clock(q0.device)
        memory = self.model.encode(requests)
        after_text = _clock(q0.device)
        q, e, states = replan(
            self.model,
            q0,
            e0,
            memory,
            condition,
            nfe,
            self.correction,
            debug=debug,
            sign_enabled=self.arm != "A-paper",
            frame_valid=valid,
        )
        after_refiner = _clock(q0.device)
        e = e * self.stats["expr_scale"].to(e) + self.stats["expr_mean"].to(e)
        outputs = []
        for i, item in enumerate(items):
            n = item["meta"]["T_hat"]
            out = {
                **item["meta"],
                "pose133": pack133(q[i, :n], e[i, :n]).cpu(),
                "nfe": nfe,
                "correction_calls": nfe - 1 if self.correction is not None else 0,
                "text_seconds": after_text - before_text,
                "refiner_seconds": after_refiner - after_text,
            }
            if debug:
                trace = torch.stack(states)[:, i, :n]
                trace[..., 123:] = (
                    trace[..., 123:] * self.stats["expr_scale"].cpu()
                    + self.stats["expr_mean"].cpu()
                )
                out.update(
                    relation=item["relation"].cpu(),
                    q0=item["q0"].cpu(),
                    states=trace,
                    q=q[i, :n].cpu(),
                )
                if item["center"] is not None:
                    out["center"] = item["center"].cpu()
            outputs.append(out)
        return outputs

    @torch.no_grad()
    def _sample(self, request, nfe, debug, dev_clip=None):
        device = next(self.model.parameters()).device
        start = _clock(device)
        item = self._prepare_request(request, dev_clip)
        (out,) = self._predict([request], [item], nfe, debug)
        out["latency_seconds"] = _clock(device) - start
        return out

    @torch.no_grad()
    def generate_batch(self, requests, nfe=None):
        """Batch text encoding and diffusion-time steps with per-request RNG keys."""
        if self.arm.startswith("F-") or not requests:
            raise ValueError("Batch generation requires text-only requests")
        items = [self._prepare_request(request) for request in requests]
        return self._predict(requests, items, nfe)

    def oracle_dev(self, request, clip, nfe=None, debug=False):
        if clip.row["split"] != "dev" or not self.arm.startswith("F-"):
            raise ValueError("Oracle generation is a dev-only intervention")
        return self._sample(request, nfe, debug, clip)


def _clock(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()

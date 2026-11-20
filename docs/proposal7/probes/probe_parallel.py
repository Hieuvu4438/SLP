"""P3b: is eps_theta just a slow-down along the geodesic? cos(eps, xi) and the parallel share.

    CUDA_VISIBLE_DEVICES= PYTHONPATH=.:src python docs/proposal7/probes/probe_parallel.py
Writes probe_parallel.json next to this file. Val cache only."""
import glob, json
from pathlib import Path
import numpy as np, torch
from methods.signmflow import geometry as G
from methods.signmflow.paths import CorrectionNet
from methods.signmflow.representation import MotionCodec, load_hand_mean
from slp.data.sign_smplx import load_stats

torch.set_grad_enabled(False)
RUNS = {"phoenix_smplx": "outputs/runs/signmflow_v2_main_phoenix_smplx/signmflow/seed0_20261116-185137-965824",
        "csl_smplx": "outputs/runs/signmflow_v2_main_csl_smplx/signmflow/seed0_20261116-185337-969580"}
res = {}
for ds, run in RUNS.items():
    mean, std = load_stats("third_party/SOKE/deps/stats")
    codec = MotionCodec(mean, std, load_hand_mean("third_party/SOKE/deps/smpl_models"), "quat")
    corr = CorrectionNet()
    corr.load_state_dict(torch.load(run + "/checkpoints/correction_final.pt", map_location="cpu", weights_only=False)["model"])
    arr = np.load(sorted(glob.glob(f"outputs/signmflow/cache/{ds}_val_*.npy"))[0], mmap_mode="r")
    torch.manual_seed(1)
    x1 = codec.encode(torch.from_numpy(np.array(arr[:2000])))[0]
    x0 = G.uniform_hemisphere(x1.shape[:2])
    xi = G.logmap(x0, x1)
    out = {}
    for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
        e = corr(torch.full((2000, 1, 1), t), x0, x1)
        n = xi.norm(dim=-1).clamp_min(1e-9)
        par = (e * xi).sum(-1) / n
        out[str(t)] = {"cos_eps_xi_median": float((par / e.norm(dim=-1).clamp_min(1e-9)).median()),
                       "parallel_energy_share_median": float((par ** 2 / (e * e).sum(-1).clamp_min(1e-12)).median()),
                       "c_t_median (eps_par = -c |xi|)": float((-par / n).median()),
                       "implied_kappa = t(1-(1-t)c)": float(t * (1 - (1 - t) * (-par / n).median()))}
    res[ds] = out
json.dump(res, open(Path(__file__).with_name("probe_parallel.json"), "w"), indent=1)
print(json.dumps(res, indent=1))

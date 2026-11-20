"""Devil's-advocate CPU checks of docs/proposal7 probes. Val + train caches (no test)."""
import glob, json, sys
import numpy as np, torch
from methods.signmflow import geometry as G
from methods.signmflow.naturalness import NaturalnessMeasure
from methods.signmflow.paths import CorrectionNet, path_point
from methods.signmflow.representation import HAND_SLICES, MotionCodec, load_hand_mean
from slp.data.sign_smplx import load_stats

torch.set_grad_enabled(False)
RUNS = {"phoenix_smplx": "outputs/runs/signmflow_v2_main_phoenix_smplx/signmflow/seed0_20261116-185137-965824",
        "csl_smplx": "outputs/runs/signmflow_v2_main_csl_smplx/signmflow/seed0_20261116-185337-969580"}
PARTS = {"body": slice(0, 10), "lhand": HAND_SLICES[0], "rhand": HAND_SLICES[1], "jaw": slice(40, 41)}
ds = sys.argv[1]; split = sys.argv[2] if len(sys.argv) > 2 else "val"
run = RUNS[ds]
mean, std = load_stats("third_party/SOKE/deps/stats")
codec = MotionCodec(mean, std, load_hand_mean("third_party/SOKE/deps/smpl_models"), "quat")
corr = CorrectionNet(); corr.load_state_dict(torch.load(run + "/checkpoints/correction_final.pt", map_location="cpu", weights_only=False)["model"]); corr.eval()
st = torch.load(run + "/checkpoints/prototypes.pt", map_location="cpu", weights_only=False)
meas = NaturalnessMeasure(st["protos"], st["weights"], 0.5, 1e-6)
arr = np.load(sorted(glob.glob(f"outputs/signmflow/cache/{ds}_{split}_*.npy"))[0], mmap_mode="r")
rng = np.random.default_rng(0)
idx = np.sort(rng.choice(len(arr), 2000, replace=False))
torch.manual_seed(3)
x1 = codec.encode(torch.from_numpy(np.asarray(arr[idx])))[0]
N = x1.shape[0]
x0 = G.uniform_hemisphere((N, 41))
xi = G.logmap(x0, x1)
d = xi.norm(dim=-1)  # [N,41]
res = {"ds": ds, "split": split}
ts = [0.1, 0.3, 0.5, 0.7, 0.9]
kap, perp, lag_dev, true_dev, dlogm_t, dlogm_eq = {}, {}, {}, {}, {}, {}
sgrid = torch.linspace(0, 1, 201)
for t in ts:
    tt = torch.full((N, 1, 1), t)
    xn = path_point("naturalness", x0, x1, tt, corr)
    xg = path_point("geodesic", x0, x1, tt)
    w = G.logmap(x0, xn)  # tangent at x0
    k = (w * xi).sum(-1) / (d ** 2).clamp_min(1e-9)  # per-sample per-joint kappa
    pv = (w - k[..., None] * xi).norm(dim=-1)
    kap[t] = k; perp[t] = pv
    lag_dev[t] = G.geodesic_distance(xg, xn)  # what P3 reports (same-t distance)
    # reparametrisation-invariant: distance from xn to the geodesic curve (min over s)
    best = torch.full((N, 41), 1e9); bests = torch.zeros(N, 41)
    for s in sgrid:
        g = G.expmap(x0, float(s) * xi)
        dd = G.geodesic_distance(g, xn)
        m = dd < best; best = torch.where(m, dd, best); bests = torch.where(m, torch.full_like(bests, float(s)), bests)
    true_dev[t] = best
    gs = G.expmap(x0, bests[..., None] * xi)  # closest geodesic point (equal progress)
    dlogm_t[t] = meas.log_mass(xn) - meas.log_mass(xg)
    dlogm_eq[t] = meas.log_mass(xn) - meas.log_mass(gs)
def med(x): return float(torch.as_tensor(x).median())
def pct(x, p): return float(np.percentile(torch.as_tensor(x).numpy(), p))
out = {}
for t in ts:
    o = {"kappa_median": med(kap[t]), "kappa_p10": pct(kap[t], 10), "kappa_p90": pct(kap[t], 90),
         "perp_dev_rad_median": med(perp[t]), "same_t_dist_median(P3 metric)": med(lag_dev[t]),
         "dist_to_geodesic_curve_median": med(true_dev[t]), "dist_to_geodesic_curve_p95": pct(true_dev[t], 95),
         "dlogm_same_t_median": med(dlogm_t[t]), "dlogm_equal_progress_median": med(dlogm_eq[t]),
         "dlogm_equal_progress_p5": pct(dlogm_eq[t], 5), "dlogm_equal_progress_p95": pct(dlogm_eq[t], 95)}
    for name, sl in PARTS.items():
        o[f"kappa_median_{name}"] = med(kap[t][:, sl])
    # dependence on distance |xi|: kappa in distance terciles
    q1, q2 = np.percentile(d.numpy(), [33, 67])
    o["kappa_by_|xi|_tercile"] = [med(kap[t][d < q1]), med(kap[t][(d >= q1) & (d < q2)]), med(kap[t][d >= q2])]
    out[str(t)] = o
res["per_t"] = out
# fit SD3 shift per part on median kappa (proper: on kappa, not on angle ratios)
def fit(kk):
    best = None
    for s in np.linspace(1, 8, 701):
        n = np.array([s * (1 - t) / (1 + (s - 1) * (1 - t)) for t in ts])
        e = float(np.sqrt(np.mean(((1 - n) - kk) ** 2)))
        if best is None or e < best[1]: best = (float(s), e)
    return best
res["shift_fit_all_median_kappa"] = fit(np.array([out[str(t)]["kappa_median"] for t in ts]))
for name in PARTS:
    res[f"shift_fit_{name}"] = fit(np.array([out[str(t)][f"kappa_median_{name}"] for t in ts]))
# per-sample fit dispersion: fit s per (sample,joint) on its kappa curve
K = torch.stack([kap[t] for t in ts], -1).numpy().reshape(-1, len(ts))
sg = np.linspace(1, 10, 181)
curves = np.stack([1 - s * (1 - np.array(ts)) / (1 + (s - 1) * (1 - np.array(ts))) for s in sg])  # [S,5]
err = ((K[:, None, :] - curves[None]) ** 2).mean(-1)
sbest = sg[err.argmin(1)]; ebest = np.sqrt(err.min(1))
res["per_sample_joint_s"] = {"p10": float(np.percentile(sbest, 10)), "p50": float(np.median(sbest)), "p90": float(np.percentile(sbest, 90)),
                             "frac_s_at_grid_edge": float(((sbest == sg[0]) | (sbest == sg[-1])).mean()),
                             "rmse_median": float(np.median(ebest)), "rmse_p90": float(np.percentile(ebest, 90))}
# global s=2.75 residual per sample
c275 = 1 - 2.75 * (1 - np.array(ts)) / (1 + 1.75 * (1 - np.array(ts)))
r = np.sqrt(((K - c275) ** 2).mean(-1))
res["global_s2.75_kappa_rmse_per_sample_joint"] = {"p50": float(np.median(r)), "p90": float(np.percentile(r, 90))}
res["monotone_kappa_frac"] = float((np.diff(K, axis=-1) > 0).all(-1).mean())
print(json.dumps(res, indent=1))
json.dump(res, open(__import__("pathlib").Path(__file__).with_name(f"probe_da_check_{ds}_{split}.json"), "w"), indent=1)

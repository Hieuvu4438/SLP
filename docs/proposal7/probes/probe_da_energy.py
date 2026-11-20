"""How much of the learned path's energy reduction can a pure time-reparametrisation of the geodesic explain?"""
import glob, json, sys
import numpy as np, torch
from methods.signmflow import geometry as G
from methods.signmflow.naturalness import NaturalnessMeasure
from methods.signmflow.paths import CorrectionNet, path_and_velocity
from methods.signmflow.representation import MotionCodec, load_hand_mean
from slp.data.sign_smplx import load_stats

RUNS = {"phoenix_smplx": "outputs/runs/signmflow_v2_main_phoenix_smplx/signmflow/seed0_20261116-185137-965824",
        "csl_smplx": "outputs/runs/signmflow_v2_main_csl_smplx/signmflow/seed0_20261116-185337-969580"}
ds = sys.argv[1]; run = RUNS[ds]
mean, std = load_stats("third_party/SOKE/deps/stats")
codec = MotionCodec(mean, std, load_hand_mean("third_party/SOKE/deps/smpl_models"), "quat")
corr = CorrectionNet(); corr.load_state_dict(torch.load(run + "/checkpoints/correction_final.pt", map_location="cpu", weights_only=False)["model"]); corr.eval()
st = torch.load(run + "/checkpoints/prototypes.pt", map_location="cpu", weights_only=False)
meas = NaturalnessMeasure(st["protos"], st["weights"], 0.5, 1e-6)
arr = np.load(sorted(glob.glob(f"outputs/signmflow/cache/{ds}_val_*.npy"))[0], mmap_mode="r")
idx = np.sort(np.random.default_rng(1).choice(len(arr), 500, replace=False))
torch.manual_seed(5)
with torch.no_grad():
    x1 = codec.encode(torch.from_numpy(np.asarray(arr[idx])))[0]
x0 = G.uniform_hemisphere(x1.shape[:2])
xi = G.logmap(x0, x1); L2 = (xi * xi).sum(-1)  # squared geodesic speed per joint
ng = 200
s_grid = (torch.arange(ng) + 0.5) / ng
# a_j(s) = cost density along the geodesic at progress s for unit dkappa/dt: quad_form(x(s), xi-velocity)
A = []
with torch.no_grad():
    for s in s_grid:
        xs = G.expmap(x0, float(s) * xi)
        # velocity of constant-speed geodesic at s: parallel transport of xi = d/ds Exp(s xi)
        n = xi.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        u = xi / n
        vs = n * (-torch.sin(float(s) * n) * x0 + torch.cos(float(s) * n) * u)
        A.append(meas.quad_form(xs, vs))
A = torch.stack(A, -1)  # [N,J,ng]
geo_energy = A.mean(-1).sum(-1)  # kappa = t
lb_joint = (A.sqrt().mean(-1) ** 2).sum(-1)  # best per-joint reparam
lb_frame = (A.sum(1).sqrt().mean(-1) ** 2)  # best shared reparam (one kappa per frame)
def shift_energy(sh):
    # E_t[a(kappa(t)) kappa'(t)^2], kappa = 1 - n, n = s(1-t)/(1+(s-1)(1-t))
    t = s_grid
    m = 1 - t
    kap = 1 - sh * m / (1 + (sh - 1) * m)
    dk = sh / (1 + (sh - 1) * m) ** 2
    # interpolate A at kap
    pos = (kap * ng - 0.5).clamp(0, ng - 1)
    lo = pos.floor().long(); hi = (lo + 1).clamp(max=ng - 1); w = pos - lo
    a = A[..., lo] * (1 - w) + A[..., hi] * w
    return (a * dk ** 2).mean(-1).sum(-1)
# learned path energy
t = torch.rand(x1.shape[0], 1, 1)
xs_, v_ = path_and_velocity("naturalness", x0, x1, t, corr)
with torch.no_grad():
    learned = meas.quad_form(xs_.detach(), v_.detach()).sum(-1)
res = {"ds": ds, "geodesic_energy_mean": float(geo_energy.mean()), "learned_energy_mean(MC t)": float(learned.mean()),
       "best_shared_reparam_lower_bound_mean": float(lb_frame.mean()),
       "best_perjoint_reparam_lower_bound_mean": float(lb_joint.mean())}
for sh in [1.5, 2.0, 2.75, 3.5, 5.0]:
    res[f"sd3_shift_{sh}_energy_mean"] = float(shift_energy(sh).mean())
print(json.dumps(res, indent=1))
json.dump(res, open(__import__("pathlib").Path(__file__).with_name(f"probe_da_energy_{ds}.json"), "w"), indent=1)

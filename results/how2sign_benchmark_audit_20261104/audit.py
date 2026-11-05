"""Read-only source/data audit; CPU checks only; never changes predictions."""
import ast
import csv
import hashlib
import importlib.util
import json
import math
import os
import pickle
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Sequence, Tuple
from urllib.request import urlopen

import numpy as np
import torch

ROOT = Path('/home/haipd/SLP')
RUN = ROOT / 'outputs/how2sign_test_blends_20261104'
OUT = ROOT / 'results/how2sign_benchmark_audit_20261104'
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)

def definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace

def constant(path, name):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)

def archive_keys(path):
    with zipfile.ZipFile(path) as archive:
        return {n[:-4] for n in archive.namelist() if n.endswith('.npy')}

bad = constant(ROOT / 'third_party/MaDiS/mGPT/data/sign_dataset.py', 'BAD_HOW2SIGN_IDS')
bad_soke = set(constant(ROOT / 'third_party/SOKE/mGPT/data/humanml/dataset_t2m.py', 'bad_how2sign_ids'))
result = {'verification_status': 'ANALYZED: source and selected CPU numerical checks; no full benchmark rerun',
          'commits': {'MaDiS': '071828ea1b94530dc7ff0a6c009c0a7c6d1f0784', 'SOKE': '5cbc55d84b5a7cbf05a9cf020c468052e8d94d00'},
          'split_counts': {}, 'bad_ids_equal': bad == bad_soke}
all_rows = {}
for split in ['train', 'val', 'test']:
    rows = list(csv.DictReader(open(ROOT / f'data/processed/madis/How2Sign/{split}/re_aligned/how2sign_realigned_{split}_preprocessed_fps.csv')))
    all_rows[split] = {r['SENTENCE_NAME']: r for r in rows}
    names = set(all_rows[split])
    poses = {p.stem for p in (ROOT / f'data/processed/madis/How2Sign/poses_video_level/{split}/poses').glob('*.pkl')}
    official = {r['SENTENCE_NAME'] for r in rows if float(r['END_REALIGNED']) - float(r['START_REALIGNED']) < 30 and r['SENTENCE_NAME'] not in bad}
    result['split_counts'][split] = {'csv': len(rows), 'poses': len(poses), 'official_annotation_filter': len(official), 'our_retrieval_loader': len(names & poses), 'our_extra_vs_official_filter': len((names & poses) - official)}
    if split == 'test':
        official_test = official
        result['test_exclusions'] = {'duration_at_least_30s': sorted(names - official - bad), 'bad_ids': sorted(names & bad)}

pred = {p.stem for p in (RUN / 'madis').glob('*.pkl')}
pair = archive_keys(RUN / 'pairs/r0.npz')
kp = archive_keys(RUN / 'kp_pairs/s0.npz')
a50_keys = archive_keys(RUN / 'a50.npz')
result['test_manifest'] = {'official_filter': len(official_test), 'madis_predictions': len(pred), 'retrieval_candidates': len(pair), 'keypoints': len(kp), 'a50': len(a50_keys), 'madis_exactly_matches_official_filter': pred == official_test, 'missing_after_projection': sorted(pred - kp), 'loss_fraction': (len(pred)-len(kp))/len(pred), 'extra_prediction_ids': sorted(pred-official_test)}
result['train_test_name_overlap'] = sorted(set(all_rows['train']) & set(all_rows['test']))
result['annotation_hashes'] = {}
for rel in ['data/raw/soke/How2Sign/how2sign_realigned_test_preprocessed_fps.csv', 'data/processed/madis/How2Sign/test/re_aligned/how2sign_realigned_test_preprocessed_fps.csv']:
    result['annotation_hashes'][rel] = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()

stats = {}
for key in ['mean', 'std']:
    pa = ROOT / f'third_party/MaDiS/hf_assets/{key}.pt'
    pb = ROOT / f'third_party/SOKE/deps/stats/{key}.pt'
    a = torch.load(pa, weights_only=True, map_location='cpu')
    b = torch.load(pb, weights_only=True, map_location='cpu')
    stats[key] = {'equal_tensors': bool(torch.equal(a, b)), 'max_abs': float((a-b).abs().max()), 'sha256_equal': hashlib.sha256(pa.read_bytes()).hexdigest() == hashlib.sha256(pb.read_bytes()).hexdigest()}
    a = a[36:]
    if key == 'mean':
        mean = torch.cat([a[:-20], a[-10:]]).float()
    else:
        std = torch.cat([a[:-20], a[-10:]]).float()
result['normalization_stats'] = stats

md = dict(torch=torch, Optional=Optional, Sequence=Sequence, Tuple=Tuple)
definitions(ROOT / 'third_party/MaDiS/mGPT/metrics/dtw.py', {'dtw_from_cost_matrix', 'pairwise_jpe', 'batched_dtw_jpe'}, md)
definitions(ROOT / 'scripts/madis_oracle_eval.py', {'dtw_with_len'}, md)
sk = dict(np=np, array=np.array, zeros=np.zeros, full=np.full, argmin=np.argmin, inf=np.inf, ndim=np.ndim, isinf=math.isinf)
definitions(ROOT / 'third_party/SOKE/mGPT/metrics/dtw.py', {'dtw', '_traceback', 'l2_dist_align'}, sk)
rng = np.random.default_rng(426)
cases = []
for T, U, N in [(1, 1, 21), (7, 11, 21), (23, 17, 144), (40, 56, 21), (56, 40, 144)]:
    x = rng.normal(size=(T, N, 3)).astype(np.float32)
    y = rng.normal(size=(U, N, 3)).astype(np.float32)
    wanted = None if N == 21 else [12,16,17,18,19,20,21,59,58,57,56,55]
    c = md['pairwise_jpe'](torch.from_numpy(x), torch.from_numpy(y), wanted, 0)
    official_value = float(md['dtw_from_cost_matrix'](c[None])[0])
    own_value = md['dtw_with_len'](c)[0]
    soke_value = float(sk['dtw'](x, y, partial(sk['l2_dist_align'], wanted=wanted, align_idx=0))[0])
    cases.append({'T': T, 'U': U, 'joints': N, 'MaDiS': official_value, 'ours': own_value, 'SOKE': soke_value})
result['synthetic_metric_check'] = {'cases': cases, 'max_abs_ours_vs_madis': max(abs(c['ours']-c['MaDiS']) for c in cases), 'max_abs_soke_vs_madis': max(abs(c['SOKE']-c['MaDiS']) for c in cases)}

os.chdir(ROOT / 'third_party/MaDiS')
sys.path.insert(0, str(ROOT / 'third_party/MaDiS'))
from mGPT.utils.human_models import get_coord, smpl_x
from mGPT.metrics.generation import GenerationMetrics
from mGPT.data.sign_loader import load_h2s_sample

own = dict(np=np, torch=torch, get_coord=get_coord, smpl_x=smpl_x, pairwise_jpe=md['pairwise_jpe'])
own['KEYS'] = constant(ROOT / 'scripts/madis_oracle_eval.py', 'KEYS')
own['SHAPE'] = constant(ROOT / 'scripts/madis_oracle_eval.py', 'SHAPE')
definitions(ROOT / 'scripts/madis_oracle_eval.py', {'Joints', 'scores', 'dtw_with_len', 'feats133'}, own)
definitions(ROOT / 'scripts/decode_sign_tokens.py', {'madis_crop'}, own)
tree = ast.parse((ROOT / 'third_party/MaDiS/mGPT/data/H2S.py').read_text())
method = next(n for cls in tree.body if isinstance(cls, ast.ClassDef) for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'feats2joints')
official_ns = dict(torch=torch, get_coord=get_coord)
exec(compile(ast.Module(body=[method], type_ignores=[]), 'official_H2S.feats2joints', 'exec'), official_ns)
fake_dm = SimpleNamespace(hparams=SimpleNamespace(mean=mean, std=std))
metric = GenerationMetrics(None)
J = own['Joints'](mean, std, 'cpu')
z50 = np.load(RUN / 'a50.npz')
bare50 = {k.removeprefix('val__'): k for k in z50.files}
eligible = sorted(kp & set(bare50))
selected = [eligible[0], eligible[len(eligible)//2], eligible[-1]]
real_checks = []
with torch.no_grad():
    for name in selected:
        row = all_rows['test'][name]
        p = ROOT / f'data/processed/madis/How2Sign/poses_video_level/test/poses/{name}.pkl'
        packed = pickle.load(open(p, 'rb'))
        raw = own['feats133'](packed)
        fps = float(row['fps'])
        ma_indices = np.arange(len(raw)) if fps <= 24 else np.linspace(0,len(raw)-1,int(24*len(raw)/fps),dtype=int)
        soke_indices = np.arange(len(raw)) if fps <= 24 else np.floor(np.arange(int(24*len(raw)/fps))*len(raw)/int(24*len(raw)/fps)).astype(int)
        ref = own['madis_crop']((raw[ma_indices]-mean.numpy())/std.numpy()).astype(np.float32)
        official_raw, _, _, _ = load_h2s_sample({'name': name, 'fps': fps, 'text': row['SENTENCE']}, p.parent, need_code=False)
        official_ref = own['madis_crop']((official_raw-mean.numpy())/(std.numpy()+1e-10)).astype(np.float32)
        pred_arr = z50[bare50[name]]
        ref_tensor, pred_tensor = torch.from_numpy(ref), torch.from_numpy(pred_arr)
        mesh_r, joints_r = official_ns['feats2joints'](fake_dm, ref_tensor[None])
        mesh_p, joints_p = official_ns['feats2joints'](fake_dm, pred_tensor[None])
        own_pred = J(pred_tensor)
        own_ref = J(ref_tensor)
        own_scores = own['scores'](own_pred, own_ref)
        nt = len(pred_arr)//4
        metric.update(feats_rst=None, feats_ref=None, joints_rst=joints_p, joints_ref=joints_r,
                      vertices_rst=mesh_p, vertices_ref=mesh_r,
                      body_tokens=[torch.zeros(nt,dtype=torch.long)], lhand_tokens=[torch.zeros(nt,dtype=torch.long)], rhand_tokens=[torch.zeros(nt,dtype=torch.long)],
                      tokens_ref=torch.zeros(1,len(ref)//4,3,dtype=torch.long), lengths=[len(ref)], lengths_rst=[len(pred_arr)], time_cost=0, split='test', src=['how2sign'], name=[name])
        official_scores = metric.name2scores[name]
        real_checks.append({'name': name, 'reference_frames': len(ref), 'generated_frames': len(pred_arr),
                            'reference_preprocessing_max_abs': float(np.abs(ref-official_ref).max()),
                            'joint_conversion_max_abs': float((own_pred[0]-joints_p).abs().max()),
                            'ours_body': own_scores['body'], 'official_body': official_scores['how2sign_DTW_JPE_body'],
                            'ours_hands': own_scores['hands'], 'official_hands': official_scores['how2sign_DTW_JPE_hands'],
                            'soke_vs_madis_sampling_different_indices': int((ma_indices!=soke_indices).sum())})
result['real_a50_official_metric_checks'] = real_checks

with zipfile.ZipFile(ROOT / 'data/raw/soke/_downloads/h2s_fit/test_poses.zip') as archive:
    member_set = set(archive.namelist())
    raw_checks = []
    for name in selected:
        packed = pickle.load(open(ROOT / f'data/processed/madis/How2Sign/poses_video_level/test/poses/{name}.pkl', 'rb'))
        for index in [0, len(packed['smplx_root_pose'])//2, len(packed['smplx_root_pose'])-1]:
            member = f'poses/{name}/{name}_{index}_3D.pkl'
            frame = pickle.loads(archive.read(member))
            raw_checks.append({'name':name,'frame':index,'all_7_parameters_exact': all(np.array_equal(packed[key][index],frame[key]) for key in own['KEYS'])})
    result['raw_pose_repack_spot_checks'] = raw_checks

def remote_check(item):
    repo,commit,rel = item
    local = ROOT / f'third_party/{repo}' / rel
    url = f'https://raw.githubusercontent.com/2000ZRL/{repo}/{commit}/{rel}'
    try:
        with urlopen(url, timeout=25) as response:
            content=response.read()
        return {'repo':repo,'path':rel,'url':url,'identical':content==local.read_bytes(),'sha256':hashlib.sha256(content).hexdigest()}
    except Exception as exc:
        return {'repo':repo,'path':rel,'url':url,'error':str(exc)}
items=[('MaDiS',result['commits']['MaDiS'],'mGPT/metrics/generation.py'),('MaDiS',result['commits']['MaDiS'],'mGPT/data/sign_loader.py'),('SOKE',result['commits']['SOKE'],'mGPT/metrics/dtw.py'),('SOKE',result['commits']['SOKE'],'mGPT/data/humanml/load_data.py')]
with ThreadPoolExecutor(max_workers=4) as executor:
    result['official_remote_source_checks']=list(executor.map(remote_check,items))

result['original_dtw_result']=json.load(open(RUN/'dtw.json'))
(OUT/'evidence.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'evidence_file':str(OUT/'evidence.json'),'manifest':result['test_manifest'],'real_checks':real_checks,'remote_source_checks':result['official_remote_source_checks'],'stats':stats},indent=2),flush=True)

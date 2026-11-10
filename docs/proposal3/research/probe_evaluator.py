"""Run isolated upstream distance functions on synthetic joints; no SMPL assets."""
from pathlib import Path
import ast,json,hashlib,numpy as np
from functools import partial
p=Path('third_party/proposal3_sources/code/SOKE/mGPT/metrics/dtw.py')
tree=ast.parse(p.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['dtw','_traceback','l2_dist_align','l2_dist']]
ns={k:getattr(np,k) for k in ['array','zeros','full','argmin','inf','ndim']};ns.update(np=np,isinf=np.isinf)
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
x=np.array([[0.,0.,0.],[1.,0.,0.],[0.,2.,0.]])
y=x+np.array([4.,3.,2.]);rot=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
cost=partial(ns['l2_dist_align'],align_idx=0)
a=np.stack([x,x]);b=np.stack([x*1.1,x*1.1])
result={'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'synthetic_unit':'arbitrary joint-coordinate unit','root_translation_cost':cost(x,y),'root_aligned_rotation_cost':cost(x,x@rot.T),'single_step_cost':ns['dtw'](a[:1],b[:1],cost)[0],'two_step_cost':ns['dtw'](a,b,cost)[0],'hand_translation_relative_body_is_invisible_to_isolated_hand_cost':bool(np.isclose(cost(x,y),0)),'scope':'Isolated upstream functions only; no claim to reproduce paper tables or full evaluator.'}
assert np.isclose(result['root_translation_cost'],0)
assert result['root_aligned_rotation_cost']>0
assert np.isclose(result['two_step_cost'],2*result['single_step_cost'])
Path('docs/proposal3/research/evaluator_probe.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

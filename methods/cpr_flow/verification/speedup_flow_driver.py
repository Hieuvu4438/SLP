import copy,json,time
from pathlib import Path
import torch,yaml,numpy as np
from cprflow.training.tasks import TaskObjective,make_model
from cprflow.training.artifacts import normalizer,scales,CachedFeatures
from cprflow.training.engine import make_optimizer,seed_everything
from cprflow.training.checkpoint import load_checkpoint
from cprflow.training.ema import EMA
from cprflow.data.dataset import MotionDataset
from cprflow.data.sampler import BalancedBatchSampler
from cprflow.data.collate import collate_motion
from cprflow.data.geometry_cache import GeometryCache
from cprflow.geometry.body_model import BodyModel
from cprflow.geometry.relations import relation33
from cprflow.geometry.temporal import pool2
from cprflow.data.stats import Normalizer
from cprflow.models.ae import MotionAE

torch.set_num_threads(2);seed_everything(17,True)
root=Path('/home/haipd/SLP/methods/cpr_flow')
cfg=yaml.safe_load((root/'runs/jobs/joint3_fast/configs/s17.train.c0.yaml').read_text())
ds=MotionDataset(Path(cfg['paths']['manifests'])/'train.jsonl','train',cfg['data']['datasets'],1024)
sampler=BalancedBatchSampler(ds.rows,64,17,cfg['data']['datasets'],cfg['data']['q_dataset'])
mn=normalizer(cfg,'motion','cuda');rn=normalizer(cfg,'relation','cuda')
ae=MotionAE(mn,execution='masked_batch').cuda()
ae.load_state_dict(load_checkpoint(root/'runs/joint3/ae_joint3_s17/checkpoints/last.pt')['ema']['shadow']);ae.freeze()
body=BodyModel(cfg['paths']['smplx_model'],512).cuda();gc=GeometryCache(cfg);tc=CachedFeatures(cfg,'text')
examples={}
for step in range(10000):
 ids=sampler.plan(step);r=ds.rows[ids[0]];bucket=next(b for b in (128,256,512,1024) if r['frames']<=b)
 examples.setdefault((r['dataset_id'],bucket),(step,ids))
 if len(examples)==sum(len(bins) for bins in sampler.buckets.values()):break
report={'synthetic':False,'scope':'Throughput proxy on real clips and cached mT5 text; AE step346 EMA generates temporary latents in RAM, identity latent normalization, freshly initialized full-size DiT. Not a measurement of a trained final CFM or semantic branch. Complete logical batch64, microbatch4, optimizer and EMA included; preparation of provisional latents excluded. Shared GPU.','batches':[]}
for (dataset,bucket),(step,ids) in sorted(examples.items()):
 rows=[ds[i] for i in ids];prepared={};geometry={}
 for row in rows:
  geometry.setdefault(row['uid'],gc.get(row));prepared[('text',row['uid'])]=tc.get(row)
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  for begin in range(0,64,8):
   group=rows[begin:begin+8];batch=collate_motion(group,mn,'cuda');z=ae.encode(batch.x,batch.lengths)
   for i,row in enumerate(group):
    n=(row['frames']+1)//2
    prepared[('latent',row['uid'])]={'z':z[i,:,:n].float().cpu().numpy()}
  for row in rows:
   geom={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in geometry[row['uid']].items()}
   relation=relation33(geom['joints'],geom['names'])
   rel=pool2(relation[None],torch.tensor([row['frames']],device='cuda'))[0]
   prepared[('relations',row['uid'])]={'r':rel.float().cpu().numpy()}
 item={'dataset':dataset,'bucket':bucket,'mean_frames':sum(r['frames'] for r in rows)/64,'measurements':{}}
 for variant,rel,auxstep in (('p4',False,40001),('p5_without_aux',True,40001),('c0_aux_update',True,40000)):
  current=copy.deepcopy(cfg);current['relation']['enabled']=rel
  if not rel:current['loss'].update(geom=0.,cycle=0.,lag=0.);current['run']['variant']='plain_cfm'
  obj=TaskObjective.__new__(TaskObjective);obj.config=current;obj.device=torch.device('cuda');obj.task='flow';obj.motion_norm=mn;obj.relation_norm=rn
  obj.latent_norm=Normalizer(torch.zeros(4,1,128),torch.ones(4,1,128)).cuda()
  obj.ae=ae;obj.body=body;obj.geometry_cache=gc;obj.scales=scales(cfg);obj.teacher=None
  obj._prepared_geometry=geometry;obj._prepared_features=prepared;obj.validation=False
  model=make_model(current,mn).cuda();optimizer=make_optimizer(model,current['train']['lr'],.01);ema=EMA(model)
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();optimizer.zero_grad(set_to_none=True)
  for begin in range(0,64,4):
   with torch.autocast('cuda',dtype=torch.bfloat16):
    vector,_=obj(model,rows[begin:begin+4],auxstep,rows);loss=vector.float().sum()/64
   loss.backward()
  norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm)
  optimizer.step();ema.update(model);torch.cuda.synchronize()
  item['measurements'][variant]={'seconds':time.perf_counter()-start,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30}
  print(json.dumps({'dataset':dataset,'bucket':bucket,'variant':variant,**item['measurements'][variant]}),flush=True)
  del model,optimizer,ema,obj,loss,vector;torch.cuda.empty_cache()
 report['batches'].append(item);(root/'verification/speedup_flow_proxy.json').write_text(json.dumps(report,indent=2))
print('DONE',flush=True)
teacher_cfg=yaml.safe_load((root/'runs/jobs/joint3_fast/configs/train.teacher.yaml').read_text())
teacher_obj=TaskObjective(teacher_cfg,'cuda');teacher=make_model(teacher_cfg,teacher_obj.motion_norm).cuda()
optimizer=make_optimizer(teacher,teacher_cfg['train']['lr'],.01);ema=EMA(teacher)
report['teacher_proxy']=[]
for (dataset,bucket),(step,ids) in sorted(examples.items()):
 rows=[ds[i] for i in ids]
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
 teacher_obj.preflight(rows,rows);optimizer.zero_grad(set_to_none=True)
 with torch.autocast('cuda',dtype=torch.bfloat16):
  vector,_=teacher_obj(teacher,rows,step,rows);loss=vector.mean()
 loss.backward();norm=torch.nn.utils.clip_grad_norm_(teacher.parameters(),1.);assert torch.isfinite(norm)
 optimizer.step();ema.update(teacher);torch.cuda.synchronize()
 result={'dataset':dataset,'bucket':bucket,'seconds':time.perf_counter()-start,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
 report['teacher_proxy'].append(result);print(json.dumps({'teacher':result}),flush=True)
 (root/'verification/speedup_flow_proxy.json').write_text(json.dumps(report,indent=2))
print('TEACHER DONE',flush=True)

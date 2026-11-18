import copy,json,time,os
from pathlib import Path
import torch,yaml
from cprflow.training.tasks import TaskObjective,make_model
from cprflow.training.engine import make_optimizer,seed_everything
from cprflow.training.checkpoint import load_checkpoint
from cprflow.training.ema import EMA
from cprflow.data.dataset import MotionDataset
from cprflow.data.sampler import BalancedBatchSampler

torch.set_num_threads(2)
seed_everything(17,True)
root=Path('/home/haipd/SLP/methods/cpr_flow')
cfg=yaml.safe_load((root/'runs/joint3/ae_joint3_s17/resolved_config.yaml').read_text())
state=load_checkpoint(root/'runs/joint3/ae_joint3_s17/checkpoints/last.pt')
ds=MotionDataset(Path(cfg['paths']['manifests'])/'train.jsonl','train',cfg['data']['datasets'],1024)
sampler=BalancedBatchSampler(ds.rows,128,17,cfg['data']['datasets'],cfg['data']['q_dataset'])
examples={}
for step in range(10000):
 ids=sampler.plan(step)
 row=ds.rows[ids[0]]
 bucket=next(b for b in (128,256,512,1024) if row['frames']<=b)
 key=(row['dataset_id'],bucket)
 examples.setdefault(key,(step,ids))
 if len(examples)==sum(len(bins) for bins in sampler.buckets.values()):break
report={'synthetic':False,'checkpoint_step':state['successful_optimizer_step'],'logical_batch':128,'device':torch.cuda.get_device_name(),'shared_gpu':True,'scope':'Real complete logical AE update, including data loading, preflight, all losses, backward, clipping, AdamW and EMA. Baseline uses current code with per-clip execution, microbatch 1; optimized microbatch 8. No live training state is modified.','batches':[]}
for (dataset,bucket),(step,ids) in sorted(examples.items()):
 item={'dataset':dataset,'bucket':bucket,'sampler_step':step,'mean_frames':sum(ds.rows[i]['frames'] for i in ids)/128,'measurements':{}}
 for micro,execution,chunk in ((1,'per_clip',128),(8,'masked_batch',512)):
  current=copy.deepcopy(cfg);current['ae']['execution']=execution;current['geometry']['fk_chunk_frames']=chunk
  obj=TaskObjective(current,'cuda')
  model=make_model(current,obj.motion_norm).cuda();model.load_state_dict(state['model']);model.train()
  optimizer=make_optimizer(model,cfg['train']['lr'],cfg['train']['weight_decay']);optimizer.load_state_dict(state['optimizer'])
  ema=EMA(model,cfg['train']['ema_decay'])
  torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();begin=time.perf_counter()
  rows=[ds[i] for i in ids];obj.preflight(rows,rows)
  data_s=time.perf_counter()-begin
  optimizer.zero_grad(set_to_none=True)
  losses=[]
  for start in range(0,128,micro):
   with torch.autocast('cuda',dtype=torch.bfloat16):
    vector,_=obj(model,rows[start:start+micro],step,rows)
    loss=vector.float().sum()/128
   losses.append(vector.detach().float().sum())
   loss.backward()
  norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
  assert torch.isfinite(norm)
  optimizer.step();ema.update(model)
  torch.cuda.synchronize()
  item['measurements'][execution]={'seconds':time.perf_counter()-begin,'data_seconds':data_s,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'loss':float(torch.stack(losses).sum()/128),'gradnorm':float(norm)}
  print(json.dumps({'dataset':dataset,'bucket':bucket,'execution':execution,**item['measurements'][execution]}),flush=True)
  del model,optimizer,ema,obj,loss,vector,losses;torch.cuda.empty_cache()
 item['speedup']=item['measurements']['per_clip']['seconds']/item['measurements']['masked_batch']['seconds']
 report['batches'].append(item)
 (root/'verification/speedup_real_measurements.json').write_text(json.dumps(report,indent=2))
print('DONE',flush=True)

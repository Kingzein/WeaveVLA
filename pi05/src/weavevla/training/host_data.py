"""Causal input batching and prefetch."""
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from .planner import Planner

N,E,P,L=32,256,512,16
class Host:
 def __init__(self, streams, capacity):
  self.streams = streams
  self.capacity = capacity
  self.contexts = [dict(key=None, planner=None) for _ in range(capacity * len(streams))]
  self.workers = ThreadPoolExecutor(max_workers=len(streams), thread_name_prefix='cache_rank')
 def close(self):self.workers.shutdown(wait=True)
 def prepare(self):
  begin=time.monotonic();futures=[self.workers.submit(s.batch,list(range(s.n))) for s in self.streams]
  data=[f.result() for f in futures];rows=[];packs=[];reset=[];finished=[];trace=[]
  for rank, batch in enumerate(data):
   obs, scenes, actions, *masks, done = batch
   valid_steps = masks[0] if masks else None
   for i,scene in enumerate(scenes):
    k=rank*self.capacity+i;ctx=self.contexts[k];key=scene.context['episode_key'];new=ctx['key']!=key
    pl=Planner(N,E,P)
    if not new:pl.restore(ctx['planner'])
    x,active,pairs=pl.pack(scene.objects,scene.context);ids={str(o['id']):j for j,o in enumerate(scene.objects)}
    label=np.zeros((N,L),np.int32);lm=np.zeros((N,L),bool);pixels=np.zeros((N,224,224),bool);allowed=np.zeros(N,bool)
    for j,oid in enumerate(active):
     if oid not in ids:continue
     oi=ids[oid];assert scene.objects[oi]['image_index']==0,'This cache uses main-view graph masks';lid=scene.label_ids[oi].numpy();mask=scene.label_mask[oi].numpy();count=int(mask.sum());assert count<=L,(oid,count)
     label[j,:count]=lid[mask];lm[j,:count]=True;pixels[j]=np.asarray(scene.objects[oi]['visual_mask']);allowed[j]=scene.objects[oi]['visible_now'] and pixels[j].any()
     if not x['keep_memory'][j]:
      points=torch.tensor(scene.objects[oi]['points']);x['points'][j]=0;x['point_mask'][j]=False
      if len(points):
       ix=torch.linspace(0,len(points)-1,min(len(points),P)).long();pc=(points[ix]-points.mean(0)).numpy();x['points'][j,:len(pc)]=pc;x['point_mask'][j,:len(pc)]=True
    x.update(label_ids=label,label_mask=lm,pixel_masks=pixels,visual_allowed=allowed)
    end=key in done
    self.contexts[k]=dict(key=None if end else key,planner=Planner(N,E,P).state() if end else pl.state())
    rows.append(dict(image=obs.images['base_0_rgb'][i].permute(1,2,0).numpy(),wrist=obs.images['left_wrist_0_rgb'][i].permute(1,2,0).numpy(),state=obs.state[i].numpy(),tokens=obs.tokenized_prompt[i].numpy().astype(np.int32),token_mask=obs.tokenized_prompt_mask[i].numpy(),actions=actions[i].numpy(),valid=True))
    if valid_steps is not None:rows[-1]['valid_steps']=valid_steps[i,:,0].numpy()
    packs.append(x);reset.append(new);finished.append(end);trace.append((key,int(scene.context['step'])))
   for _ in range(self.capacity-len(scenes)):
    row={key:np.zeros_like(value) for key,value in rows[-1].items()};row['tokens'][0]=2;row['token_mask'][0]=True;row['valid']=False
    rows.append(row);packs.append({key:np.zeros_like(value) for key,value in packs[-1].items()});reset.append(True);finished.append(True);trace.append(('padding',-1))
  assert len(rows) == len(self.contexts)
  arrays={k:np.stack([r[k] for r in rows]) for k in rows[0]};arrays['reset']=np.array(reset);arrays['finished']=np.array(finished)
  pack={k:np.stack([p[k] for p in packs]) for k in packs[0]}
  return dict(arrays=arrays,pack=pack,prepare_seconds=time.monotonic()-begin)

class Prefetch:
 def __init__(self,host):self.host=host;self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='causal_prefetch');self.pending=self.pool.submit(host.prepare)
 def take(self):
  batch=self.pending.result();self.pending=self.pool.submit(self.host.prepare);return batch
 def close(self):
  self.pool.shutdown(wait=True);self.host.close()

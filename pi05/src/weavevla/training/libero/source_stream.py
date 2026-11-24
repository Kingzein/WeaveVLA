import json,random
from pathlib import Path
from functools import lru_cache
from types import SimpleNamespace
import numpy as np
import torch
import sentencepiece as spm
from weavevla_pi05.adapter import GraphScene,image_digest,tokenize_object_labels
from weavevla.geometry.cache import read_frame
from weavevla.policy.masks import attach_masks
@lru_cache(256)
def targets(path):
 with np.load(path) as z:return z['actions'],z['states']
class Streams:
 def __init__(self, rank, manifest_path, tokenizer):
  self.root = manifest_path.parent
  self.m=json.loads(manifest_path.read_text());self.records=self.m['episodes'];self.rank=rank;self.n=43 if rank<4 else 42;offset=sum(43 if r<4 else 42 for r in range(rank));self.rng=random.Random(42+rank)
  self.pool=[list(range(offset+i,len(self.records),256)) for i in range(self.n)]
  for p in self.pool:self.rng.shuffle(p)
  self.pos=[0]*self.n;self.t=[self.rng.randrange(min(8,len(self.records[p[0]]['frames']))) for p in self.pool]
  self.tok=spm.SentencePieceProcessor(model_file=str(tokenizer))
 def batch(self,slots):
  scenes=[];images=[];wrists=[];states=[];tokens=[];masks=[];actions=[];finished=[]
  for i in slots:
   r=self.records[self.pool[i][self.pos[i]]];t=self.t[i];objects,meta,arr=read_frame(self.root / r['frames'][t]);assert meta['recipe_sha256']==self.m['recipe']
   objects=attach_masks(objects,arr['masks'],meta['mask_instance_ids']);rgb=torch.from_numpy(arr['rgb'].copy()).permute(2,0,1).float()/127.5-1;wr=torch.from_numpy(arr['wrist'].copy()).permute(2,0,1).float()/127.5-1
   assert rgb.shape==wr.shape==(3,224,224)
   key=f"{r['key']}::slot{self.rank}_{i}"
   ctx=dict(episode_key=key,step=t,prompt_ids=meta['prompt_ids'],geometry_generation=meta['geometry']['generation'])
   ids,lmask=tokenize_object_labels(objects,self.tok);scenes.append(GraphScene(objects,ids,lmask,ctx,{'base_0_rgb':image_digest(rgb)},8))
   a,s=targets(self.root / r['targets']);a=a[np.minimum(np.arange(t,t+10),len(a)-1)];state=s[t]
   def norm(x,name):
    n=self.m['normalization'][name];return (x-np.array(n['q01']))/(np.array(n['q99'])-n['q01']+1e-6)*2-1
   padded=np.zeros((10,32),np.float32);padded[:,:7]=norm(a,'actions');st=np.zeros(32,np.float32);st[:8]=norm(state,'state')
   text=r['instruction'].strip().replace('_',' ').replace('\n',' ');tok=(self.tok.encode(text,add_bos=True)+self.tok.encode('\n'))[:200]
   tokens.append(tok+[0]*(200-len(tok)));masks.append([True]*len(tok)+[False]*(200-len(tok)));actions.append(torch.from_numpy(padded));states.append(torch.from_numpy(st));images.append(rgb);wrists.append(wr)
   self.t[i]+=8
   if self.t[i]>=len(r['frames']):
    finished.append(key);self.pos[i]=(self.pos[i]+1)%len(self.pool[i]);self.t[i]=self.rng.randrange(min(8,len(self.records[self.pool[i][self.pos[i]]]['frames'])))
  x=torch.stack(images);ok=torch.ones(len(slots),dtype=torch.bool)
  obs=SimpleNamespace(images=dict(base_0_rgb=x,left_wrist_0_rgb=torch.stack(wrists),right_wrist_0_rgb=torch.zeros_like(x)),image_masks=dict(base_0_rgb=ok,left_wrist_0_rgb=ok,right_wrist_0_rgb=~ok),state=torch.stack(states),tokenized_prompt=torch.tensor(tokens),tokenized_prompt_mask=torch.tensor(masks),token_ar_mask=None,token_loss_mask=None)
  return obs,scenes,torch.stack(actions),finished

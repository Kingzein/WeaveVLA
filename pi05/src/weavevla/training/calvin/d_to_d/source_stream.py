import os
import json
import random
from types import SimpleNamespace
import numpy as np
import torch
import sentencepiece as spm
from pathlib import Path
from weavevla_pi05.adapter import GraphScene, image_digest, tokenize_object_labels
from weavevla_pi05.alignment import resize_image_and_masks
import hashlib,io
from functools import lru_cache
from weavevla.geometry.prompt import PromptMatcher
SIZES=[8]*6
@lru_cache(maxsize=512)
def episode(path,digest):
    path=Path(path)
    raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==digest
    with np.load(io.BytesIO(raw),allow_pickle=False) as z:
        return {**{k:z[k] for k in ['actions','robot_obs','maps','points','rgb_static','rgb_gripper']},'frames':json.loads(str(z['metadata']))}
@lru_cache(maxsize=1)
def matcher():return PromptMatcher(Path(os.environ['WEAVE_PROMPT_ROOT']))



from threading import Lock
match_lock = Lock()
def prompt_ids(language, nodes):
    with match_lock:
        return matcher()(language, nodes)['prompt_ids']

class Streams:
    def __init__(self, rank, manifest_path, tokenizer):
        manifest = json.loads(manifest_path.read_text())
        assert manifest.get('split') == 'D/training'
        records=manifest['episodes']
        assert manifest['complete'] and len(records)==5124 and manifest['frames']==308918
        self.m=dict(recipe=manifest['geometry_recipe'])
        self.n=SIZES[rank]
        self.pools=[records[sum(SIZES[:rank])+i::48] for i in range(self.n)]
        self.rng=random.Random(13711+rank)
        for pool in self.pools:self.rng.shuffle(pool)
        self.pos=[0]*self.n
        self.t=[self.rng.randrange(min(8,p[0]['length'])) for p in self.pools]
        self.tokenizer=spm.SentencePieceProcessor(model_file=str(tokenizer))

    def batch(self,slots):
        assert len(set(slots))==len(slots) and all(0<=i<self.n for i in slots)
        device='cpu';states=[]
        imgs=[]; wrists=[]; prompts=[]; masks=[]; scenes=[]; targets=[]; valid=[]; finished=[]
        for i in slots:
            pool=self.pools[i]
            r=pool[self.pos[i]]; t=self.t[i]; key=f"D-{r['episode']}"
            d=episode(r['files'][0],r['sha256']);frame=d['frames'][t];labels=frame['labels'];maps=d['maps'][t]
            nodes=[]
            for original in frame['nodes']:
                node=original.copy();a,b=node.pop('point_range');node['points']=d['points'][a:b]
                node['visual_mask']=np.isin(maps,[int(k) for k,v in labels.items() if v=='gripper fingers']) if node['id']=='gripper' else maps==int(node['id'])
                nodes.append(node)
            lang=r['language'];ctx=dict(episode_key=key,step=t,prompt_ids=prompt_ids(lang,nodes),sampling_offset=t%8,geometry_generation=0)
            chunk=np.zeros((8,7),np.float32)
            actual=d['actions'][t:t+8]
            chunk[:len(actual)]=actual
            chunk[len(actual):,6]=actual[-1,6]
            ex=dict(lang=lang,action=chunk)
            st=np.zeros(32,np.float32);states.append(st)
            raw,wr=[torch.tensor(d[k][t]).permute(2,0,1) for k in ('rgb_static','rgb_gripper')]
            rgb,m=resize_image_and_masks(raw,torch.stack([torch.as_tensor(o['visual_mask']) for o in nodes]))
            rgb=rgb.float()/127.5-1
            wr,_=resize_image_and_masks(wr,torch.empty((0,*wr.shape[1:]),dtype=torch.bool))
            for o,mask in zip(nodes,m):
                o.update(visual_mask=mask,image_index=0)
            ids,lmask=tokenize_object_labels(nodes,self.tokenizer)
            scenes.append(GraphScene(nodes,ids,lmask,ctx,{'base_0_rgb':image_digest(rgb)},8))
            text=ex['lang'].strip().replace('_',' ').replace('\n',' ')
            tok=(self.tokenizer.encode(text,add_bos=True)+self.tokenizer.encode('\n'))[:48]
            prompts.append(tok+[0]*(48-len(tok)));masks.append([True]*len(tok)+[False]*(48-len(tok)))
            imgs.append(rgb);wrists.append(wr.float()/127.5-1)
            a=torch.zeros(8,32);a[:,:7]=torch.as_tensor(ex['action']);targets.append(a)
            valid.append(torch.arange(8)[:,None] < r['length']-t)
            self.t[i]+=8
            if self.t[i]>=r['length']:
                finished.append(key);self.pos[i]=(self.pos[i]+1)%len(pool)
                self.t[i]=self.rng.randrange(min(8,pool[self.pos[i]]['length']))
        x=torch.stack(imgs).to(device);ok=torch.ones(len(slots),dtype=torch.bool,device=device)
        obs=SimpleNamespace(images=dict(base_0_rgb=x,left_wrist_0_rgb=torch.stack(wrists).to(device),right_wrist_0_rgb=torch.zeros_like(x)), image_masks=dict(base_0_rgb=ok,left_wrist_0_rgb=ok,right_wrist_0_rgb=~ok),state=torch.tensor(np.stack(states)), tokenized_prompt=torch.tensor(prompts,device=device),tokenized_prompt_mask=torch.tensor(masks,device=device),token_ar_mask=None,token_loss_mask=None)
        return obs,scenes,torch.stack(targets).to(device),torch.stack(valid).to(device),finished

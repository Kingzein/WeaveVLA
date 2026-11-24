import os,sys,json,time,random,datetime,hashlib,fcntl,argparse
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

def atomic(path,d):
 path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(path)

def sync_gradients(params,world):
 used=torch.tensor([p.grad is not None for p in params],device='cuda',dtype=torch.int32)
 dist.all_reduce(used,op=dist.ReduceOp.MAX)
 selected=[]
 for p,u in zip(params,used.tolist()):
  if not u:continue
  if p.grad is None:p.grad=torch.zeros_like(p)
  selected.append(p)
 # Bounded dtype-homogeneous buckets; unused-on-all-ranks params stay None.
 for dtype in sorted({p.dtype for p in selected},key=str):
  bucket=[];size=0
  def flush():
   if not bucket:return
   flat=torch.cat([p.grad.reshape(-1) for p in bucket]);dist.all_reduce(flat);flat.div_(world)
   offset=0
   for p in bucket:
    p.grad.copy_(flat[offset:offset+p.numel()].view_as(p));offset+=p.numel()
  for p in selected:
   n=p.numel()*p.element_size()
   if bucket and size+n>32*1024**2:flush();bucket=[];size=0
   bucket.append(p);size+=n
  flush()


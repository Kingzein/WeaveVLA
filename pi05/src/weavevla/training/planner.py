"""Causal ID/topology planner, no trainable computation or ground truth."""
import copy,numpy as np
class Planner:
 def __init__(self,n=64,e=512,p=512):self.n=n;self.e=e;self.p=p;self.reg={};self.edge={};self.pairs=[];self.step=-1;self.generation=None
 def pack(self,objects,ctx):
  step=int(ctx['step']);assert step>self.step;self.step=step
  ids=[str(x['id']) for x in objects];assert len(ids)==len(set(ids))
  gen={int(x['geometry_generation']) for x in objects};assert len(gen)<=1;generation=next(iter(gen),ctx.get('geometry_generation',self.generation))
  if generation!=self.generation:
   self.edge={};self.pairs=[]
   for r in self.reg.values():r['usable']=False
  self.generation=generation;source={};update={};keep={};obj_byid=dict(zip(ids,objects))
  for oid,o in zip(ids,objects):
   label=o['label'];s=o['geometry_source'];pts=np.asarray(o['points'],np.float32);prev=self.reg.get(oid);valid=len(pts)>=16
   if prev:assert prev['label']==label
   if s in ['fresh','static_supported','tcp_only']:
    assert s=='tcp_only' or (valid and o['visible_now'])
    if label=='robot gripper':assert o['position_source']=='tcp'
    slot=prev['slot'] if prev else len(self.reg);assert slot<self.n
    self.reg[oid]=dict(slot=slot,label=label,usable=True,center=np.array(o['geometry_center'],np.float32))
    update[oid]=True;keep[oid]=s=='tcp_only' and prev is not None
   elif prev:
    if s=='occluded_hold' and valid and label!='robot gripper':prev['usable']=True;prev['center']=pts.mean(0)
    else:prev['usable']=False
    update[oid]=False;keep[oid]=True
   source[oid]=s
  for oid in set(self.reg)-set(source):source[oid]='missing';self.reg[oid]['usable']=False;keep[oid]=True;update[oid]=False
  grippers=[k for k,v in self.reg.items() if v['label']=='robot gripper'];assert len(grippers)<=1;g=grippers[0] if grippers else None
  usable={k for k,v in self.reg.items() if v['usable']};observed={k for k in usable if source[k] in ['fresh','static_supported','tcp_only']};prompt=set(map(str,ctx['prompt_ids']))&usable
  ob=usable-({g} if g else set());prompt &= ob
  pairs={(a,b) for a in prompt for b in prompt if a!=b}
  if g in usable:pairs|={(a,g) for a in prompt}|{(g,a) for a in prompt}
  pairs|={(a,b) for a,b in self.pairs if a in ob and b in ob and a!=b and (a in prompt or b in prompt) and (a not in observed or b not in observed)}
  pairs|={(b,a) for a,b in tuple(pairs)}
  for a in sorted(prompt&observed):
   ranked=sorted((ob&observed)-{a},key=lambda b:(float(np.linalg.norm(self.reg[b]['center']-self.reg[a]['center'])),b))
   for b in ranked[:2]:pairs|={(a,b),(b,a)}
  pairs=sorted(pairs);self.pairs=pairs
  active=sorted(usable|(set(map(str,ctx['prompt_ids']))&self.reg.keys())|set(grippers));assert len(active)<=self.n and len(pairs)<=self.e
  self.edge={k:v for k,v in self.edge.items() if step-v<=64}
  x=dict(node_mask=np.zeros(self.n,bool),node_slots=np.zeros(self.n,np.int32),points=np.zeros((self.n,self.p,3),np.float32),point_mask=np.zeros((self.n,self.p),bool),keep_memory=np.zeros(self.n,bool),update_content=np.zeros(self.n,bool),src=np.zeros(self.e,np.int32),dst=np.zeros(self.e,np.int32),edge_mask=np.zeros(self.e,bool),edge_observed=np.zeros(self.e,bool),edge_exists=np.zeros(self.e,bool),geometry=np.zeros((self.e,4),np.float32))
  ix={v:i for i,v in enumerate(active)}
  for i,oid in enumerate(active):
   x['node_mask'][i]=True;x['node_slots'][i]=self.reg[oid]['slot'];x['keep_memory'][i]=keep.get(oid,True);x['update_content'][i]=update.get(oid,False)
   if not x['keep_memory'][i]:
    pts=np.array(obj_byid[oid]['points'],np.float32)
    if len(pts):
     indices=np.linspace(0,len(pts)-1,min(len(pts),self.p),dtype=np.float32).astype(np.int64);pc=pts[indices]-pts.mean(0);x['points'][i,:len(pc)]=pc;x['point_mask'][i,:len(pc)]=True
  for i,(a,b) in enumerate(pairs):
   x['src'][i]=ix[a];x['dst'][i]=ix[b];x['edge_mask'][i]=True;x['edge_observed'][i]=a in observed and b in observed;x['edge_exists'][i]=(a,b) in self.edge
   delta=self.reg[b]['center']-self.reg[a]['center'];x['geometry'][i]=np.r_[delta,np.linalg.norm(delta)]
   if x['edge_observed'][i]:self.edge[a,b]=step
  return x,active,pairs
 def state(self):return copy.deepcopy(self.__dict__)
 def restore(self,s):self.__dict__=copy.deepcopy(s)

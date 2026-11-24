"""JAX training update for the WeaveVLA policy."""
import copy
import numpy as np
import jax,jax.numpy as j,optax
from flax import nnx
from openpi.models import model as bm
from openpi.models.pi0_config import Pi0Config
from openpi.training import sharding
from openpi.shared import array_typing as at
from .adapter_jax import GraphPi05

def state_from(template,pure):
 x=copy.deepcopy(template);x.replace_by_pure_dict(pure);return x

def spec(tree):
 with at.disable_typechecking():return jax.tree.map(lambda x:jax.ShapeDtypeStruct(x.shape,x.dtype),tree)

def observation(a):
 return bm.Observation(images=dict(base_0_rgb=a['image'],left_wrist_0_rgb=a['wrist']),image_masks={k:np.ones(len(a['image']),bool) for k in ['base_0_rgb','left_wrist_0_rgb']},state=a['state'],tokenized_prompt=a['tokens'],tokenized_prompt_mask=a['token_mask'])

class Core:
 def __init__(self,graph_init,*,dtype='bfloat16',fsdp=2, action_horizon=8, peak_lr=5e-5, warmup=10000, geo_weight=.1):
  self.geo_weight = geo_weight
  self.cfg=Pi0Config(pi05=True,action_horizon=action_horizon,discrete_state_input=False,dtype=dtype)
  model=nnx.eval_shape(lambda:GraphPi05(self.cfg,nnx.Rngs(0),graph_init))
  self.gd,self.p_shape,self.rest=nnx.split(model,nnx.Param,...)
  self.schedule = optax.linear_schedule(peak_lr / (warmup + 1), peak_lr, max(warmup, 1)) if warmup else optax.constant_schedule(peak_lr)
  self.tx=optax.chain(optax.clip_by_global_norm(1.),optax.adamw(self.schedule,b1=.9,b2=.95,eps=1e-8,weight_decay=1e-10))
  self.o_shape=jax.eval_shape(self.tx.init,self.p_shape)
  self.mesh=sharding.make_mesh(fsdp);self.rep=jax.sharding.NamedSharding(self.mesh,jax.sharding.PartitionSpec())
  self.ds=jax.sharding.NamedSharding(self.mesh,jax.sharding.PartitionSpec(sharding.DATA_AXIS))
  self.ps=sharding.fsdp_sharding(self.p_shape,self.mesh);self.os=sharding.fsdp_sharding(self.o_shape,self.mesh)
 def compile(self,batch,memory):
  ob=observation(batch['arrays']);a=batch['arrays']
  def update(p,o,ema,mem,ob,actions,pack,valid,reset,finished,rng):
   mem=jax.tree.map(lambda x:j.where(reset.reshape((-1,)+(1,)*(x.ndim-1)),0.,x),mem)
   step_rng,next_rng=jax.random.split(rng)
   m=nnx.merge(self.gd,p,self.rest);m.train()
   def loss_fn(m):
    errors,newmem,_,(geos,has_geometry)=m.graph_loss(step_rng,ob,actions,pack,mem,train=True)
    action_mask=valid.astype(j.float32)
    flow=(errors*action_mask[...,None]).sum()/j.maximum(action_mask.sum()*7,1)
    count=has_geometry.reshape(6,8).sum(-1)
    geo=(geos.reshape(6,8).sum(-1)/j.maximum(count,1)).mean()
    return flow+self.geo_weight*geo,(newmem,flow,geo)
   (loss,(newmem,flow,geo)),grad=nnx.value_and_grad(loss_fn,has_aux=True)(m)
   updates,newopt=self.tx.update(grad,o,p);newp=optax.apply_updates(p,updates)
   newema=jax.tree.map(lambda x,y:.999*x+.001*y,ema,newp)
   newmem=jax.tree.map(lambda x:j.where(finished.reshape((-1,)+(1,)*(x.ndim-1)),0.,x),newmem)
   return newp,newopt,newema,newmem,next_rng,dict(loss=loss,flow_loss=flow,geometry_loss=geo,geometry_head_grad_norm=optax.global_norm({k:v for k,v in grad['graph_params'].items() if k.startswith('geometry_head.')}),gradient_norm=optax.global_norm(grad))
  fn=jax.jit(update,in_shardings=(self.ps,self.os,self.ps,self.ds,self.ds,self.ds,self.ds,self.ds,self.ds,self.ds,self.rep),out_shardings=(self.ps,self.os,self.ps,self.ds,self.rep,self.rep),donate_argnums=(0,1,2,3))
  with at.disable_typechecking(),sharding.set_mesh(self.mesh):
   return fn.lower(self.p_shape,self.o_shape,self.p_shape,spec(memory),spec(ob),spec(a['actions']),spec(batch['pack']),spec(a['valid_steps']),spec(a['reset']),spec(a['finished']),jax.ShapeDtypeStruct((2,),np.uint32)).compile()
 def inputs(self,batch):
  a=batch['arrays'];return tuple(jax.device_put(v,self.ds) for v in [observation(a),a['actions'],batch['pack'],a['valid_steps'],a['reset'],a['finished']])

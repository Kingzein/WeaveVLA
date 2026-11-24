"""JAX action-expert graph adapter; official Pi0 backbone/loss computation.

No change to the original prefix attention. Graph tokens are prepended to the
expert suffix. Explicit graph memory is an auxiliary non-optimized return.
"""
import dataclasses
import numpy as np
import jax,jax.numpy as j
from flax import nnx
import augmax
from augmax import InputType
from openpi.models.pi0 import Pi0,make_attn_mask
from ..graph_jax import graph_batch_apply

def augment_aligned(rng,observation,masks):
 images={};warped=masks
 for key,image in observation.images.items():
  transforms=[]
  if 'wrist' not in key:transforms=[augmax.RandomCrop(212,212),augmax.Resize(224,224),augmax.Rotate((-5,5))]
  transforms += [augmax.ColorJitter(brightness=.3,contrast=.4,saturation=.5)]
  keys=jax.random.split(rng,len(image))
  if key=='base_0_rgb':
   chain=augmax.Chain(*transforms,input_types=(InputType.IMAGE,InputType.MASK))
   image,m=jax.vmap(chain)(keys,((image+1)/2,j.moveaxis(masks,1,-1).astype(j.float32)));warped=j.moveaxis(m,-1,1)>.5
  else:image=jax.vmap(augmax.Chain(*transforms))(keys,(image+1)/2)
  images[key]=image*2-1
 return dataclasses.replace(observation,images=images),warped

class GraphPi05(Pi0):
 def __init__(self,config,rngs,graph_init):
  super().__init__(config,rngs)
  self.graph_params=nnx.Dict({k:nnx.Param(j.array(v,j.float32)) for k,v in graph_init.items()})
 def graph_loss(self,rng,obs,actions,pack,memory,*,train=False,noise=None,time=None,return_tokens=False):
  obs=dataclasses.replace(obs,images={k:v for k,v in obs.images.items() if k!='right_wrist_0_rgb'},image_masks={k:v for k,v in obs.image_masks.items() if k!='right_wrist_0_rgb'})
  prng,nrng,trng=jax.random.split(rng,3)
  masks=pack['pixel_masks']
  if train:obs,masks=augment_aligned(prng,obs,masks)
  # Inputs are aligned224 and normalized by the host adapter. Empty right camera
  # is omitted explicitly, matching the active PyTorch optimized policy.
  prefix,pm,pa=super().embed_prefix(obs)
  label=pack['label_ids'];b,n,l=label.shape
  # Reuse the exact unscaled FP32 embedding table, matching the Torch graph.
  # Calling Linen Embedder.encode would scale then round to BF16 first.
  tables=[v for v in jax.tree.leaves(nnx.state(self.PaliGemma.llm,nnx.Param)) if hasattr(v,'shape') and v.ndim==2 and v.shape[0]>=257152 and v.shape[1]==2048]
  assert len(tables)==1,'Expected one shared PaliGemma vocabulary table'
  sem=tables[0][label].astype(j.float32)
  lm=pack['label_mask'][...,None];sem=(sem*lm).sum(2)/j.maximum(lm.sum(2),1)
  coverage=masks.reshape(b,n,16,14,16,14).sum((3,5),dtype=j.float32).reshape(b,n,256)
  coverage=coverage/j.maximum(coverage.sum(-1,keepdims=True),1)
  coverage=coverage*pack['visual_allowed'][...,None]
  visuals=j.einsum('bnl,bld->bnd',coverage,prefix[:,:256].astype(j.float32)).astype(prefix.dtype).astype(j.float32)
  graph_fields={k:v for k,v in pack.items() if k not in ['pixel_masks','label_ids','label_mask','visual_allowed']}
  graph_fields.update(semantics=sem,visuals=visuals,prefix=prefix,prefix_mask=pm)
  with jax.default_matmul_precision('highest'):
   tokens,new_memory,geometry_aux=graph_batch_apply({k:v.value for k,v in self.graph_params.items()},graph_fields,memory)
  if noise is None:noise=jax.random.normal(nrng,actions.shape)
  if time is None:time=jax.random.beta(trng,1.5,1.,actions.shape[:-2])*.999+.001
  xt=time[...,None,None]*noise+(1-time[...,None,None])*actions;target=noise-actions
  suffix,sm,sa,cond=super().embed_suffix(obs,xt,time)
  valid=j.ones(tokens.shape[:2],bool);boundary=j.zeros((tokens.shape[1],),bool).at[0].set(True)
  suffix=j.concatenate([tokens.astype(suffix.dtype),suffix],1);sm=j.concatenate([valid,sm],1);sa=j.concatenate([boundary,sa])
  im=j.concatenate([pm,sm],1);ar=j.concatenate([pa,sa]);mask=make_attn_mask(im,ar);positions=j.cumsum(im,axis=1)-1
  (_,out),_=self.PaliGemma.llm([prefix,suffix],mask=mask,positions=positions,adarms_cond=[None,cond])
  velocity=self.action_out_proj(out[:,-self.action_horizon:]);error=j.square(velocity-target).mean(-1)
  if return_tokens:return error,new_memory,velocity,geometry_aux,tokens
  return error,new_memory,velocity,geometry_aux

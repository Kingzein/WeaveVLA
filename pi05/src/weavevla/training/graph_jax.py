"""Pure JAX neural graph. IDs/topology are causal host metadata.

Flat PyTorch state-dict parameter names are intentionally preserved so parity
uses the exact same learned weights. All trainable math is traceable. Persistent
memory is explicit, FP32, and stop_gradient implements BPTT=1.
"""
import jax
import jax.numpy as j

def linear(p,k,x):return x@p[k+'.weight'].T+p[k+'.bias']
def mlp(p,k,x):return linear(p,k+'.2',jax.nn.gelu(linear(p,k+'.0',x),approximate=False))
def norm(p,k,x):
 mean=j.mean(x,-1,keepdims=True);var=j.mean(j.square(x-mean),-1,keepdims=True)
 return (x-mean)*jax.lax.rsqrt(var+1e-5)*p[k+'.weight']+p[k+'.bias']
def attention(p,k,q,kv,mask=None,heads=8):
 d=q.shape[-1]
 if k+'.in_proj_weight' in p:
  w=p[k+'.in_proj_weight'];b=p[k+'.in_proj_bias'];qq=q@w[:d].T+b[:d];kk=kv@w[d:2*d].T+b[d:2*d];vv=kv@w[2*d:].T+b[2*d:]
 else:
  b=p[k+'.in_proj_bias'];qq=q@p[k+'.q_proj_weight'].T+b[:d];kk=kv@p[k+'.k_proj_weight'].T+b[d:2*d];vv=kv@p[k+'.v_proj_weight'].T+b[2*d:]
 width=d//heads;qq=qq.reshape(-1,heads,width);kk=kk.reshape(-1,heads,width);vv=vv.reshape(-1,heads,width)
 score=j.einsum('ihd,jhd->hij',qq,kk)/j.sqrt(float(width))
 if mask is not None:score=j.where(mask[None,None,:],score,-j.inf)
 a=jax.nn.softmax(score,-1);out=j.einsum('hij,jhd->ihd',a,vv).reshape(-1,d)
 return linear(p,k+'.out_proj',out)
def sparse(p,k,nodes,edges,src,dst,edge_mask,node_mask,heads=8):
 n,d=nodes.shape;w=d//heads;q,kk,v=j.moveaxis(linear(p,k+'.qkv',nodes).reshape(n,3,heads,w),1,0)
 scores=j.einsum('ihd,jhd->hij',q,kk)/j.sqrt(float(w))
 bias=j.zeros((n,n,heads),nodes.dtype).at[src,dst].add(linear(p,k+'.bias',edges)*edge_mask[:,None])
 ev=j.zeros((n,n,heads,w),nodes.dtype).at[src,dst].add(linear(p,k+'.edge_value',edges).reshape(-1,heads,w)*edge_mask[:,None,None])
 allowed=j.eye(n,dtype=bool).at[src,dst].max(edge_mask)
 weights=jax.nn.softmax(j.where(allowed[None],scores+j.moveaxis(bias,2,0),-j.inf),-1)
 message=j.einsum('hij,ijhd->ihd',weights,v[None]+ev).reshape(n,d)
 updated=norm(p,k+'.node_norm',nodes+linear(p,k+'.out',message));updated=norm(p,k+'.ffn_norm',updated+mlp(p,k+'.ffn',updated))
 edges=norm(p,k+'.edge_norm',edges+mlp(p,k+'.edge_update',j.concatenate([nodes[src],nodes[dst],edges],-1)))
 return updated*node_mask[:,None],edges*edge_mask[:,None]
def gru(p,x,h):
 ix=x@p['edge_gru.weight_ih'].T+p['edge_gru.bias_ih'];hx=h@p['edge_gru.weight_hh'].T+p['edge_gru.bias_hh']
 ir,iz,inn=j.split(ix,3,-1);hr,hz,hn=j.split(hx,3,-1);r=jax.nn.sigmoid(ir+hr);z=jax.nn.sigmoid(iz+hz);n=j.tanh(inn+r*hn)
 return (1-z)*n+z*h

def initial_memory(registry_size=64,dim=256):
 return dict(nodes=j.zeros((registry_size,dim),j.float32),edges=j.zeros((registry_size,registry_size,dim),j.float32))

def graph_apply(p,x,memory,*,pre_layers=2,task_layers=2,qformer_layers=2):
 """One scene, fixed padded N/E/P. x masks make padding gradient-neutral.

 node_slots select persistent ID slots. keep_memory selects existing content
 for occluded/tcp-only nodes; update_content commits fresh/tcp-only content.
 edge_exists is previous valid GRU state after host TTL/generation checks.
 edge_observed controls GRU update, independent of mask visibility.
 """
 nm=x['node_mask'];em=x['edge_mask'];src=x['src'];dst=x['dst'];slots=x['node_slots']
 shape=mlp(p,'point_encoder',x['points']);shape=j.max(j.where(x['point_mask'][...,None],shape,-j.inf),axis=1);shape=j.where(j.any(x['point_mask'],-1,keepdims=True),shape,0.)
 parts=[linear(p,'semantic',x['semantics']),shape,linear(p,'visual_projection',x['visuals'])]
 content=mlp(p,'node_encoder',j.concatenate(parts,-1));content=j.where(x['keep_memory'][:,None],jax.lax.stop_gradient(memory['nodes'][slots]),content)*nm[:,None]
 current_nodes=j.where(x['update_content'][:,None],content,jax.lax.stop_gradient(memory['nodes'][slots]))
 next_node_memory=memory['nodes'].at[slots].add((current_nodes-memory['nodes'][slots])*nm[:,None])
 geometry=x['geometry'];edges=mlp(p,'edge_encoder',geometry)*em[:,None];nodes=content
 for i in range(pre_layers):nodes,edges=sparse(p,f'pre.{i}',nodes,edges,src,dst,em,nm)
 gs,gd=slots[src],slots[dst]
 prev=j.where(x['edge_exists'][:,None],jax.lax.stop_gradient(memory['edges'][gs,gd]),0.)
 hidden=j.where(x['edge_observed'][:,None],gru(p,edges,prev),prev)
 edges=mlp(p,'memory_fusion',j.concatenate([edges,hidden],-1))*em[:,None]
 context=attention(p,'relation_attention',edges,x['prefix'].astype(j.float32),x['prefix_mask'])
 edges=norm(p,'fusion_norm',edges+context);edges=norm(p,'fusion_ffn_norm',edges+mlp(p,'fusion_ffn',edges))*em[:,None]
 for i in range(task_layers):nodes,edges=sparse(p,f'task.{i}',nodes,edges,src,dst,em,nm)
 valid_geometry=em & x['edge_observed']
 residual=mlp(p,'geometry_head',edges)-jax.lax.stop_gradient(geometry)
 absolute=j.abs(residual);huber=j.where(absolute<1.,.5*j.square(residual),absolute-.5)
 geometry_count=valid_geometry.sum()
 geometry_loss=(huber*valid_geometry[:,None]).sum()/j.maximum(geometry_count*4,1)
 bank=j.concatenate([nodes,edges]);bm=j.concatenate([nm,em])
 # No-node scenes must see the learned empty_scene, not padded object features.
 bank=j.concatenate([bank,p['empty_scene']],axis=0);bm=j.concatenate([bm,j.array([~j.any(nm)])])
 queries=p['queries'][0]
 for i in range(qformer_layers):
  k=f'qformer.{i}';queries=norm(p,k+'.norms.0',queries+attention(p,k+'.self_attn',queries,queries));queries=norm(p,k+'.norms.1',queries+attention(p,k+'.cross_attn',queries,bank,bm));queries=norm(p,k+'.norms.2',queries+mlp(p,k+'.ffn',queries))
 out=linear(p,'output_projection',queries)
 # Invalid padded edges may share slot0; additive zeros avoid overwriting it.
 delta=(hidden-memory['edges'][gs,gd])*(em & x['edge_observed'])[:,None]
 next_edges=memory['edges'].at[gs,gd].add(delta)
 state=jax.tree.map(jax.lax.stop_gradient,dict(nodes=next_node_memory,edges=next_edges))
 return out,state,(geometry_loss,geometry_count>0)

def graph_batch_apply(p,x,memory):return jax.vmap(lambda a,b:graph_apply(p,a,b))(x,memory)

import json,os,pathlib,torch,numpy as np,sentencepiece as spm
from types import SimpleNamespace
from weavevla_pi05.adapter import GraphPI05,GraphScene,image_digest,tokenize_object_labels
from openpi.models.pi0_config import Pi0Config
from weavevla.policy.masks import attach_masks
class Policy(GraphPI05):
 def embed_prefix(self,images,img_masks,tokens,mask):
  if len(images)==3 and not img_masks[2].any():images,img_masks=images[:2],img_masks[:2]
  return super().embed_prefix(images,img_masks,tokens,mask)
def move_tied_policy(model,device,dtype=None):
 # These two entries contain identical vocabulary weights. Temporarily detach
 # the unused output alias during transfer so it cannot allocate a second copy.
 head=model.paligemma_with_expert.paligemma.lm_head
 head.weight=None
 model.to(device=device,dtype=dtype)
 head.weight=model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight
 assert head.weight is model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight
 return model
class OnlinePolicy:
 def __init__(self,weights,metadata,out):
  torch.set_num_threads(2);self.norm=json.loads(pathlib.Path(metadata).read_text());self.norm=self.norm.get('normalization',self.norm);self.tok=spm.SentencePieceProcessor(model_file=os.environ['PI05_TOKENIZER'])
  m=Policy(Pi0Config(pi05=True,action_horizon=10,discrete_state_input=False,pytorch_compile_mode=None),geometry_stride=8)
  state=torch.load(weights,mmap=True,map_location='cpu',weights_only=True)
  # The PaliGemma head is the vocabulary embedding alias. JAX deliberately
  # excludes the expert vocab head (unused by action inference/training).
  alias='paligemma_with_expert.paligemma.lm_head.weight';emb='paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight';state[alias]=state[emb]
  m.paligemma_with_expert.gemma_expert.lm_head=torch.nn.Identity()
  m.load_state_dict(state,strict=True,assign=True);del state
  move_tied_policy(m,'cuda',torch.bfloat16);m.graph_model.float();m.eval();self._framework=m;self.context=None;self.current_state=None
  self.metadata=dict(training_data_mix='libero_all',available_unnorm_keys=['franka'],action_chunk_size=8,predicted_action_horizon=10,parameter_choice=os.environ.get('PI05_PARAM_CHOICE','params'))
  self._framework.graph_model.record_trace=False
  pathlib.Path(out,'checkpoint_validation.json').write_text(json.dumps(dict(strict_active_load_passed=True,unused_expert_vocab_head_removed=True,paligemma_head_tied=True,prediction_horizon=10,execute_steps=8,geometry_stride=8,policy_parameters='BF16 inference copy',graph_parameters='FP32',normalization='training q01/q99; no clipping; no additional delta transform'),indent=2))
 def move_to(self,device):
  move_tied_policy(self._framework,device)
 def predict_action(self,examples,unnorm_key=None):
  ex=examples[0];im,wr=ex['image'];rgb=torch.from_numpy(np.array(im).copy()).permute(2,0,1).float()/127.5-1;wrist=torch.from_numpy(np.array(wr).copy()).permute(2,0,1).float()/127.5-1
  objects=attach_masks(ex['pilot_objects'],self.masks,[o['id'] for o in self.registry]);ids,lm=tokenize_object_labels(objects,self.tok);scene=GraphScene(objects,ids,lm,self.context,{'base_0_rgb':image_digest(rgb)},8)
  n=self.norm['state'];raw=np.array(self.current_state);assert raw.shape==(8,);v=(raw-np.array(n['q01']))/(np.array(n['q99'])-n['q01']+1e-6)*2-1;state=np.zeros(32,np.float32);state[:8]=v
  text=ex['lang'].strip().replace('_',' ').replace('\n',' ');tokens=(self.tok.encode(text,add_bos=True)+self.tok.encode('\n'))[:200];mask=[True]*len(tokens)+[False]*(200-len(tokens));tokens+=[0]*(200-len(tokens));dev='cuda';ok=torch.ones(1,dtype=torch.bool,device=dev)
  ob=SimpleNamespace(images=dict(base_0_rgb=rgb[None].to(dev),left_wrist_0_rgb=wrist[None].to(dev),right_wrist_0_rgb=torch.zeros_like(rgb)[None].to(dev)),image_masks=dict(base_0_rgb=ok,left_wrist_0_rgb=ok,right_wrist_0_rgb=~ok),state=torch.tensor(state,device=dev)[None],tokenized_prompt=torch.tensor(tokens,device=dev)[None],tokenized_prompt_mask=torch.tensor(mask,device=dev)[None],token_ar_mask=None,token_loss_mask=None)
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):out=self._framework.sample_actions(dev,ob,num_steps=10,graph_batch=[scene])[0,:,:7].float().cpu().numpy()
  n=self.norm['actions'];out=(out+1)/2*(np.array(n['q99'])-n['q01']+1e-6)+n['q01'];assert out.shape==(10,7) and np.isfinite(out).all();return {'actions':out[None,:8].astype(np.float32)}

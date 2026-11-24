"""Isolated inference process. Observations contain RGB and robot state only."""
import os,subprocess,traceback
from pathlib import Path
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from types import SimpleNamespace
import numpy as np
EVAL=Path(__file__).resolve().parent
import torch
from weavevla.geometry.prompt import PromptMatcher

def action_contract(action):
    action=np.array(action,np.float32,copy=True)
    assert action.shape==(7,) and np.isfinite(action).all()
    action[:6]=np.clip(action[:6],-1,1)
    action[6]=1. if action[6]>0 else -1.
    return action

class Inference:
    def __init__(self,checkpoint):
        self.episode=None
        import sentencepiece as spm
        from openpi.models.pi0_config import Pi0Config
        from .pi_policy import Policy
        self.model=Policy(Pi0Config(pi05=True,action_horizon=8,discrete_state_input=False,pytorch_compile_mode=None),geometry_stride=8).float()
        payload=torch.load(checkpoint,mmap=True,map_location='cpu',weights_only=True)
        state=payload.get('model',payload)
        # The JAX export has input embeddings and no vocabulary-logit heads.
        # Action sampling calls the decoder bodies and action_out_proj.
        backbone=self.model.paligemma_with_expert
        for name in ('paligemma','gemma_expert'):
            branch=getattr(backbone,name)
            key=f'paligemma_with_expert.{name}.lm_head.weight'
            if key not in state:
                branch.lm_head=None
        self.model.load_state_dict(state,strict=True)
        self.model.cuda()
        self.tokenizer=spm.SentencePieceProcessor(model_file=os.environ['PI05_TOKENIZER'])
        self.matcher=PromptMatcher(Path(os.environ['WEAVE_PROMPT_ROOT']))
        self.model.eval().requires_grad_(False);self.model.graph_model.record_trace=False
        self.front,self.other=Pipe()
        env=os.environ.copy();env.update(PYOPENGL_PLATFORM='egl',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',YOLO_CONFIG_DIR=str(Path(os.environ['EVAL_OUT'])/'yolo_config'))
        self.front_process=subprocess.Popen([os.environ['CALVIN_FRONTEND_PYTHON'],str(EVAL/'frontend_service.py'),str(self.other.fileno())],pass_fds=[self.other.fileno()],env=env)
        self.other.close()
        try:
            if not self.front.poll(600):raise TimeoutError('Frozen frontend did not initialize')
            self.receipt=self.front.recv()
            if not self.receipt.get('ready'):raise RuntimeError(self.receipt)
        except BaseException:
            self.close()
            raise
        self.start_sequence(-1)
        self.reset()
    def start_sequence(self,ident):
        self.episode=f'D-sequence-{ident}';self.ident=int(ident);self.t=0;self.chunk=0
        self.model.reset_graph()
    def reset(self):self.chunk=0;self.new_subtask=True
    @torch.inference_mode()
    def step(self,obs,lang):
        rgb=obs['rgb_obs']['rgb_static'];need=self.chunk%8==0
        self.front.send(dict(op='observe',episode=self.episode,step=self.t,instruction=lang,rgb=rgb,robot_obs=obs['robot_obs'],geometry=need))
        if not self.front.poll(300):raise TimeoutError('Frozen frontend request timed out')
        geometry=self.front.recv()
        if 'error' in geometry:raise RuntimeError(geometry['error'])
        if need:
            nodes=geometry['pilot_objects'];assert nodes is not None
            context=dict(episode_key=self.episode,step=self.t,new_subtask=self.new_subtask,prompt_ids=self.matcher(lang,nodes)['prompt_ids'],sampling_offset=0,geometry_generation=geometry['generation'])
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(20261107+self.ident*100000+self.t)
                from weavevla_pi05.adapter import GraphScene,image_digest,tokenize_object_labels
                from weavevla_pi05.alignment import resize_image_and_masks
                raw=torch.from_numpy(np.array(rgb,copy=True)).permute(2,0,1)
                image,masks=resize_image_and_masks(raw,torch.stack([torch.as_tensor(o['visual_mask']) for o in nodes]))
                image=image.float()/127.5-1
                wrist=torch.from_numpy(np.array(obs['rgb_obs']['rgb_gripper'],copy=True)).permute(2,0,1)
                wrist,_=resize_image_and_masks(wrist,torch.empty((0,*wrist.shape[1:]),dtype=torch.bool));wrist=wrist.float()/127.5-1
                for obj,mask in zip(nodes,masks):obj.update(visual_mask=mask,image_index=0)
                ids,lmask=tokenize_object_labels(nodes,self.tokenizer)
                scene=GraphScene(nodes,ids,lmask,context,{'base_0_rgb':image_digest(image)},8)
                text=lang.strip().replace('_',' ').replace('\n',' ')
                tokens=(self.tokenizer.encode(text,add_bos=True)+self.tokenizer.encode('\n'))[:48];mask=[True]*len(tokens)+[False]*(48-len(tokens));tokens += [0]*(48-len(tokens))
                x=image[None].cuda();ok=torch.ones(1,dtype=torch.bool,device='cuda')
                observation=SimpleNamespace(images=dict(base_0_rgb=x,left_wrist_0_rgb=wrist[None].cuda(),right_wrist_0_rgb=torch.zeros_like(x)),image_masks=dict(base_0_rgb=ok,left_wrist_0_rgb=ok,right_wrist_0_rgb=~ok),state=x.new_zeros(1,32),tokenized_prompt=torch.tensor([tokens],device='cuda'),tokenized_prompt_mask=torch.tensor([mask],device='cuda'),token_ar_mask=None,token_loss_mask=None)
                with torch.autocast('cuda',dtype=torch.bfloat16):result=self.model.sample_actions(torch.device('cuda'),observation,num_steps=10,graph_batch=[scene])
                self.actions=result[0,:,:7].float().cpu().numpy()
            self.new_subtask=False
            assert self.actions.shape==(8,7) and np.isfinite(self.actions).all()
        action=action_contract(self.actions[self.chunk%8]);self.t+=1;self.chunk+=1
        return action
    def close(self):
        try:self.front.send(dict(op='close'))
        except (BrokenPipeError,EOFError,OSError):pass
        try:self.front_process.wait(timeout=30)
        except subprocess.TimeoutExpired:self.front_process.terminate();self.front_process.wait(timeout=10)
        self.matcher.close()

def serve(pipe_fd, checkpoint):
    ch=Connection(pipe_fd);policy=None
    try:
        torch.set_num_threads(2);torch.cuda.set_device(0)
        policy=Inference(checkpoint);ch.send(policy.receipt)
        while True:
            req=ch.recv();op=req.pop('op')
            if op=='close':break
            if op=='start':policy.start_sequence(req['id']);ch.send(True)
            elif op=='reset':policy.reset();ch.send(True)
            elif op=='step':
                assert set(req)=={'rgb_static','rgb_gripper','robot_obs','language'}
                obs=dict(rgb_obs={k:req[k] for k in ['rgb_static','rgb_gripper']},robot_obs=req['robot_obs'])
                ch.send(policy.step(obs,req['language']))
            else:raise ValueError(op)
    except BaseException:
        try:ch.send(dict(error=traceback.format_exc()))
        except Exception:pass
        raise
    finally:
        if policy is not None:policy.close()


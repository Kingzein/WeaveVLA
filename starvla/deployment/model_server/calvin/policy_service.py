"""Isolated inference process. Observations contain RGB and robot state only."""
import os,sys,json,subprocess,traceback
from pathlib import Path
from multiprocessing import Pipe
from multiprocessing.connection import Connection
import numpy as np
from PIL import Image
EVAL=Path(__file__).resolve().parent
import torch
from examples.CALVIN.train_files.d.data_contract import denormalize_actions
from starVLA.model.modules.weavevla.geometry.prompt import PromptMatcher

def action_contract(action):
    action=np.array(action,np.float32,copy=True)
    assert action.shape==(7,) and np.isfinite(action).all()
    action[:6]=np.clip(action[:6],-1,1)
    action[6]=1. if action[6]>0 else -1.
    return action

class Inference:
    def __init__(self,checkpoint):
        self.episode=None
        from starVLA.model.modules.weavevla.policy.runtime import load_new_model
        from starVLA.model.modules.weavevla.geometry.geometry import recipe_digest

        recipe=json.loads(Path(os.environ['EVAL_RECIPE']).read_text())
        self.digest=recipe_digest(recipe)
        self.model,_=load_new_model(os.environ['EVAL_CONFIG'],recipe=recipe,base_vlm=os.environ.get('EVAL_BASE_VLM'))
        self.model.load_state_dict(torch.load(checkpoint,mmap=True,map_location='cpu',weights_only=True),strict=True)
        self.stats=json.loads(Path(os.environ['EVAL_STATS']).read_text())['franka']['action']
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
        self.model.graph_model.reset()
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
                self.model.graph_model.contexts=[context]
                images=[Image.fromarray(x).resize((224,224)) for x in (rgb,obs['rgb_obs']['rgb_gripper'])]
                nodes=[dict(node,visual_mask=np.asarray(Image.fromarray(np.asarray(node['visual_mask'],np.uint8)).resize((224,224),Image.Resampling.NEAREST),bool)) for node in nodes]
                example=dict(image=images,lang=lang,pilot_objects=nodes,geometry_recipe_sha256=self.digest)
                with torch.autocast('cuda',dtype=torch.bfloat16):self.actions=np.asarray(self.model.predict_action(examples=[example])['normalized_actions'][0])
                self.actions=denormalize_actions(self.actions,self.stats)
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

def main():
    ch=Connection(int(sys.argv[1]));checkpoint=Path(sys.argv[2]);policy=None
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

if __name__=='__main__':main()

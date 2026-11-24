"""Closed-loop graph inference bridge; all caches generated from current observations."""
import os,json,io,time,traceback,random,uuid
from contextlib import contextmanager
from datetime import datetime,timezone
from pathlib import Path
from multiprocessing.connection import Listener
import numpy as np,torch
from PIL import Image
OUT=Path(os.environ['EVAL_OUT']);OUT.mkdir(parents=True,exist_ok=True)
from cutie_tracker import CutieMaskTracker
from perception_pool import PerceptionPool
from starVLA.model.modules.weavevla.geometry.depth import DA3Metric
from starVLA.model.modules.weavevla.policy.geometry import TemporalGeometry, default_recipe, recipe_digest

from starVLA.model.modules.weavevla.geometry.prompt import PromptMatcher
from starVLA.model.modules.weavevla.geometry.contracts import array_digest
from policy import OnlinePolicy
ckpt=os.environ['EVAL_CKPT'];torch.manual_seed(7);np.random.seed(7);random.seed(7)
policy=OnlinePolicy(ckpt,os.environ['EVAL_STATS'],OUT)
pilot_graph=policy._framework.graph_model
new_recipe=default_recipe();recipe=recipe_digest(new_recipe)
suite=os.environ['EVAL_SUITE'];dataset=os.environ['EVAL_DATASET'];matcher=PromptMatcher(Path(os.environ["WEAVE_PROMPT_ROOT"]))
prompt_binding=None
depthmodel=DA3Metric(os.environ['DA3_WEIGHTS'])
geometry_stabilizer=TemporalGeometry(new_recipe['geometry'])
depth_provenance=None
tracker=CutieMaskTracker(dict(cutie_source=os.environ['CUTIE_SOURCE'],cutie_checkpoint=os.environ['CUTIE_WEIGHTS'],cutie_mem_every=5,cutie_max_internal_size=256)).cuda().eval()
perception_pool=PerceptionPool(OUT)
# policy_device_staging: reserve GPU for initial segmentation.
policy.move_to('cpu');torch.cuda.empty_cache()
perception_pool.start()
RUN_ID=uuid.uuid4().hex
@contextmanager
def timed(timings,name):
 torch.cuda.synchronize();started=time.perf_counter()
 try:yield
 finally:
  torch.cuda.synchronize();timings[name]=timings.get(name,0.)+(time.perf_counter()-started)*1000

def depth(im):
 global depth_provenance
 grid_K=np.array(K,np.float32);grid_K[0]*=im.width/256;grid_K[1]*=im.height/256
 result,depth_provenance=depthmodel(im,grid_K)
 return result
def run_stage(folder,name):
 assert name in ('general_detect','segment'),name
 perception_pool.run(name,folder)
 print('PERCEPTION_STAGE_RESIDENT_V2',name,flush=True)
last=None;folder=None;seeds=[];registry=[];chunk=None
sock=os.environ['EVAL_SOCK']
# Launcher lock ensures this is our stale endpoint, never another active evaluator.
import fcntl
socket_lock=open(sock+'.lock','a');fcntl.flock(socket_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
Path(sock).unlink(missing_ok=True)
listener=Listener(sock,family='AF_UNIX',authkey=b'weavevla-libero-goal-eval')
(OUT/'server_ready.json').write_text(json.dumps({'ready':True,'socket':sock,'pid':os.getpid()}));print('POLICY_READY',policy.metadata,flush=True)
try:
 conn=listener.accept();conn.send_bytes(json.dumps(policy.metadata).encode())
 while True:
  try:payload=conn.recv_bytes()
  except EOFError:
   conn.close();print('WAITING_FOR_NEXT_TASK_CLIENT',flush=True)
   conn=listener.accept();conn.send_bytes(json.dumps(policy.metadata).encode());continue
  if payload==b'STOP':break
  try:
   request_start=time.perf_counter();timings={};request_id=uuid.uuid4().hex
   with np.load(io.BytesIO(payload),allow_pickle=False) as z:
    msg=json.loads(str(z['metadata']));im=Image.fromarray(z['primary']);wrist=Image.fromarray(z['wrist']);robot=z['robot'].astype(bool);grip=z['grip'].astype(bool);K=z['K'].tolist();policy.current_state=z['state'].copy()
   provenance=msg.get('robot_mask',{})
   if (provenance.get('version')!='visible_scene_v4_two_fingers_20261020' or provenance.get('gripper_scope')!='two_finger_subtrees_only' or provenance.get('scene_occlusion') is not True or provenance.get('rgb_state_verified') is not True or not provenance.get('scene_sha256') or provenance.get('segmentation_msaa_samples')!=0):
    raise ValueError('Unverified robot mask: synchronized visible-scene client required')
   if provenance.get('rgb_sha256')!=array_digest(np.asarray(im)) or provenance.get('robot_mask_sha256')!=array_digest(robot) or provenance.get('gripper_mask_sha256')!=array_digest(grip):
    raise ValueError('RGB/mask payload digest mismatch')
   key=(int(msg['task']),int(msg['trial']));step=int(msg['step']);instruction=msg['instruction'];reset=key!=last
   assert im.size==(224,224) and robot.shape==(224,224) and not np.any(grip&~robot)
   with torch.inference_mode():
    dep=None
    if reset:
     assert step==0;last=key;chunk=None;pilot_graph.reset();geometry_stabilizer.reset();prompt_binding=None;tracker.reset('live');torch.manual_seed(7+key[0]*50+key[1]);np.random.seed(7+key[0]*50+key[1])
     policy.move_to('cpu');torch.cuda.empty_cache()
     folder=OUT/f'task_{key[0]:02d}'/f'trial_{key[1]:02d}';folder.mkdir(parents=True,exist_ok=True)
     info={'instruction':instruction,'frames':[0],'episode':key[1]};(folder/'instruction.json').write_text(json.dumps(info));(folder/'all_frames.json').write_text(json.dumps(info))
     (folder/'robot_mask_provenance.json').write_text(json.dumps(msg['robot_mask'],indent=2))
     im.save(folder/'000.png');masked=np.array(im);masked[robot]=0;Image.fromarray(masked).save(folder/'000_detector_input.png');np.save(folder/'000_robot.npy',robot);np.save(folder/'000_gripper.npy',grip)
     with timed(timings,'depth_ms'):dep=depth(im)
     np.save(folder/'000_depth.npy',dep)
     for name in ('general_detect','segment'):
      with timed(timings,name+'_subprocess_ms'):run_stage(folder,name)
     policy.move_to('cuda');torch.cuda.empty_cache()
     objects=json.loads((folder/'report.json').read_text())['frames'][0]['objects'][:-1]
     z=np.load(folder/'000_final.npz');occupied=np.zeros((224,224),bool);seeds=[];registry=[]
     for obj,m in zip(objects,z['object_masks']):
      m=m&~occupied
      if m.sum()<16:continue
      occupied|=m;registry.append(dict(id=len(registry)+1,label=obj['label'],branch=obj['branch']));seeds.append(dict(label=obj['label'],mask=m,score=len(objects)-len(seeds)))
     registry.append(dict(id=len(registry)+1,label='robot gripper',branch='geometry'))
     (folder/'online_registry.json').write_text(json.dumps(registry,indent=2));print('EPISODE_SEEDED',key,registry,flush=True)
    masks=np.zeros((len(registry),224,224),bool)
    if seeds:
     with timed(timings,'tracker_ms'):tracked=tracker.step(im,seeds if reset else [],'live',reset=reset)
     for obj in tracked:
      assert obj['label']==registry[obj['id']-1]['label'];masks[obj['id']-1]=obj['mask']&~robot
    masks[-1]=grip
    assert not (masks[:-1]&robot).any()
    policy_inference=step%policy.metadata['action_chunk_size']==0 or chunk is None
    if policy_inference:
     if dep is None:
      with timed(timings,'depth_ms'):dep=depth(im)
     ex={'image':[im,wrist],'lang':instruction,'graph_workspace_image_index':0,'graph_camera_intrinsics':K,'graph_intrinsics_size':[256,256]}
     grid_K=np.array(K,np.float32);grid_K[0]*=im.width/256;grid_K[1]*=im.height/256
     scene,stability=geometry_stabilizer.update(f'{suite}:{key[0]}:{key[1]}',step,np.array(im),dep,masks,registry,grid_K,fixed_camera=not bool(msg.get('camera_moving',False)),camera_key=msg.get('camera_key','main'),tcp=provenance.get('tcp'))
     ex['pilot_objects']=scene;ex['geometry_recipe_sha256']=recipe
     (folder/f'stability_{step:06d}.json').write_text(json.dumps(dict(depth=depth_provenance,geometry=stability)))
     np.savez_compressed(folder/f'geometry_used_{step:06d}.npz',**{f'points_{i}':o['points'] for i,o in enumerate(scene)},metadata=json.dumps([{k:v for k,v in o.items() if k not in ('points','visual_mask')} for o in scene]))
     if prompt_binding is None:
      prompt_binding=matcher(instruction,scene)
      (folder/'prompt_binding.json').write_text(json.dumps(prompt_binding,indent=2))
     pilot_graph.contexts=[dict(episode_key=f'{suite}:{key[0]}:{key[1]}',step=step,prompt_ids=prompt_binding['prompt_ids'],geometry_generation=stability['generation'])]
     policy.context=pilot_graph.contexts[0];policy.masks=masks;policy.registry=registry
     with timed(timings,'policy_ms'):chunk=policy.predict_action([ex],unnorm_key=dataset if dataset in policy.metadata['available_unnorm_keys'] else 'franka')['actions'][0]
     assert chunk.shape==(8,7) and np.isfinite(chunk).all()
     if step%40==0:print('ACTION',key,step,'seconds',round(timings['policy_ms']/1000,3),'range',float(chunk.min()),float(chunk.max()),flush=True)
    action=chunk[step%len(chunk)].tolist()
   timing={'perception_mode':'resident_v2','run_id':RUN_ID,'request_id':request_id,'utc':datetime.now(timezone.utc).isoformat(),'task':key[0],'trial':key[1],'step':step,'first_frame':reset,'policy_inference':policy_inference,'action_chunk_size':policy.metadata['action_chunk_size'],**timings,'server_request_ms':(time.perf_counter()-request_start)*1000}
   with (OUT/'inference_timing.jsonl').open('a') as f:f.write(json.dumps(timing)+'\n')
   conn.send_bytes(json.dumps({'action':action,'labels':[o['label'] for o in registry],'timing':timing}).encode())
  except Exception:
   error=traceback.format_exc();(OUT/'server_error.txt').write_text(error);conn.send_bytes(json.dumps({'error':error}).encode());raise
finally:
 listener.close();Path(sock).unlink(missing_ok=True);matcher.close()
 if perception_pool is not None:perception_pool.close()

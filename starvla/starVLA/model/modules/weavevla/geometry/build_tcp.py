"""Robot-only FK sidecars bound to committed canonical RGB; no object state used."""
import argparse,json,hashlib,os,fcntl
from pathlib import Path
import numpy as np
import mujoco
from .tcp import camera_tcp

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--render-root',required=True);ap.add_argument('--output',required=True);ap.add_argument('--part',type=int,default=0);ap.add_argument('--parts',type=int,default=1);ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
 root=Path(a.render_root);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 lock=(out/('part_%s.lock'%a.part)).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 records=json.loads((root/'source_manifest.json').read_text())['episodes'][a.part::a.parts]
 if a.limit:records=[r for r in records if (root/'rgb'/r['key']/'complete.json').exists()][:a.limit]
 frames=0;episodes=0;max_error=0.
 for rec in records:
  src=root/'rgb'/rec['key'];marker=src/'complete.json'
  if not marker.exists():
   if a.limit:continue
   raise ValueError('Render incomplete: '+rec['key'])
  marker_hash=sha(marker);complete=json.loads(marker.read_text());hashes={r['file']:r['sha256'] for r in complete['files']}
  dest=out/rec['key'];dest.mkdir(parents=True,exist_ok=True)
  model=mujoco.MjModel.from_xml_path(str(src/'render_model.xml'));data=mujoco.MjData(model)
  # Only robot and gripper scalar joints enter FK. Object qpos remain defaults.
  addresses=[]
  for j in range(model.njnt):
   name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j) or ''
   if name.startswith(('robot0_','gripper0_')):
    if int(model.jnt_type[j]) not in (2,3):raise ValueError('Unsupported nonscalar robot joint')
    addresses.append(int(model.jnt_qposadr[j]))
  if len(addresses)!=9:raise ValueError('Expected Panda 7 joints + 2 fingers: '+str(addresses))
  cid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,'agentview')
  if int(model.cam_bodyid[cid])!=0:raise ValueError('Offline recipe requires world-fixed camera')
  for step in range(complete['frames']):
   scene=src/f'{step:03d}_scene.npz';rgb=src/f'{step:03d}.png'
   assert sha(scene)==hashes[scene.name] and sha(rgb)==hashes[rgb.name]
   with np.load(scene,allow_pickle=False) as z:
    state=z['state'];proprio=z['proprio'];meta=json.loads(str(z['metadata']))
    assert meta['control_step']==step and meta['episode_key']==rec['key']
    mujoco.mj_resetData(model,data);data.qpos[addresses]=state[1+np.asarray(addresses)]
    mujoco.mj_forward(model,data);tcp=camera_tcp(model,data)
    err=float(np.max(np.abs(np.asarray(tcp['position_world'])-proprio[:3])))
    if err>1e-6:raise ValueError('FK disagrees with recorded EEF: '+str(err))
    max_error=max(max_error,err)
   tcp.update(control_step=step,episode_id=rec['key'],rgb_file_sha256=hashes[rgb.name],scene_file_sha256=hashes[scene.name],render_complete_sha256=marker_hash,robot_qpos=data.qpos[addresses].tolist(),fk_eef_max_error=err)
   target=dest/f'{step:06d}.json';tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(tcp,allow_nan=False));tmp.replace(target)
   frames+=1
  episodes+=1
  (dest/'complete.json').write_text(json.dumps(dict(complete=True,frames=complete['frames'],render_complete_sha256=marker_hash)))
  (out/f'progress_{a.part}.json').write_text(json.dumps(dict(episodes=episodes,frames=frames,max_fk_error=max_error)))
 print(json.dumps(dict(episodes=episodes,frames=frames,max_fk_error=max_error)),flush=True)
if __name__=='__main__':main()

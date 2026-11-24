import os
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import sam3.model_builder as builder
from sam3.model.sam3_image_processor import Sam3Processor
O=Path(os.environ['SCENE_WORK']) if 'SCENE_WORK' in os.environ else None
from filtering import filter_masks
load_info={}
def checked_load(model,path):
 ckpt=torch.load(path,map_location='cpu',weights_only=True)
 if 'model' in ckpt and isinstance(ckpt['model'],dict):ckpt=ckpt['model']
 weights={k.replace('detector.',''):v for k,v in ckpt.items() if 'detector' in k}
 weights.update({k.replace('tracker.','inst_interactive_predictor.model.'):v for k,v in ckpt.items() if 'tracker' in k})
 result=model.load_state_dict(weights,strict=False)
 missing_parameters=sorted(set(dict(model.named_parameters()))&set(result.missing_keys))
 load_info.update(missing_keys=result.missing_keys,unexpected_key_count=len(result.unexpected_keys),missing_parameters=missing_parameters)
 assert not missing_parameters,missing_parameters
 print('SAM3_PARAMETER_LOAD_OK',load_info,flush=True)
builder._load_checkpoint=checked_load
model=builder.build_sam3_image_model(checkpoint_path=os.environ['LIBERO_SAM_WEIGHTS'],load_from_HF=False,enable_inst_interactivity=True).eval().requires_grad_(False)
processor=Sam3Processor(model)

phase='yoloe_only'
info=json.loads((O/'queries.json').read_text());frames=info['frames']
det=json.loads((O/('prompt_detections.json' if phase=='prompt' else 'general_detections.json')).read_text())
report=dict(instruction=info['instruction'],queries=[],extractor='none',mask_model='SAM3 box-prompt instance predictor',pipeline_order=['YOLOE_v2','SAM3'],detection_thresholds={'yoloe':.30},sam_mask_predicted_iou_threshold=.25,frames=[])
def iou(a,b):return float((a&b).sum()/max((a|b).sum(),1))
def biou(a,b):
 z=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));return z/max((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-z,1e-9)
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
 for i in frames:
  im=Image.open(O/f'{i:03d}.png').convert('RGB');robot=np.load(O/f'{i:03d}_robot.npy');grip=np.load(O/f'{i:03d}_gripper.npy');masked=np.asarray(Image.open(O/f'{i:03d}_detector_input.png').convert('RGB'))
  assert np.all(masked[robot]==0) and np.array_equal(masked[~robot],np.asarray(im)[~robot])
  state=processor.set_image(im);objs=sorted(det[str(i)],key=lambda o:-o['score']);raw=[]
  for obj in objs:
   mask,q,_=model.predict_inst(state,box=np.array(obj['box'],np.float32),multimask_output=False,normalize_coords=True);raw.append(mask[0].astype(bool)&~robot);obj['sam_predicted_iou']=float(np.asarray(q).reshape(-1)[0])
  raw=np.stack(raw) if raw else np.empty((0,im.height,im.width),bool)
  decisions=filter_masks(raw,[dict(o,yolo_score=o['score']) for o in objs],np.load(O/f'{i:03d}_depth.npy')) if objs else []
  kept=[];objects=[];prompt_count=0
  duplicates=[]
  for j,(mask,decision) in enumerate(zip(raw,decisions)):
   if objs[j]['sam_predicted_iou']<.25:
    decision.update(status='delete',reason='sam_predicted_iou_below_0.25')
   if decision['status']=='delete' or mask.sum()<48:continue
   matches=[k for k,m in enumerate(kept) if iou(mask,m)>=.6 or biou(objs[j]['box'],objects[k]['box'])>=.7]
   if matches:duplicates.append(dict(candidate=objs[j],kept=[objects[k]['label'] for k in matches]));continue
   # Earlier prompt instances own overlap pixels as well as the object label.
   if phase=='merge' and prompt_count:mask=mask&~np.logical_or.reduce(kept[:prompt_count])
   if mask.sum()<48:continue
   kept.append(mask);objects.append(objs[j])
  merged=np.stack(kept) if kept else np.empty((0,im.height,im.width),bool)
  assert not (merged&robot).any() and not (grip&~robot).any()
  if phase=='prompt':
   np.savez_compressed(O/f'{i:03d}_prompt_masks.npz',masks=merged)
   report.setdefault('prompt_frames',{})[str(i)]=dict(objects=objects,duplicates=duplicates,decisions=decisions)
  else:
   final=np.concatenate([merged,grip[None]],axis=0);finalobjs=objects+[dict(label='robot gripper',branch='geometry',source='Panda visual mesh',score=None)]
   np.savez_compressed(O/f'{i:03d}_final.npz',object_masks=merged,combined_masks=final,robot_mask=robot,gripper_mask=grip,labels=np.array([o['label'] for o in finalobjs]),branches=np.array([o['branch'] for o in finalobjs]))
   report['frames'].append(dict(frame=i,objects=finalobjs,object_count=len(objects),prompt_count=prompt_count,general_count=len(objects)-prompt_count,duplicates=duplicates,decisions=decisions))
  print('SAM3_'+phase.upper(),i,len(objects),flush=True)
if phase=='prompt':(O/'prompt_masks.json').write_text(json.dumps(report['prompt_frames'],indent=2))
else:report['complete']=True;(O/'report.json').write_text(json.dumps(report,indent=2))

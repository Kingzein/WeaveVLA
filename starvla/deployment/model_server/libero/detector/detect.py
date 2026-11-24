"""Fixed official vocabulary from fine-tuned YOLOE; no text extractor or DINO."""
import os,json
from pathlib import Path
import numpy as np
from PIL import Image
from ultralytics import YOLOE
W=Path(__file__).resolve().parent;O=Path(os.environ['SCENE_WORK'])
labels=json.loads((W/'labels.json').read_text());info=json.loads((O/'all_frames.json').read_text());i=info['frames'][0]
model=YOLOE(os.environ['LIBERO_YOLO_WEIGHTS']).cuda()
assert list(model.names.values())==labels,'Checkpoint vocabulary mismatch'
im=np.asarray(Image.open(O/f'{i:03d}.png').convert('RGB'));robot=np.load(O/f'{i:03d}_robot.npy');masked=im.copy();masked[robot]=0;Image.fromarray(masked).save(O/f'{i:03d}_detector_input.png')
pred=model.predict(Image.fromarray(masked),imgsz=640,conf=.30,iou=.5,agnostic_nms=True,device=0,verbose=False)[0]
objects=[dict(label=pred.names[int(cls)],box=box,score=score,branch='yoloe_finetuned',source='yoloe_instruction_v2') for box,score,cls in zip(pred.boxes.xyxy.cpu().tolist(),pred.boxes.conf.cpu().tolist(),pred.boxes.cls.cpu().tolist())]
(O/'general_detections.json').write_text(json.dumps({str(i):objects}))
(O/'queries.json').write_text(json.dumps(dict(info,frames=[i],queries=[],extractor='none; fixed fine-tuned official vocabulary')))
print('YOLOE_V2',len(objects),flush=True)

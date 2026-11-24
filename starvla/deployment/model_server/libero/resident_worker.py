"""Reuse final v2 models; execute unchanged stage bodies in fresh request namespaces."""
import os,sys,json,ctypes,signal,traceback,time,argparse
from pathlib import Path
from multiprocessing.connection import Listener
SOURCE=Path(__file__).resolve().parent/'detector'
sys.path.insert(0, str(SOURCE))
AUTHKEY=b'weavevla-yoloe-v2-resident'
p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--socket');p.add_argument('--parent-pid',type=int);a=p.parse_args()
assert ctypes.CDLL(None).prctl(1,signal.SIGTERM,0,0,0)==0
assert os.getppid()==a.parent_pid
import torch
if a.model=='yoloe':
 import numpy as np
 from PIL import Image
 from ultralytics import YOLOE
 labels=json.loads((SOURCE/'labels.json').read_text())
 model=YOLOE(os.environ['LIBERO_YOLO_WEIGHTS']).cuda();assert list(model.names.values())==labels
 src=(SOURCE/'detect.py').read_text();body=compile(src[src.index('im=np.asarray'):],'detect_body','exec')
 state=dict(np=np,Image=Image,model=model,json=json)
else:
 src=(SOURCE/'segment.py').read_text();prefix,body=src.split("phase='yoloe_only'",1)
 state={'__name__':'resident_sam3'};exec(compile(prefix,'sam3_load','exec'),state)
 body=compile("phase='yoloe_only'"+body,'segment_body','exec')
listener=Listener(a.socket,family='AF_UNIX',authkey=AUTHKEY);print('RESIDENT_READY',a.model,os.getpid(),flush=True)
try:
 while True:
  c=listener.accept()
  try:
   m=c.recv();folder=Path(m['folder']);assert folder.is_dir()
   ns=dict(state,O=folder)
   if a.model=='yoloe':
    info=json.loads((folder/'all_frames.json').read_text());ns.update(info=info,i=info['frames'][0])
   started=time.perf_counter();exec(body,ns);torch.cuda.synchronize()
   elapsed=(time.perf_counter()-started)*1000
   del ns
   c.send(dict(ok=True,ms=elapsed,pid=os.getpid()));print('RESIDENT_DONE',folder,elapsed,flush=True)
  except Exception:
   error=traceback.format_exc();print(error,flush=True);c.send(dict(ok=False,error=error))
  finally:c.close()
finally:listener.close();Path(a.socket).unlink(missing_ok=True)

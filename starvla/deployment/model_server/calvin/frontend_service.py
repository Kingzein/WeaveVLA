"""Frozen D-cache frontend; RGB/proprioception only, no simulator labels/depth."""
import json,os,sys,traceback,hashlib
from pathlib import Path
from multiprocessing.connection import Connection
import numpy as np
CACHE=Path(os.environ['CALVIN_FRONTEND_ROOT'])
sys.path[:0]=[str(CACHE),str(CACHE/'frontend')]
from robot_projection import RobotProjection
from annotation_frontend import FrozenModels,AnnotationFrontend

def main():
    ch=Connection(int(sys.argv[1]));projector=None
    try:
        import torch
        torch.set_num_threads(2);torch.set_grad_enabled(False)
        cal=json.loads(Path(os.environ['CALVIN_CALIBRATION']).read_text());K=np.asarray(cal['K'],np.float32);E=np.asarray(cal['world_to_camera'])
        models=FrozenModels(load_depth=True);projector=RobotProjection(K,E,cal);frontend=AnnotationFrontend(models,mode='depth')
        loaded=Path(models.detector.detector.model.ckpt_path)
        digest=hashlib.sha256(loaded.read_bytes()).hexdigest()
        expected=os.environ.get('CALVIN_DETECTOR_SHA256')
        if expected and digest!=expected:raise ValueError(f'Detector checksum differs: {digest}')
        ch.send(dict(ready=True,detector_sha256=digest))
        while True:
            req=ch.recv()
            if req['op']=='close':break
            assert set(req)=={'op','episode','step','instruction','rgb','robot_obs','geometry'}
            rgb=np.asarray(req['rgb']);obs=np.asarray(req['robot_obs']);cad=projector(obs)
            out=frontend.observe(req['episode'],req['step'],req['instruction'],rgb,obs,K,E,cad,need_geometry=req['geometry'])
            ch.send(dict(pilot_objects=out['pilot_objects'],generation=out['generation'],rgb_sha256=out['rgb_sha256']))
    except BaseException:
        try:ch.send(dict(error=traceback.format_exc()))
        except Exception:pass
        raise
    finally:
        if projector:projector.close()
if __name__=='__main__':main()

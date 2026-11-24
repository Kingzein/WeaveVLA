"""Versioned per-frame cache of monocular predicted depth and 2D instances."""
import hashlib,json
from pathlib import Path
import numpy as np
from torch import nn
from starVLA.model.modules.graph.monocular_perception import rgb_image,camera_matrix,unproject

VERSION=1

def digest(image, instruction):
    a=np.asarray(rgb_image(image),dtype=np.uint8)
    h=hashlib.sha256();h.update(str(a.shape).encode());h.update(a.tobytes());h.update(str(instruction).encode())
    return h.hexdigest()

def path_for(root,dataset,episode,step):
    if Path(dataset).name!=dataset:raise ValueError('Unsafe dataset name')
    return Path(root)/dataset/f'episode_{int(episode):06d}'/f'{int(step):06d}.npz'

def attach_cache(sample,cfg,dataset,episode,step):
    if cfg is not None and cfg.get('graph_cache_root'):
        sample['graph_cache_file']=str(path_for(cfg['graph_cache_root'],dataset,episode,step))
    return sample

class CachedMonocularPerception(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg
    def forward(self,examples,track=False):
        batches=[]
        for ex in examples:
            if 'graph_cache_file' not in ex:raise ValueError('Cached perception needs graph_cache_file; preprocess this exact RGB/instruction before training or evaluation')
            view=int(ex.get('graph_workspace_image_index',self.cfg.get('workspace_image_index',0)))
            image=rgb_image(ex['image'][view]);w,h=image.size
            with np.load(ex['graph_cache_file'],allow_pickle=False) as data:
                meta=json.loads(str(data['metadata']))
                if meta['version']!=VERSION or meta['depth_source']!='Depth-Anything-V2-Metric-Indoor-Small-hf':raise ValueError('Unsupported cache provenance')
                recipes=self.cfg.get('cache_recipes')
                if recipes is not None:
                    dataset=meta.get('dataset')
                    # Bind provenance to both the configured dataset and its on-disk namespace.
                    if dataset not in recipes or Path(ex['graph_cache_file']).parent.parent.name!=dataset:
                        raise ValueError('Unknown or mismatched cache dataset')
                    expected=recipes[dataset]
                else:
                    expected=self.cfg['cache_recipe']
                if meta['recipe']!=expected:raise ValueError('Stale perception cache recipe')
                if meta['rgb_instruction_sha256']!=digest(image,ex['lang']):raise ValueError('Cache RGB/instruction mismatch: augmentation or wrong sample')
                if meta['image_index']!=view:raise ValueError('Cache camera mismatch')
                depth=data['depth'].astype(np.float32);masks=data['masks'].astype(bool);labels=data['labels'].tolist()
                if depth.shape!=(h,w) or masks.shape!=(len(labels),h,w):raise ValueError('Cache dimensions mismatch')
                objects=[];k=camera_matrix(ex,self.cfg,(h,w))
                for mask,label in zip(masks,labels):
                    points=unproject(depth,mask,k)
                    if len(points)<int(self.cfg.get('min_points',16)):continue
                    ys,xs=np.nonzero(mask&np.isfinite(depth)&(depth>0));i=np.argmin((xs-xs.mean())**2+(ys-ys.mean())**2)
                    objects.append(dict(label=str(label),points=points,id=len(objects),image_index=view,pixel_uv_normalized=[(float(xs[i])+.5)/w,(float(ys[i])+.5)/h]))
                batches.append(objects[:int(self.cfg.get('max_objects',32))])
        return batches

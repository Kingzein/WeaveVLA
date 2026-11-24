"""Frozen Cutie mask propagation; RGB only, independent state per video stream.

Initial masks are supplied by the segmentation backend. Later frames propagate
them with Cutie's learned memory. Unseen objects need an explicit reseed/reset;
per-frame detections are not silently treated as identity annotations.
"""
from collections import OrderedDict
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from omegaconf import OmegaConf


class CutieMaskTracker(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        root=Path(cfg['cutie_source']).resolve()
        sys.path.insert(0,str(root)) if str(root) not in sys.path else None
        from cutie.model.cutie import CUTIE
        from cutie.inference.inference_core import InferenceCore
        self.core_cls=InferenceCore
        self.config=OmegaConf.load(root/'cutie/config/eval_config.yaml')
        self.config.model=OmegaConf.load(root/'cutie/config/model/base.yaml')
        self.config.model.pretrained_backbones=False
        self.config.mem_every=int(cfg.get('cutie_mem_every',5))
        self.config.use_long_term=True
        self.config.max_internal_size=int(cfg.get('cutie_max_internal_size',480))
        self.config.flip_aug=False
        self.network=CUTIE(self.config).eval()
        weights=torch.load(cfg['cutie_checkpoint'],map_location='cpu',weights_only=True)
        self.network.load_state_dict(weights,strict=True)
        self.requires_grad_(False)
        self.streams=OrderedDict()
        self.max_streams=int(cfg.get('cutie_max_streams',8))
        if self.max_streams<1:raise ValueError('cutie_max_streams must be positive')

    def train(self,mode=True):
        return super().train(False)

    def _apply(self,fn,recurse=True):
        result=super()._apply(fn,recurse=recurse)
        self.network.float()
        # InferenceCore state tensors are not registered module buffers.
        self.streams.clear()
        return result

    def reset(self,stream):
        self.streams.pop(str(stream),None)

    @torch.inference_mode()
    def step(self,image,instances,stream,reset=False):
        stream=str(stream)
        if reset:self.reset(stream)
        device=next(self.network.parameters()).device
        rgb=np.array(image.convert('RGB'));h,w=rgb.shape[:2]
        old=self.streams.pop(stream,None)
        annotation=None
        if old is None:
            labels={};mask=np.zeros((h,w),dtype=np.int64)
            for instance in sorted(instances,key=lambda o:o.get('score',1),reverse=True):
                candidate=np.asarray(instance['mask'],dtype=bool)
                if candidate.shape!=(h,w):raise ValueError('Cutie seed mask shape differs from RGB')
                candidate=candidate&(mask==0)
                if candidate.sum()<16:continue
                ident=len(labels)+1;mask[candidate]=ident;labels[ident]=str(instance['label'])
            if not labels:return []
            core=self.core_cls(self.network,cfg=self.config)
            annotation=torch.from_numpy(mask).to(device)
            old=(core,labels,(h,w))
        core,labels,size=old
        if size!=(h,w):raise ValueError('Cutie stream resolution changed; reset the stream')
        image_tensor=torch.from_numpy(rgb).permute(2,0,1).to(device=device,dtype=torch.float32)/255
        with torch.autocast(device.type,enabled=False):
            prob=core.step(image_tensor,annotation,objects=list(labels) if annotation is not None else None)
        pred=core.output_prob_to_mask(prob).cpu().numpy()
        self.streams[stream]=old
        while len(self.streams)>self.max_streams:self.streams.popitem(last=False)
        return [dict(id=ident,label=label,mask=pred==ident) for ident,label in labels.items() if (pred==ident).sum()>=16]

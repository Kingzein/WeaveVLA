from pathlib import Path
from PIL import Image
from starVLA.model.modules.weavevla.geometry import runtime as base
from starVLA.model.modules.weavevla.geometry.cache import read_frame
from starVLA.model.modules.weavevla.geometry.geometry import default_recipe, recipe_digest
from .visual_patch import MODE
from .masks import attach_masks
from .contracts import pooling_contract

PreparedPerception=base.PreparedPerception

def install_graph(model,recipe=None):
    from .framework import Qwen_GR00T

    if not isinstance(model,Qwen_GR00T):raise TypeError('Use the isolated mask-pooling framework')
    if not model.config.framework.graph.get('use_visual_patches',True):
        raise ValueError('WeaveVLA requires use_visual_patches=True')
    base.install_graph(model,recipe)
    model.config.framework.graph.visual_pooling=MODE
    model.config.framework.graph.visual_pooling_contract=pooling_contract()
    return model

def prepare_training_example(path,recipe=None,allow_provisional=False):
    objects,meta,arrays=read_frame(path,recipe,allow_provisional=allow_provisional)
    attach_masks(objects,arrays['masks'],meta['mask_instance_ids'])
    ex=dict(image=[Image.fromarray(arrays['rgb']),Image.fromarray(arrays['wrist'])],lang=meta['instruction'],action=arrays['action'],pilot_objects=objects,geometry_recipe_sha256=recipe_digest(recipe or default_recipe()))
    if 'state' in arrays:ex['state']=arrays['state']
    ctx=dict(episode_key=meta['episode_id'],step=meta['control_step'],prompt_ids=meta['prompt_ids'],sampling_offset=meta['phase'],geometry_generation=meta['geometry']['generation'])
    return ex,ctx

def load_new_model(base_config,recipe=None,device='cuda',*,base_vlm=None):
    from copy import deepcopy
    from omegaconf import DictConfig, OmegaConf
    from starVLA.model.framework.share_tools import apply_config_compat
    recipe=recipe or default_recipe()
    cfg=deepcopy(base_config) if isinstance(base_config, DictConfig) else OmegaConf.load(Path(base_config))
    cfg=apply_config_compat(cfg)
    if cfg.trainer.pretrained_checkpoint is not None:raise ValueError('Use an explicit initialization plan for fine-tuning')
    if base_vlm is not None:
        cfg.framework.qwenvl.base_vlm=str(base_vlm)
    if not cfg.framework.qwenvl.get('base_vlm'):
        raise ValueError('Pass --base-vlm with the local Qwen backbone directory')
    from .framework import Qwen_GR00T

    model=install_graph(Qwen_GR00T(cfg),recipe)
    model.qwen_vl_interface.model.config.use_cache=False
    return model.to(device),cfg

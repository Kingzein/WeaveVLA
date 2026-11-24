"""Explicit graph installation and shared online geometry adapter."""
from pathlib import Path

from torch import nn

from .geometry import TemporalGeometry, default_recipe, recipe_digest
from .graph import EdgeMemoryGraph
from .contracts import model_contract


class PreparedPerception(nn.Module):
    def __init__(self, recipe):
        super().__init__()
        self.recipe_sha256 = recipe_digest(recipe)

    def forward(self, examples, track=False):
        for ex in examples:
            if ex.get('geometry_recipe_sha256') != self.recipe_sha256:
                raise ValueError('Prepared perception is not from this DA3 geometry recipe')
        return [ex['pilot_objects'] for ex in examples]


def install_graph(model, recipe=None):
    recipe = recipe or default_recipe()
    cfg = model.config.framework.graph
    for key, value in recipe['graph'].items(): cfg[key] = value
    cfg['depth_model'] = recipe['depth']['model']
    cfg['cache_recipe'] = recipe_digest(recipe)
    cfg['version'] = recipe['version']
    cfg['model_contract_sha256'] = model_contract(recipe)
    # Do this before optimizer construction. Old memory checkpoints are not
    # compatible; load a new-version state dict strictly after installation.
    model.graph_model = EdgeMemoryGraph(model.qwen_vl_interface.model.config.hidden_size, cfg)
    model.graph_perception = PreparedPerception(recipe)
    return model


class LiveGeometry:
    def __init__(self, depth_predictor, recipe=None):
        self.recipe = recipe or default_recipe()
        self.depth = depth_predictor
        self.state = TemporalGeometry(self.recipe['geometry'])

    def reset(self):
        self.state.reset()

    def observe(self, episode, step, rgb, masks, instances, K, fixed_camera=True, camera_key='main', tcp=None):
        depth, provenance = self.depth(rgb, K)
        objects, trace = self.state.update(episode, step, rgb, depth, masks, instances, K, fixed_camera, camera_key, tcp=tcp)
        return objects, dict(depth=provenance, geometry=trace, geometry_recipe_sha256=recipe_digest(self.recipe)), depth


def prepare_training_example(path, recipe=None, allow_provisional=False):
    from PIL import Image
    from .cache import read_frame
    recipe = recipe or default_recipe()
    objects, meta, arrays = read_frame(path, recipe, allow_provisional=allow_provisional)
    if 'action' not in arrays:
        raise ValueError('Cache frame lacks original action target')
    example = dict(image=[Image.fromarray(arrays['rgb']), Image.fromarray(arrays['wrist'])],
                   lang=meta['instruction'], action=arrays['action'], pilot_objects=objects,
                   geometry_recipe_sha256=recipe_digest(recipe))
    if 'state' in arrays: example['state'] = arrays['state']
    context = dict(episode_key=meta['episode_id'], step=meta['control_step'], prompt_ids=meta['prompt_ids'],
                   sampling_offset=meta['phase'], geometry_generation=meta['geometry']['generation'])
    return example, context


def load_new_model(base_config, recipe=None, device='cuda'):
    from omegaconf import OmegaConf
    from starVLA.model.framework.share_tools import apply_config_compat
    from starVLA.model.framework.VLM4A.QwenGR00T import Qwen_GR00T
    recipe = recipe or default_recipe()
    cfg = apply_config_compat(OmegaConf.load(Path(base_config)))
    if cfg.trainer.pretrained_checkpoint is not None:
        raise ValueError('New version starts from original VLM, not historical VLA checkpoint')
    cfg.run_id = recipe['version']
    cfg.framework.graph.cache_recipe = recipe_digest(recipe)
    cfg.framework.graph.depth_model = recipe['depth']['model']
    model = install_graph(Qwen_GR00T(cfg), recipe)
    model.qwen_vl_interface.model.config.use_cache = False
    return model.to(device), cfg

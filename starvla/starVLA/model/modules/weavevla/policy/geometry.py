from starVLA.model.modules.weavevla.geometry.geometry import TemporalGeometry as BaseGeometry, default_recipe, recipe_digest
from .masks import attach_masks

class TemporalGeometry(BaseGeometry):
    def update(self, episode, step, rgb, depth_raw, masks, instances, K, *args, **kwargs):
        objects,trace=super().update(episode,step,rgb,depth_raw,masks,instances,K,*args,**kwargs)
        attach_masks(objects,masks,[o['id'] for o in instances])
        return objects,trace

from weavevla_pi05.adapter import GraphPI05

class Policy(GraphPI05):
    def _apply_checkpoint(self,func,*args,**kwargs):
        if func.__qualname__.endswith('PI0Pytorch.forward.<locals>.forward_func'):return func(*args,**kwargs)
        return super()._apply_checkpoint(func,*args,**kwargs)
    def embed_prefix(self,images,img_masks,lang_tokens,lang_masks):
        if len(images)==3 and not img_masks[2].any():images,img_masks=images[:2],img_masks[:2]
        return super().embed_prefix(images,img_masks,lang_tokens,lang_masks)

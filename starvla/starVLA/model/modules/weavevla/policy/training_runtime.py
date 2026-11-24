"""Training adapters; the existing trainer can bind these without changing old runs."""
from pathlib import Path
from starVLA.model.modules.weavevla.geometry.training_runtime import seed_all, tree_map
from .runtime import load_new_model, prepare_training_example
W=Path(__file__).resolve().parent
R=W.parents[1]
DATA_W=W.parent/'geometry'

def load_model(fast_graph=False,base_config=None):
    seed_all(42)
    model,cfg=load_new_model(base_config if base_config is not None else DATA_W/'train.yaml')
    model.graph_model.record_trace=not fast_graph
    model.qwen_vl_interface.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    return model.train(),cfg

def read_frame(record,index,bindings=None,allow_provisional=False,data_root=None):
    root=Path(data_root if data_root is not None else DATA_W).resolve()
    relative=Path(record['frames'][index])
    frame=(root/relative).resolve()
    if relative.is_absolute() or root not in frame.parents:
        raise ValueError('Training frame must stay inside the audited data root')
    ex,ctx=prepare_training_example(frame,allow_provisional=allow_provisional)
    if ctx['episode_key']!=record['key'] or ex['lang']!=record['instruction']:
        raise ValueError('Training example identity mismatch')
    if ctx['step']!=record.get('offset',0)+8*index:raise ValueError('Training sampling phase mismatch')
    return ex,ctx

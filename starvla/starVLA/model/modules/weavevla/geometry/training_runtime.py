import random
from pathlib import Path

import numpy as np
import torch

from .runtime import load_new_model, prepare_training_example
from .geometry import default_recipe

W = Path(__file__).resolve().parent
R = W.parents[1]
DATA_W = W


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(fast_graph=False):
    seed_all(42)
    model, cfg = load_new_model(W/'train.yaml')
    model.graph_model.record_trace = not fast_graph
    model.qwen_vl_interface.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    cfg.memory_training.initialization = 'original Qwen3-VL-4B; new graph and action heads'
    return model.train(), cfg


def read_frame(record, index, bindings=None, allow_provisional=False):
    path = W/record['frames'][index]
    example, context = prepare_training_example(path, allow_provisional=allow_provisional)
    if context['episode_key'] != record['key'] or example['lang'] != record['instruction']:
        raise ValueError('Training example identity mismatch')
    if context['step'] != record.get('offset', 0)+8*index:
        raise ValueError('Training sampling phase mismatch')
    return example, context


def tree_map(value, fn):
    if torch.is_tensor(value): return fn(value)
    if isinstance(value, dict): return {k:tree_map(v,fn) for k,v in value.items()}
    if isinstance(value, list): return [tree_map(v,fn) for v in value]
    if isinstance(value, tuple): return tuple(tree_map(v,fn) for v in value)
    return value

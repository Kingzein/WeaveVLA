import json
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from deployment.model_server.policy_norm_processor import PolicyNormProcessor
from starVLA.model.modules.weavevla.policy.runtime import load_new_model


class OnlinePolicy:
    def __init__(self, weights, stats_file, out):
        config = Path(os.environ["EVAL_CONFIG"])
        self._framework, cfg = load_new_model(config, base_vlm=os.environ.get("EVAL_BASE_VLM"))
        state = torch.load(weights, mmap=True, map_location="cpu", weights_only=True)
        self._framework.load_state_dict(state, strict=True)
        self._framework.eval().requires_grad_(False)
        self._framework.graph_model.record_trace = False
        norm_stats = json.loads(Path(stats_file).read_text())
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        self.norm = PolicyNormProcessor(
            weights, "franka", metadata=(config_dict, norm_stats)
        )
        horizon = int(cfg.framework.action_model.action_horizon)
        if horizon != 8:
            raise ValueError(f"Expected LIBERO horizon 8, got {horizon}")
        self.metadata = dict(action_chunk_size=horizon, available_unnorm_keys=["franka"],
                             training_data_mix=cfg.datasets.vla_data.data_mix)

    def move_to(self, device):
        self._framework.to(device)

    def predict_action(self, examples, unnorm_key=None):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self._framework.predict_action(examples=examples)
        actions = np.asarray(out["normalized_actions"])
        return {"actions": np.stack([self.norm.unapply_actions(row) for row in actions])}

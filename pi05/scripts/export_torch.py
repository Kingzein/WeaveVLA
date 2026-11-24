"""Convert a native Orbax policy checkpoint into Torch inference weights."""
import argparse
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Native WeaveVLA step directory")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.checkpoint.is_dir() or args.output.exists():
        parser.error("Pass a checkpoint directory and a new output file")

    import numpy as np
    import torch
    from flax.traverse_util import flatten_dict
    from openpi.models import gemma
    from openpi.models.model import restore_params
    from openpi.models_pytorch.weight_layout import slice_gemma_state_dict, slice_paligemma_state_dict

    params = restore_params(args.checkpoint, restore_type=np.ndarray, dtype=np.float32)
    if "graph_params" not in params:
        raise ValueError("Expected a WeaveVLA checkpoint containing graph parameters")
    graph = params.pop("graph_params")
    arrays = flatten_dict(params, sep="/")
    config = SimpleNamespace(
        vision_config=SimpleNamespace(hidden_size=1152, num_hidden_layers=27),
        text_config=SimpleNamespace(hidden_size=2048, num_hidden_layers=18,
                                    num_attention_heads=8, head_dim=256),
    )
    vision, expert = slice_paligemma_state_dict(
        {key.removeprefix("PaliGemma/"): value for key, value in arrays.items() if key.startswith("PaliGemma/")},
        config,
    )
    state = dict(vision, **slice_gemma_state_dict(
        expert, gemma.get_config("gemma_300m"), num_expert=1,
        checkpoint_dir="pi05", pi05=True,
    ))
    for name in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"):
        state[f"{name}.weight"] = arrays[f"{name}/kernel"].transpose()
        state[f"{name}.bias"] = arrays[f"{name}/bias"]
    state.update({f"graph_model.{name}": array for name, array in graph.items()})
    state = {name: torch.as_tensor(value) for name, value in state.items()}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.output.parent, prefix="weights-") as file:
        torch.save(state, file)
        file.flush()
        os.link(file.name, args.output)


if __name__ == "__main__":
    main()

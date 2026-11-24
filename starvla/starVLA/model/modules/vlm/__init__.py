def get_vlm_model(config):
    model_path = config.framework.qwenvl.base_vlm
    if "Qwen3-VL" not in model_path:
        raise NotImplementedError(f"WeaveVLA supports Qwen3-VL, got {model_path!r}")

    from .QWen3 import _QWen3_VL_Interface

    return _QWen3_VL_Interface(config)

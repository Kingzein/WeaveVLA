# Environment setup

The repository's [uv.lock](../uv.lock) resolves the dependencies for the source under `src/`. Use `uv sync --locked --extra cuda` on the training system and `uv sync --locked` for CPU checks. Apply the bundled Transformers patch after installing or replacing Transformers:

```bash
python scripts/install_transformers_patch.py
```

You can also use an existing compatible Python environment. The [Python package inventory](../environment/python310.lock.txt) and [CUDA overlay inventory](../environment/jax-cuda-overlay.lock.txt) document the reference dependency versions.

Use dedicated simulators and perception environments. For evaluation, pass their interpreters with `--sim-python`, `--policy-python` and the benchmark-specific Python flags described in the [evaluation guide](../../docs/evaluation.md). Additional DA3 and SAM3 source trees can be exposed with `--da3-source` and `--sam-source`.

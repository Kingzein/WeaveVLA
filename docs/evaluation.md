# Closed-loop evaluation

Use the policy checkpoint and normalization from the matching training profile. Each launcher starts a policy worker and a simulator client, which exchange observations and actions locally. Give each shard its own output directory. See [assets](assets.md) for simulator, detector and language-model requirements.

## LIBERO

The standard suites are `libero_spatial`, `libero_object`, `libero_goal`, and `libero_10`. Each runs 10 tasks with 50 initial states per task by default. Use `--pairs` with a list such as `[[0, 0], [0, 1]]` for a subset.

### StarVLA on LIBERO

Run in `starvla/`. Download the three matching files from [StarVLA LIBERO on Hugging Face](https://huggingface.co/Kensag/WeaveVLA/tree/main/starvla/libero). The launcher reads `config.yaml` beside `model.pt`. Pass `dataset_statistics.json` as `--stats` and the Qwen backbone directory as `--base-vlm`.

```bash
bash examples/LIBERO/eval_files/run_eval.sh \
  --checkpoint "$POLICY_PT" --stats "$LIBERO_STATS" --output runs/eval_spatial \
  --suite libero_spatial --libero-source "$LIBERO_SDK" --runtime-root "$WEAVEVLA_ASSETS" \
  --policy-python "$POLICY_PYTHON" --sim-python "$LIBERO_PYTHON" \
  --depth-weights "$DA3" --cutie-source "$CUTIE_SRC" --cutie-weights "$CUTIE" \
  --sam-python "$SAM_PYTHON" --sam-weights "$SAM3" \
  --yolo-python "$YOLO_PYTHON" --yolo-weights "$LIBERO_YOLO" --base-vlm "$QWEN_WEIGHTS"
```

### π0.5 on LIBERO

Run in `pi05/`. Download `model.pt` and `normalization.json` from [π0.5 LIBERO on Hugging Face](https://huggingface.co/Kensag/WeaveVLA/tree/main/pi05/libero). Pass the matching JSON as `--stats` and the [PaliGemma SentencePiece model used by OpenPI](https://storage.googleapis.com/big_vision/paligemma_tokenizer.model) as `--tokenizer`. The policy predicts 10 actions and executes the first 8 before replanning.

```bash
python examples/libero/main.py \
  --checkpoint "$POLICY_PT" --stats "$LIBERO_STATS" --output runs/eval_spatial \
  --suite libero_spatial --libero-source "$LIBERO_SDK" --runtime-root "$WEAVEVLA_ASSETS" \
  --policy-python "$POLICY_PYTHON" --sim-python "$LIBERO_PYTHON" \
  --depth-weights "$DA3" --cutie-source "$CUTIE_SRC" --cutie-weights "$CUTIE" \
  --sam-python "$SAM_PYTHON" --sam-weights "$SAM3" \
  --yolo-python "$YOLO_PYTHON" --yolo-weights "$LIBERO_YOLO" --tokenizer "$TOKENIZER"
```

For a local JAX checkpoint, use [`export_torch.py`](../pi05/scripts/export_torch.py) and the normalization from its own training run.

Use the standard LIBERO SDK with its BDDL files and EGL setup. The detector must match the vocabularies shipped with [StarVLA](../starvla/deployment/model_server/libero/detector/labels.json) and [π0.5](../pi05/src/weavevla/serving/libero/detector/labels.json). Outputs include results, actions and videos.

## CALVIN D

Both backends use the same 1,000 frozen five-task chains. The environment resets at each chain's start. Each task has a 360-step limit. The official oracle reports success rates for one through five consecutive successes and the mean chain length.

Pass the D validation `.hydra/merged_config.yaml` to `--env-config`. The frozen frontend root must contain `frontend/annotation_frontend.py` and its detector, segmentation, tracking and depth dependencies. A custom calibration must include the camera parameters and robot base pose. Use `--lane` and `--lanes` for sharding, `--limit` for a short run, and `--detector-sha256` to verify the detector weights.

### StarVLA on CALVIN D

Run in `starvla/`. The evaluator reads `geometry_recipe.json` beside the checkpoint unless you pass `--recipe`. Pass the D action statistics used in training to `--stats`.

```bash
bash examples/CALVIN/eval_files/run_eval.sh \
  --checkpoint "$POLICY_PT" --output runs/eval_calvin_d \
  --runtime-root "$WEAVEVLA_ASSETS" --env-root "$CALVIN_ENV" \
  --env-config "$D_VALIDATION_CONFIG" --calvin-conf "$CALVIN_MODELS/conf" \
  --frontend-root "$CALVIN_FRONTEND" --frontend-python "$FRONTEND_PYTHON" \
  --policy-python "$POLICY_PYTHON" --sim-python "$CALVIN_PYTHON" \
  --stats "$CALVIN_D_STATS" --base-vlm "$QWEN_WEIGHTS"
```

### π0.5 on CALVIN D

Run in `pi05/` using the exported D→D checkpoint. This profile predicts eight raw relative actions per chunk.

```bash
python examples/calvin/main.py \
  --checkpoint "$POLICY_PT" --output runs/eval_calvin_d \
  --runtime-root "$WEAVEVLA_ASSETS" --env-root "$CALVIN_ENV" \
  --env-config "$D_VALIDATION_CONFIG" --calvin-conf "$CALVIN_MODELS/conf" \
  --frontend-root "$CALVIN_FRONTEND" --frontend-python "$FRONTEND_PYTHON" \
  --policy-python "$POLICY_PYTHON" --sim-python "$CALVIN_PYTHON" --tokenizer "$TOKENIZER"
```

For either benchmark, `--prompt-root` overrides the default location of the language assets under the runtime root. Use `--da3-source` or `--sam-source` when those components are supplied as source directories.

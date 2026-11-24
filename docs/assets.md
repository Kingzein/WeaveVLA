# Data, weights and runtime assets

## Training

| Asset | StarVLA | π0.5 |
| --- | --- | --- |
| Base model | Qwen3-VL-4B-Instruct directory passed as `--base-vlm` | Native OpenPI `pi05_base/params` directory passed as `--base-params`, with a PaliGemma SentencePiece tokenizer |
| LIBERO | Audited graph cache manifest and associated frames | Audited cache manifest with state/action quantile statistics |
| CALVIN D | Audited D training cache and `franka.action` statistics passed as `--action-stats` | Audited raw-action D training cache |
| Language assets | `--prompt-root` for CALVIN | `--prompt-root` for CALVIN |

Use the manifest created for the corresponding backend and training split. Preserve its relative paths, cache recipe and file identities. The release reads an audited cache, so verify the caches against their source datasets before launching distributed training. Locate the π0.5 base parameters in the OpenPI release at `gs://openpi-assets/checkpoints/pi05_base`.

Place MiniLM in `minilm/` and the spaCy interpreter in `spacy/bin/python` within the prompt root. Override these locations with `WEAVEVLA_MINILM` and `WEAVEVLA_SPACY_PYTHON`. The selected versions are documented in [StarVLA's language configuration](../starvla/starVLA/model/modules/weavevla/language/selected_config.json) and [π0.5's language configuration](../pi05/src/weavevla/language/selected_config.json).

## Evaluation

- **LIBERO** requires the [simulator SDK](https://github.com/Lifelong-Robot-Learning/LIBERO) with BDDL files and standard initial states. The online pipeline uses a [matching YOLOE detector](https://huggingface.co/Kensag/WeaveVLA/tree/main/yolo/libero), [SAM3](https://github.com/facebookresearch/sam3), [Cutie](https://github.com/hkchengrex/Cutie), and [DA3METRIC-LARGE](https://huggingface.co/depth-anything/DA3METRIC-LARGE). The class vocabulary is included with each backend.
- **CALVIN** requires the D validation environment, its merged Hydra configuration, the official task oracle, and the frozen online perception frontend. The source package contains the calibration and frozen sequence list for [StarVLA](../starvla/examples/CALVIN/eval_files/) and [π0.5](../pi05/assets/calvin/).
- **Normalization** must match training. StarVLA LIBERO reads `dataset_statistics.json`. For CALVIN, give StarVLA the D training statistics used for action denormalization. π0.5 LIBERO accepts its training manifest or a JSON file containing `state` and `actions` quantiles under `normalization`. π0.5 CALVIN predicts raw relative actions.
- **Policy checkpoints** come from the selected backend's training output. See [evaluation](evaluation.md) for checkpoint selection and launch arguments.

#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
usage() {
    echo "Usage: NUM_PROCESSES=N PER_DEVICE_BATCH_SIZE=B bash scripts/train.sh {libero|calvin-d} --manifest FILE --base-vlm DIR --output DIR [options]"
}
case "${1:-}" in
    libero)
        module=starVLA.training.train_libero
        config=libero.yaml
        ;;
    calvin-d)
        module=examples.CALVIN.train_files.d.train_stream
        config=calvin_d.yaml
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
shift

export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

python="${STARVLA_PYTHON:-python}"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    exec "$python" -m "$module" --help
fi
if [[ ! "${NUM_PROCESSES:-}" =~ ^[1-9][0-9]*$ || ! "${PER_DEVICE_BATCH_SIZE:-}" =~ ^[1-9][0-9]*$ ]]; then
    usage >&2
    exit 2
fi

exec "$python" -m torch.distributed.run --standalone --nproc_per_node="$NUM_PROCESSES" \
    -m "$module" --base-config "$root/starVLA/config/training/$config" \
    --zero-config "$root/starVLA/config/deepseeds/zero2.json" \
    --microbatch "$PER_DEVICE_BATCH_SIZE" --random-offset --fast-graph "$@"

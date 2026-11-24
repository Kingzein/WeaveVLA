#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec "${WEAVEVLA_PYTHON:-python}" "$root/scripts/evaluate.py" calvin "$@"

#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PI05_PYTHON:-python}" "$root/scripts/train.py" "$@"

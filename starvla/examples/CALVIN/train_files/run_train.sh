#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
if [[ "${1:-}" == d ]]; then
    shift
fi
exec bash "$root/scripts/train.sh" calvin-d "$@"

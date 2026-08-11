#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON:-python3}"

mode="full"
args=()
for arg in "$@"; do
  case "$arg" in
    --smoke) mode="smoke" ;;
    --full) mode="full" ;;
    *) args+=("$arg") ;;
  esac
done

echo "SUPER714-Plus mode=${mode} DATA_DIR=${DATA_DIR:-$PWD/data}"
if [[ "$mode" == "smoke" ]]; then
  exec "$PYTHON_BIN" -u src_super/train_super714_plus.py --smoke "${args[@]}"
fi
exec "$PYTHON_BIN" -u src_super/train_super714_plus.py "${args[@]}"

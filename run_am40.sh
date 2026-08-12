#!/usr/bin/env bash
# AM40 一键复现 / 验收（秒级，不训练）
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON:-python3}"

mode="fuse"
args=()
for arg in "$@"; do
  case "$arg" in
    --verify) mode="verify" ;;
    *) args+=("$arg") ;;
  esac
done

echo "AM40 mode=${mode} DATA_DIR=${DATA_DIR:-$PWD/data}"
if [[ "$mode" == "verify" ]]; then
  exec "$PYTHON_BIN" -u src_super/fuse_am40.py --verify-only "${args[@]}"
fi
exec "$PYTHON_BIN" -u src_super/fuse_am40.py "${args[@]}"

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
  "$PYTHON_BIN" -u src_super/train_super714_plus.py --smoke "${args[@]}"
  "$PYTHON_BIN" -u src_super/fuse_plus_weights.py --suffix _smoke
  exit 0
fi
"$PYTHON_BIN" -u src_super/train_super714_plus.py "${args[@]}"
# 兼容「训练中途升级脚本」的旧跑次：再跑一遍后处理，确保 w62/wbest 落盘
"$PYTHON_BIN" -u src_super/fuse_plus_weights.py

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON:-python3}"

mode="full"
train_args=()
for arg in "$@"; do
  case "$arg" in
    --smoke) mode="smoke" ;;
    --full) mode="full" ;;
    --verify) mode="verify" ;;
    *) train_args+=("$arg") ;;
  esac
done

echo "SUPER714 mode=${mode} DATA_DIR=${DATA_DIR:-$PWD/data}"

if [[ "$mode" == "verify" ]]; then
  exec "$PYTHON_BIN" -u src_super/verify_super714.py "${train_args[@]}"
fi

if [[ "$mode" == "smoke" ]]; then
  "$PYTHON_BIN" -u src_super/train_super714.py --smoke "${train_args[@]}"
else
  "$PYTHON_BIN" -u src_super/train_super714.py "${train_args[@]}"
fi

"$PYTHON_BIN" -u src_super/verify_super714.py "${train_args[@]}"
echo "完成：submissions/submission_super714.csv"

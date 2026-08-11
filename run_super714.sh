#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON:-python3}"

mode="full"
train_args=()
verify_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)
      mode="smoke"
      shift
      ;;
    --full)
      mode="full"
      shift
      ;;
    --verify)
      mode="verify"
      shift
      ;;
    --data-dir)
      [[ $# -ge 2 ]] || { echo "--data-dir 缺少路径" >&2; exit 2; }
      train_args+=("$1" "$2")
      verify_args+=("$1" "$2")
      shift 2
      ;;
    --data-dir=*)
      train_args+=("$1")
      verify_args+=("$1")
      shift
      ;;
    *)
      train_args+=("$1")
      shift
      ;;
  esac
done

echo "SUPER714 mode=${mode} DATA_DIR=${DATA_DIR:-$PWD/data}"

if [[ "$mode" == "verify" ]]; then
  exec "$PYTHON_BIN" -u src_super/verify_super714.py "${verify_args[@]}"
fi

if [[ "$mode" == "smoke" ]]; then
  "$PYTHON_BIN" -u src_super/train_super714.py --smoke "${train_args[@]}"
else
  "$PYTHON_BIN" -u src_super/train_super714.py "${train_args[@]}"
fi

"$PYTHON_BIN" -u src_super/verify_super714.py "${verify_args[@]}"
echo "完成：submissions/submission_super714.csv"

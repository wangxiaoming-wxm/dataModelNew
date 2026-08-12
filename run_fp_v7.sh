#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON:-python3}"
if [[ "${1:-}" == "--verify" ]]; then
  exec "$PYTHON_BIN" -u src_super/fuse_fp_v7.py --verify-only --reuse-caches
fi
exec "$PYTHON_BIN" -u src_super/fuse_fp_v7.py "$@"

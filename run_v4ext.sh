#!/usr/bin/env bash
# V4-ext: fuse + independent audit (seconds; uses committed arm_*.npz).
set -euo pipefail
cd "$(dirname "$0")"

python3 -m pip install -q -r requirements.txt

mkdir -p artifacts/v4_ext
PYTHONPATH=src2:src3:src4 python3 src4/fuse4.py \
  --dir artifacts/v4 \
  --submission submissions/submission_v4ext.csv \
  --report artifacts/v4_ext/fusion_report_v4ext.json \
  --seeds 20

PYTHONPATH=src2:src3:src4 python3 src4/audit_v4.py \
  --dir artifacts/v4 \
  --submission submissions/submission_v4ext.csv \
  --target 0.707 \
  --out artifacts/v4_ext/audit.json

echo
echo "V4-ext done. Submission: submissions/submission_v4ext.csv"
echo "See docs/DELIVERY.md"

#!/usr/bin/env bash
# 完整复现 best v1（线上 0.71453）
# 用法:
#   bash reproduce.sh           # 全量 8seed×5fold×3bag×2臂
#   bash reproduce.sh --smoke   # 冒烟
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
PY="${PYTHON:-python3}"

echo "================================================"
echo "  best v1 / real714  目标: max2 OOF ≈ 0.70128"
echo "  DATA_DIR=${DATA_DIR:-"(auto)"}"
echo "================================================"

"$PY" -u src/explore_best.py "$@"

echo ""
echo "验证产物..."
"$PY" verify_artifacts.py
echo "完成。提交: submissions/submission_best.csv"

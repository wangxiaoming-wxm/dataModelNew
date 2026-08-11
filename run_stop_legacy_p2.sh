#!/usr/bin/env bash
set -euo pipefail
cd /workspace
echo "[stop] monitoring P1 ..."
while true; do
  n=$(ls artifacts/beat_max3/train/part_ord_noxb_new16_s*.npz 2>/dev/null | wc -l)
  echo "[stop] parts=$n/16 $(date -Is)"
  if [[ "$n" -ge 16 ]]; then
    python3 - <<'PY'
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
parts=sorted(Path('artifacts/beat_max3/train').glob('part_ord_noxb_new16_s*.npz'))[:16]
y=pd.read_csv('data/train.csv')['label'].astype(int).values
o=np.mean([np.load(p)['oof'] for p in parts],0)
t=np.mean([np.load(p)['test_pred'] for p in parts],0)
np.savez('artifacts/beat_max3/ord_noxb_new16.npz', oof=o, test_pred=t)
print('bagged', float(roc_auc_score(y,o)))
PY
    break
  fi
  if [[ -f artifacts/beat_max3/ord_noxb_new16.npz ]]; then
    break
  fi
  sleep 60
done
while pgrep -f 'train_ord_noxb.py --tag ord_noxb_new16' >/dev/null; do
  echo "[stop] wait train exit"; sleep 20
done
pkill -f 'bash run_beat_max3.sh' 2>/dev/null || true
pkill -f 'run_beat_max3_hq' 2>/dev/null || true
echo LEGACY_P2_STOPPED

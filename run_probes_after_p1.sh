#!/usr/bin/env bash
set -euo pipefail
cd /workspace
export PYTHONPATH=/workspace/src PYTHONUNBUFFERED=1
echo "[probes] wait for P1 done"
while true; do
  if grep -q LEGACY_P2_STOPPED logs/beat_max3/stop_p2.log 2>/dev/null; then break; fi
  n=$(ls artifacts/beat_max3/train/part_ord_noxb_new16_s*.npz 2>/dev/null | wc -l)
  if [[ "$n" -ge 16 ]] && ! pgrep -f 'train_ord_noxb.py --tag ord_noxb_new16' >/dev/null; then
    python3 - <<'PY'
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
parts=sorted(Path('artifacts/beat_max3/train').glob('part_ord_noxb_new16_s*.npz'))[:16]
y=pd.read_csv('data/train.csv')['label'].astype(int).values
o=np.mean([np.load(p)['oof'] for p in parts],0); t=np.mean([np.load(p)['test_pred'] for p in parts],0)
np.savez('artifacts/beat_max3/ord_noxb_new16.npz', oof=o, test_pred=t)
print('bagged', float(roc_auc_score(y,o)))
PY
    pkill -f 'bash run_beat_max3.sh' 2>/dev/null || true
    break
  fi
  echo "[probes] waiting parts=$n $(date -Is)"; sleep 60
done
echo "[probes] START"
python3 src_beat/run_strategy_probes.py --exps exp1 exp2 exp3 --seeds 2900 2901 2902 2903 \
  2>&1 | tee logs/beat_max3/strategy_probes.log
echo PROBES_DONE

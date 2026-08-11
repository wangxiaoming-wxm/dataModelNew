#!/usr/bin/env bash
# When best_v1 finishes (or interim seeds land), rebuild ship and optionally resume P1/plus.
set -euo pipefail
cd /workspace
export PYTHONPATH=/workspace/src PYTHONUNBUFFERED=1
mkdir -p logs/beat_max3
last=-1
while true; do
  n=$(ls artifacts/beat_max3/best_v1/part_main_s*.npz 2>/dev/null | wc -l)
  a=$(ls artifacts/beat_max3/best_v1/part_alt_s*.npz 2>/dev/null | wc -l)
  echo "[bestwatch] main=$n alt=$a $(date -Iseconds)"
  if [[ -f artifacts/beat_max3/report_best_v1.json ]]; then
    python3 src_beat/build_ship_candidates.py | tee -a logs/beat_max3/best_v1_ship.log
    # if best_v1 crowned, copy as beat_max3 alias already handled in builder
    echo "[bestwatch] report present; refresh done"
  elif [[ "$n" -ge 2 && "$a" -ge 2 && "$n" -ne "$last" ]]; then
    # interim: pool available seeds only by re-running trainer resume path
    python3 - <<'PY' || true
from pathlib import Path
import numpy as np, pandas as pd, json
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
ART=Path('artifacts/beat_max3/best_v1')
y=pd.read_csv('data/train.csv')['label'].astype(int).values
tid=pd.read_csv('data/test.csv')['id']
def pool(arm):
  parts=sorted(ART.glob(f'part_{arm}_s*.npz'))
  if len(parts)<2: return None
  o=np.mean([np.load(p)['oof_rank'] for p in parts],0)
  t=np.mean([np.load(p)['test_rank'] for p in parts],0)
  return o,t,len(parts)
m=pool('main'); a=pool('alt')
if m and a:
  fo=np.maximum(m[0],a[0]); ft=np.maximum(m[1],a[1])
  np.savez('artifacts/beat_max3/best_v1.npz', oof=fo, test_pred=ft, main_oof=m[0], alt_oof=a[0], main_te=m[1], alt_te=a[1], interim=True, n_main=m[2], n_alt=a[2])
  pd.DataFrame({'id':tid,'label':np.clip(ft,0.001,0.999)}).to_csv('submissions/submission_best_v1_interim.csv',index=False)
  print('interim fuse', roc_auc_score(y,fo), 'n', m[2], a[2])
PY
    python3 src_beat/build_ship_candidates.py | tee -a logs/beat_max3/best_v1_ship.log || true
    last=$n
  fi
  if ! pgrep -f 'train_best_v1.py' >/dev/null && [[ -f artifacts/beat_max3/report_best_v1.json ]]; then
    echo "[bestwatch] training done"
    # resume P1 remaining seeds if incomplete
    n16=$(ls artifacts/beat_max3/train/part_ord_noxb_new16_s*.npz 2>/dev/null | wc -l)
    if [[ "$n16" -lt 16 ]] && ! pgrep -f 'train_ord_noxb.py' >/dev/null; then
      echo "[bestwatch] resume P1"
      nohup python3 src_beat/train_ord_noxb.py --tag ord_noxb_new16 --seeds 2100 2101 2102 2103 2104 2105 2106 2107 2108 2109 2110 2111 2112 2113 2114 2115 --depth 7 --iterations 1200 --lr 0.03 \
        > logs/beat_max3/ord_noxb_new16_resume.log 2>&1 &
    fi
    break
  fi
  sleep 120
done

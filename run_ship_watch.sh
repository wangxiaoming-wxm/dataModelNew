#!/usr/bin/env bash
# Continuously refresh ship when new P1 seeds land; queue plus bag after P1≥12.
set -euo pipefail
cd /workspace
export PYTHONPATH=/workspace/src PYTHONUNBUFFERED=1
mkdir -p logs/beat_max3
last_n=-1
while true; do
  n=$(ls artifacts/beat_max3/train/part_ord_noxb_new16_s*.npz 2>/dev/null | wc -l)
  ts=$(date -Iseconds)
  echo "[watch] parts=$n $ts"
  if [[ "$n" -ne "$last_n" && "$n" -ge 4 ]]; then
    python3 src_beat/build_ship_candidates.py | tee -a logs/beat_max3/ship_watch.log
    last_n=$n
  fi
  # When P1 mostly done, start plus_new if not running
  if [[ "$n" -ge 12 ]]; then
    if ! pgrep -f 'train_plus.py' >/dev/null; then
      if [[ ! -f artifacts/beat_max3/train/part_plus_new8_s2600.npz ]]; then
        echo "[watch] launching plus_new8"
        nohup python3 src_beat/train_plus.py --tag plus_new8 --seeds 2600 2601 2602 2603 2604 2605 2606 2607 \
          > logs/beat_max3/plus_new8.log 2>&1 &
      fi
    fi
  fi
  # Stop when P1 done AND plus done (or plus already present) AND probes done
  p1_alive=$(pgrep -f 'train_ord_noxb.py' >/dev/null && echo 1 || echo 0)
  probe_alive=$(pgrep -f 'run_strategy_probes.py' >/dev/null && echo 1 || echo 0)
  if [[ "$n" -ge 16 && "$p1_alive" -eq 0 && "$probe_alive" -eq 0 ]]; then
    python3 src_beat/build_ship_candidates.py | tee -a logs/beat_max3/ship_watch.log
    echo "[watch] final refresh done"
    break
  fi
  sleep 90
done

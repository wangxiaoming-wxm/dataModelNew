#!/usr/bin/env bash
# Watcher: when P1 finishes, stop the OLD (lighter) P2–P3 loop and start HQ continue.
set -euo pipefail
cd /workspace
export PYTHONUNBUFFERED=1

echo "[watch] monitoring P1 ..."
while true; do
  n=$(ls artifacts/beat_max3/train/part_ord_noxb_new16_s*.npz 2>/dev/null | wc -l)
  echo "[watch] new16_parts=$n/16 $(date -Is)"
  if [[ "$n" -ge 16 ]] || [[ -f artifacts/beat_max3/ord_noxb_new16.npz ]]; then
    echo "[watch] P1 complete — stopping legacy run_beat_max3.sh P2+ if still running"
    # kill only the bash driver (legacy), not random pythons mid-seed if still on P1
    if [[ -f artifacts/beat_max3/ord_noxb_new16.npz ]] || [[ "$n" -ge 16 ]]; then
      # if train_ord for new16 still running, wait for it to exit cleanly
      while pgrep -f 'train_ord_noxb.py --tag ord_noxb_new16' >/dev/null; do
        echo "[watch] waiting train_ord_noxb_new16 exit"; sleep 30
      done
      # kill legacy driver if it is about to/already starting weak P2
      pkill -f 'bash run_beat_max3.sh' 2>/dev/null || true
      sleep 2
      break
    fi
  fi
  # interim refresh every cycle
  PYTHONPATH=/workspace/src python3 src_beat/refresh_stage_best.py || true
  sleep 120
done

echo "[watch] launching HQ continue"
bash run_beat_max3_hq.sh 2>&1 | tee -a logs/beat_max3/hq_loop.log

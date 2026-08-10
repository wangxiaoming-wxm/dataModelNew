#!/usr/bin/env bash
set -u
cd /workspace
export PYTHONPATH=src4:src3:src2:src
OUT=artifacts/v4_alt_r24_parts
LOG=logs/v4/alt_r24
mkdir -p "$OUT" "$LOG"
NWORKERS=4
echo "alt_r24 start $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG/driver.log"
pids=()
for s in $(seq 28000 28007); do
  (
    echo "START alt_d6l6_$s $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG/driver.log"
    python3 src4/run_world.py --world alt --preset d6l6 --seed "$s" --folds 10 --threads 1 --out "$OUT" \
      > "$LOG/alt_d6l6_$s.log" 2>&1
    echo "DONE alt_d6l6_$s status=$? $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG/driver.log"
  ) &
  pids+=($!)
  while [ $(jobs -rp | wc -l) -ge $NWORKERS ]; do sleep 5; done
done
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
echo "ALL_DONE fail=$fail $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG/driver.log"
exit $fail

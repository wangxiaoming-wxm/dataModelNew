#!/usr/bin/env bash
set -u
cd /workspace
export PYTHONPATH=src4:src3:src2:src
OUT=artifacts/v4_w12_parts
LOG=logs/v4/w12
mkdir -p "$OUT" "$LOG"
NWORKERS=4
echo "w12 start $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG/driver.log"
pids=()
for s in $(seq 31000 31007); do
  (
    echo "START w12_d6_$s $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG/driver.log"
    python3 src4/run_world.py --world w12 --preset d6 --seed "$s" --folds 10 --threads 1 --out "$OUT" \
      > "$LOG/w12_d6_$s.log" 2>&1
    echo "DONE w12_d6_$s status=$? $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG/driver.log"
  ) &
  pids+=($!)
  while [ $(jobs -rp | wc -l) -ge $NWORKERS ]; do sleep 5; done
done
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
echo "ALL_DONE fail=$fail $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG/driver.log"
exit $fail

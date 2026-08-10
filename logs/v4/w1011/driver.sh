#!/usr/bin/env bash
set -u
cd /workspace
export PYTHONPATH=src4:src3:src2:src
OUT=artifacts/v4_w1011_parts
LOG=logs/v4/w1011
mkdir -p "$OUT" "$LOG"
NWORKERS=4

jobs=()
for s in $(seq 29000 29007); do
  jobs+=("w10 d6l6 $s")
done
for s in $(seq 29100 29107); do
  jobs+=("w11 d6 $s")
done

echo "total jobs ${#jobs[@]} $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG/driver.log"

run_one() {
  local world=$1 preset=$2 seed=$3
  local tag="${world}_${preset}_${seed}"
  echo "START $tag $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG/driver.log"
  python3 src4/run_world.py \
    --world "$world" --preset "$preset" --seed "$seed" \
    --folds 10 --threads 1 --out "$OUT" \
    > "$LOG/${tag}.log" 2>&1
  local st=$?
  echo "DONE $tag status=$st $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG/driver.log"
  return $st
}

export -f run_one
export OUT LOG PYTHONPATH

# simple pool
i=0
pids=()
tags=()
fail=0
for spec in "${jobs[@]}"; do
  set -- $spec
  world=$1; preset=$2; seed=$3
  run_one "$world" "$preset" "$seed" &
  pids+=($!)
  tags+=("${world}_${preset}_${seed}")
  # throttle
  while [ $(jobs -rp | wc -l) -ge $NWORKERS ]; do
    sleep 5
  done
done
for idx in "${!pids[@]}"; do
  if ! wait "${pids[$idx]}"; then
    echo "FAIL ${tags[$idx]}" | tee -a "$LOG/driver.log"
    fail=1
  fi
done
echo "ALL_DONE fail=$fail $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG/driver.log"
exit $fail

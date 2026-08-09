#!/usr/bin/env bash
# Screen CatBoost configurations four-at-a-time on identical folds.
# One thread per worker beats one config at a time with four threads on this box.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src2:src3
mkdir -p artifacts/screen logs/screen

CONFIGS=${CONFIGS:-"base ordered rsm08 lr02 ctr_rich mvs newton2 l2_20 depth4 ordered_d6 rsm06 langevin ctr_prior bern08 l2_4 rstr2 lr015 ctr_bins50 ordered_lr02"}
SEEDS=${SEEDS:-"900 901"}
VIEW=${VIEW:-main}
JOBS=${JOBS:-4}

printf '%s\n' $CONFIGS | xargs -P "$JOBS" -I{} sh -c \
  "python3 src3/screen.py --config {} --seeds $SEEDS --view $VIEW --threads 1 \
     > logs/screen/${VIEW}_{}.log 2>&1; echo \"done {} rc=\$?\""
echo "ALL SCREENING DONE"

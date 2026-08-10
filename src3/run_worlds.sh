#!/usr/bin/env bash
# Train the new encoding worlds, one seed per single-threaded process, four at
# a time.  Waits for any screening workers to drain first so the box is not
# oversubscribed.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src2:src3
mkdir -p artifacts/worlds logs/worlds

while pgrep -f "screen.py --config" >/dev/null 2>&1; do
  echo "waiting for screening workers to finish ... $(date -u +%H:%M:%S)"
  sleep 60
done

WORLDS=${WORLDS:-"w4 w5"}
SEEDS=${SEEDS:-"21000 21001 21002 21003 21004 21005 21006 21007"}
PRESET=${PRESET:-d6l6}
FOLDS=${FOLDS:-5}
JOBS=${JOBS:-4}

JOBLIST=$(mktemp)
for w in $WORLDS; do
  for s in $SEEDS; do
    echo "$w $s" >> "$JOBLIST"
  done
done

xargs -a "$JOBLIST" -P "$JOBS" -L 1 sh -c \
  'python3 src3/run_world.py --world $0 --seed $1 --preset '"$PRESET"' --folds '"$FOLDS"' \
     --threads 1 > logs/worlds/'"$PRESET"'_f'"$FOLDS"'_$0_$1.log 2>&1; \
   echo "done $0 seed $1 rc=$?"'
rm -f "$JOBLIST"
echo "ALL WORLD TRAINING DONE"

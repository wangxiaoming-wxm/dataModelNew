#!/usr/bin/env bash
# Reproduce submission_v3.csv from scratch.
#
# What changes vs run_all.sh (V2):
#   * the three arms that enter views_max are trained under 10-fold CV
#     (8 seeds each) instead of 5-fold (12 seeds);
#   * fusion is src3/fuse2.py, which reports the nested selection averaged
#     over 20 block seeds;
#   * src3/audit.py re-checks honesty independently of src2/verify.py.
#
# Wall time on 4 cores, one thread per worker, four workers: ~8-10 hours.
# Nothing here needs a GPU.  Intermediate parts land in artifacts/worlds10/
# so a crash only costs the unfinished seeds.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=src2:src3

log() { printf '\n=== %s ===\n' "$1"; }
mkdir -p artifacts/worlds10 artifacts/v3_f10 artifacts/v3 artifacts/audit logs/worlds10

SEEDS=(21000 21001 21002 21003 21004 21005 21006 21007)
JOBS="${JOBS:-4}"

run_batch() {
  # $1=world  $2=preset
  local world="$1" preset="$2"
  local joblist
  joblist="$(mktemp)"
  for s in "${SEEDS[@]}"; do
    # skip seeds that already finished (resume-friendly)
    if [[ -f "artifacts/worlds10/part_${world}_${preset}_s${s}_f10.npz" ]]; then
      echo "skip ${world}/${preset}/s${s} (exists)"
      continue
    fi
    echo "${world} ${preset} ${s}" >> "$joblist"
  done
  if [[ ! -s "$joblist" ]]; then
    rm -f "$joblist"
    return 0
  fi
  xargs -a "$joblist" -P "$JOBS" -L 1 sh -c \
    'python3 src3/run_world.py --world "$0" --preset "$1" --seed "$2" --folds 10 \
       --threads 1 --out artifacts/worlds10 \
       > "logs/worlds10/${0}_${1}_${2}.log" 2>&1; \
     echo "done $0 $1 $2 rc=$?"'
  rm -f "$joblist"
}

log "10-fold main/d5 (8 seeds)"
run_batch main d5

log "10-fold main/d6 (8 seeds)"
run_batch main d6

log "10-fold alt/d6l6 (8 seeds)"
run_batch alt d6l6

log "merge per-seed parts into bagged arms"
python3 src3/merge_seeds.py --world main --preset d5   --folds 10 \
    --dir artifacts/worlds10 --name cat_d5 --out artifacts/v3_f10
python3 src3/merge_seeds.py --world main --preset d6   --folds 10 \
    --dir artifacts/worlds10 --name cat_d6 --out artifacts/v3_f10
python3 src3/merge_seeds.py --world alt  --preset d6l6 --folds 10 \
    --dir artifacts/worlds10 --name cat_alt --out artifacts/v3_f10

log "assemble artifacts/v3 (10-fold winners + V2 leftover arms for the rule set)"
# Prefer a previously built V2 tree; if missing, the fusion still runs on the
# three views_max members alone.
if [[ ! -d artifacts/v2 ]]; then
  echo "WARN: artifacts/v2 missing; fusion will only see the three 10-fold arms." >&2
  echo "      Run bash run_all.sh once if you need the full pre-registered rule set." >&2
fi
cp artifacts/v3_f10/arm_cat_d5.npz artifacts/v3/arm_cat_d5.npz
cp artifacts/v3_f10/arm_cat_d6.npz artifacts/v3/arm_cat_d6.npz
cp artifacts/v3_f10/arm_cat_alt.npz artifacts/v3/arm_cat_alt.npz
for a in cat_alt2 gap lgb_te glm; do
  if [[ -f "artifacts/v2/arm_${a}.npz" ]]; then
    cp "artifacts/v2/arm_${a}.npz" "artifacts/v3/arm_${a}.npz"
  fi
done

log "fuse (pre-registered rules, nested over 20 block seeds) → submission_v3.csv"
python3 src3/fuse2.py --dir artifacts/v3 \
    --submission submissions/submission_v3.csv \
    --report artifacts/v3/fusion_report_v3.json

log "independent supervisor"
python3 src3/audit.py --dir artifacts/v3 \
    --submission submissions/submission_v3.csv \
    --out artifacts/audit/audit_v3.json

log "done — submissions/submission_v3.csv"
python3 - <<'PY'
import json
r = json.load(open("artifacts/v3/fusion_report_v3.json"))
print(f"nested_oof_mean = {r['nested_oof_mean']:.5f}")
print(f"submitted_rule  = {r['submitted_rule']}")
print(f"views_max full  = {r['rule_full_oof_auc'].get('views_max')}")
PY

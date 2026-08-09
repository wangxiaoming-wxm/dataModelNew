#!/usr/bin/env bash
# Full reproduction: ~3.5 h on 4 cores.
#
# Runs are split into segments of four seeds so each segment can use a different
# jitter stream family (--stream-base) and so a crashed run only costs one
# segment.  merge_runs.py pools segments by equal-weight averaging, which is
# exact as long as every segment contributes the same number of models.
#
# Nothing here needs a GPU and nothing writes outside the repository.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=src2

log() { printf '\n=== %s ===\n' "$1"; }
mkdir -p logs/training

log "main encoding world (12 seeds, in three segments)"
python3 src2/run_oof.py --seeds 20260 20261 20262 20263 --stream-base 0 \
    --out artifacts/v2a 2>&1 | tee logs/training/oof_run.log
python3 src2/run_oof.py --seeds 20264 20265 20266 20267 --stream-base 10 \
    --out artifacts/v2b 2>&1 | tee logs/training/oof_run_b.log
python3 src2/run_oof.py --arms cat_d5 cat_d6 --seeds 20268 20269 20270 20271 --stream-base 20 \
    --out artifacts/v2c 2>&1 | tee logs/training/oof_run_c.log

log "second encoding world (12 seeds)"
python3 src2/run_oof.py --view alt --arms cat_alt --seeds 20280 20281 20282 20283 --stream-base 0 \
    --out artifacts/v2alt 2>&1 | tee logs/training/oof_alt.log
python3 src2/run_oof.py --view alt --arms cat_alt --seeds 20284 20285 20286 20287 --stream-base 10 \
    --out artifacts/v2alt2 2>&1 | tee logs/training/oof_alt2.log
python3 src2/run_oof.py --view alt --arms cat_alt --seeds 20288 20289 20294 20295 --stream-base 20 \
    --out artifacts/v2alt3 2>&1 | tee logs/training/oof_alt3.log

log "third encoding world (8 seeds; not in the winning rule, kept for future work)"
python3 src2/run_oof.py --view alt2 --arms cat_alt2 --seeds 20300 20301 20302 20303 --stream-base 0 \
    --out artifacts/v2alt2w 2>&1 | tee logs/training/oof_alt2w.log
python3 src2/run_oof.py --view alt2 --arms cat_alt2 --seeds 20304 20305 20306 20307 --stream-base 10 \
    --out artifacts/v2alt2w_b 2>&1 | tee logs/training/oof_alt2w_b.log

log "previous solution's gap view, re-run under this branch's honest protocol"
python3 src2/run_gap_arm.py --seeds 20290 20291 20292 20293 \
    --out artifacts/v2gap 2>&1 | tee logs/training/gap_run.log

log "merge, fuse, submit"
# two passes: the first keeps the 8-seed lgb_te/glm (v2c only ran the CatBoost
# arms), the second overwrites cat_d5/cat_d6 with their 12-seed bags
python3 src2/merge_runs.py --inputs artifacts/v2a artifacts/v2b --out artifacts/v2main
python3 src2/merge_runs.py --inputs artifacts/v2a artifacts/v2b artifacts/v2c --out artifacts/v2main
python3 src2/merge_runs.py --inputs artifacts/v2alt artifacts/v2alt2 artifacts/v2alt3 --out artifacts/v2altmerged
python3 src2/merge_runs.py --inputs artifacts/v2alt2w artifacts/v2alt2w_b --out artifacts/v2alt2merged
python3 src2/collect.py --out artifacts/v2
python3 src2/fuse.py --dir artifacts/v2 --submission submissions/submission_v2.csv

log "honesty checks (data hashes, shuffled-label control, submission format)"
python3 src2/verify.py 2>&1 | tee logs/training/verify.log

log "done - submissions/submission_v2.csv"

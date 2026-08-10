# V4 independent supervision（停止时定稿）

Branch: `cursor/honest-auc-v4-145a`

Scope: adversarial audit of delivered V4 artefacts under `artifacts/v4`.
No V3 files were edited. Train labels + OOF arms only; no test labels; no LB feedback.
Per user request, iteration stopped at the current best honest result; incomplete
w12 / f30 runs are **not** part of the endorsed delivery.

## Commands run

```bash
PYTHONPATH=src2:src3:src4 python3 src4/fuse4.py \
  --dir artifacts/v4 \
  --submission submissions/submission_v4.csv \
  --report artifacts/v4/fusion_report_v4.json

PYTHONPATH=src2:src3:src4 python3 src4/audit_v4.py \
  --dir artifacts/v4 \
  --submission submissions/submission_v4.csv \
  --target 0.707 \
  --out artifacts/audit_v4/audit.json
```

## Gate table

| Gate | Result |
|---|---|
| Data integrity | PASS |
| Protocol scan (`src4`) | PASS |
| Arms present / sanity | PASS |
| Nested selection | PASS |
| Permutation no-signal | PASS |
| Submission format | PASS |
| Target `>= 0.707` | **FAIL** |

`honesty_passed = true`, `target_reached = false`.

## Endorsed headline

```text
nested_oof_mean = 0.7030285030340446
nested_oof_sd   = 0.0001551649037618552
submitted_rule  = views_max_10_20_r16_r16b
best_full_oof   = 0.7033678813195666
honesty_passed  = true
target_reached  = false
gap_to_0.707    = 0.0039714969659554
vs_V3_nested    = +0.0017837813644681
```

Matches `artifacts/v4/fusion_report_v4.json` and `submissions/submission_v4.csv`.

## Opinion

I endorse the above as the honest local AUC for this stop point.
I do **not** endorse any claim that the 0.707 target was reached.
Further incomplete experiments (w12, f30) must not be mixed into the headline
without a fresh fuse + audit cycle.

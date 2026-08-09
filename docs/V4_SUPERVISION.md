# V4 independent supervision

Branch: `cursor/honest-auc-v4-145a`

Scope: adversarial audit of the current V4 artefacts under `artifacts/v4`.
No V3 files were edited. The audit uses train labels and OOF arms only; it does
not read test labels or leaderboard feedback.

## Commands run

```bash
PYTHONPATH=src2:src3:src4 python3 src4/audit_v4.py \
  --dir artifacts/v4 \
  --submission submissions/submission_v4.csv \
  --target 0.707 \
  --out artifacts/audit_v4/audit.json
```

Additional evidence was recomputed into:

```text
artifacts/audit_v4/evidence_v4.json
```

## Gate table

| Gate | Result | Evidence |
|---|---:|---|
| Data integrity | PASS | `train.csv`, `test.csv`, and `submit_sample.csv` SHA256 match expected hashes. |
| Protocol scan | PASS | `src4/**/*.py` token scan found no `eval_set`, early stopping, `use_best_model`, `od_wait`, `od_type`, or test-label reads. |
| Arms present | PASS | Current arms: `cat_alt`, `cat_alt2`, `cat_d5`, `cat_d6`, `cat_w5`, `gap`. |
| Arm sanity | PASS | Arm OOF AUCs are plausible: 0.69044 to 0.69818. No arm has suspicious near-perfect AUC. |
| Nested selection | PASS | Pre-registered nested rule selection is active. |
| Permutation no-signal | PASS | Audit permutation nested mean = 0.5116308568; direct permuted-label `views_max` AUC = 0.4995007281. |
| Submission format | PASS | `submissions/submission_v4.csv` has aligned ids, 6398 rows, finite labels in [0, 1]. |
| Target reached (`>= 0.707`) | FAIL | Current honest nested OOF mean = 0.7015141215. |

## Current headline

The only honest headline I can endorse right now is:

```text
nested_oof_mean = 0.7015141215
nested_oof_sd   = 0.0
honesty_passed  = true
target_reached  = false
```

This is below the required V4 gate by:

```text
0.707 - 0.7015141215 = 0.0054858785
```

Honesty passes; the target does not. This must not be rebranded as a V4 target
success.

## Bayes ceiling recomputation

Using the current V4/V3-copied strongest score
`views_max(rank(cat_d5), rank(cat_d6), rank(cat_alt))`, cross-fitted isotonic
calibration gives:

| Quantity | Value |
|---|---:|
| Score OOF AUC | 0.7015141215 |
| Base rate | 0.1002009377 |
| Cross-fitted calibrated p(x) mean | 0.1002139484 |
| Implied Bayes AUC of this risk function | 0.7056591125 |
| Same-score redrawn-label AUC mean | 0.7049130208 |
| Same-score redrawn-label AUC sd | 0.0070744398 |
| Redrawn-label 95% interval | [0.6906487736, 0.7190831892] |

Interpretation: the calibrated current risk function is already close to the
0.707 gate but estimates a ceiling below it by about 0.00134. This is evidence
against the target, not a mathematical proof that no new label-free signal can
exist.

## Neighbourhood concordance

Nearest-neighbour concordance was recomputed in the established signal space
(`days`, source-normalised condition, log-ratio, age, region, source, grade, and
binary flags).

| 1-NN distance percentile | n | P(same label) | Chance | Lift |
|---|---:|---:|---:|---:|
| 0-5 | 748 | 0.8302139037 | 0.7644628099 | +0.0657510938 |
| 5-25 | 2987 | 0.8101774356 | 0.7885151021 | +0.0216623335 |
| 25-50 | 3733 | 0.8432895794 | 0.8348701631 | +0.0084194163 |
| 50-75 | 3733 | 0.8207875703 | 0.8184161089 | +0.0023714614 |
| 75-100 | 3733 | 0.8435574605 | 0.8432526229 | +0.0003048376 |

The in-sample optimistic 10-NN label-rate AUC is only 0.6019177038. If labels
were close to deterministic functions of the available signal space, the nearest
rows would share labels far more strongly than this.

## Selection optimism

I recomputed selection over the usable pre-registered V4/fuse4 rule set:

| Rule | Full OOF AUC |
|---|---:|
| `views_max` | 0.7015141215 |
| `four_max_w5` | 0.7004805480 |
| `views_half` | 0.7002505416 |
| `views_mean` | 0.6999299507 |
| `cat_pair_max` | 0.6979007939 |
| `cat_d5_only` | 0.6977134052 |

Nested 20-seed mean over the same rules is 0.7015141215, with `views_max`
picked in all 100 inner selections. Measured optimism
(`best_full_oof_auc - nested_20seed_mean`) is 0.0 for the current artefacts.

## Protocol scan notes

Manual and token-based scans of `src4/**/*.py` found no hard-rule violation:

- `src4/run_world.py` fits CatBoost with fixed iterations and no `eval_set`.
- `src4/worlds_v4.py` constructs label-free edges/features; train+test use is
  unsupervised and does not include test labels.
- `src4/fuse4.py` performs nested selection over explicit rule dictionaries.
- `src4/audit_v4.py` now serializes NumPy scalar booleans/floats/ints in all
  report paths and records submission-gate diagnostics.

Negative-control thought experiment, not implanted in live code: a line like
`m.fit(..., eval_set=[...], use_best_model=True)` or
`test["label"]` would be detected by the current token scan. No such violating
code is present.

## Risks and veto

1. Current V4 artefacts are essentially the V3-copied arm family; no new strong
   W6/W7 arm is present in `artifacts/v4`.
2. The 0.707 target is above the calibrated-risk Bayes estimate from the current
   strongest score.
3. Neighbourhood concordance shows substantial irreducible label noise.
4. The current selection rule is clean, but it also has no hidden gain: `views_max`
   wins every nested split and remains at 0.7015141215.

Verdict: honesty PASS, target FAIL. I veto any claim that the current V4 push has
reached the required honest nested OOF >= 0.707.

## Monitoring poll

After the audit, I polled `artifacts/v4/`, `artifacts/v4_probe/`, and `logs/v4/`.
No new final `artifacts/v4/arm_*.npz` or replacement submission appeared. The
available fast probe summaries were:

| Probe | OOF AUC |
|---|---:|
| `main_fast_logloss_s22000_f5` | 0.6889156156 |
| `alt_fast_logloss_s22000_f5` | 0.6897393098 |
| `w6_fast_logloss_s22000_f5` | 0.6792893301 |
| `w7_fast_logloss_s22000_f5` | 0.6823615891 |

These probe outputs are screening evidence only, not headline arms. They do not
support changing the target verdict.


# V4 independent supervision

Branch: `cursor/honest-auc-v4-145a`

Scope: adversarial audit of current V4 artefacts under `artifacts/v4`.
No V3 files were edited. Train labels + OOF arms only; no test labels; no LB feedback.

## Commands run

```bash
PYTHONPATH=src2:src3:src4 python3 src4/audit_v4.py \
  --dir artifacts/v4 \
  --submission submissions/submission_v4.csv \
  --target 0.707 \
  --out artifacts/audit_v4/audit.json
```

Additional recomputation (fuse4 rule set, Bayes, arm risks):

```text
artifacts/audit_v4/evidence_v4.json
```

## Gate table

| Gate | Result | Evidence |
|---|---:|---|
| Data integrity | PASS | `train.csv` / `test.csv` / `submit_sample.csv` SHA256 match expected. |
| Protocol scan | PASS | `src4/**/*.py` token scan: no `eval_set`, `use_best_model`, `od_wait`, `od_type`, `early_stopping_rounds`, or `test["label"]`. |
| Arms present | PASS | 18 arms including new `*_f20`, `*_r16`, `*_s16`, `*_sf85`, `cat_alt_d5`. |
| Arm sanity | PASS | Arm OOF AUC in `[0.69044, 0.69926]` — no near-perfect / leak-shaped arm. |
| Nested selection | PASS | Nested multi-seed selection is active in `fuse4.py` and audit. |
| Permutation no-signal | PASS | Audit perm nested mean `0.5114961768`; fuse4-rule perm mean `0.5117344182` (both ≪ 0.55). |
| Submission format | PASS | `submissions/submission_v4.csv`: 6398 rows, id-aligned, finite labels in `[0,1]`. |
| Target reached (`>= 0.707`) | FAIL | Endorsed honest nested OOF mean = **0.7030017071**. |

`honesty_passed = true`, `target_reached = false`, overall `passed = false`.

## Endorsed headline (exact)

Do **not** cite the raw `audit_v4.py` nested number as the competition headline.
That script’s rule set **lags** `fuse4.RULES` (missing `views_max_10_20_r16`, `views_max_s16`, `views_max_s16_f20`, `views_max_10_20_alt_d5`, `views_half*`, …), so it always locks onto `views_max_10_r16` and reports a flat `0.7032505768`.

The honest, submission-aligned headline I endorse is the fuse4 20-seed nested mean over the **full usable pre-registered/working rule set**:

```text
nested_oof_mean = 0.7030017070980409
nested_oof_sd   = 0.0000983027559991
nested_oof_min  = 0.7028542541910182
nested_oof_max  = 0.7031766861399642
best_full_rule  = views_max_10_20_r16
best_full_oof   = 0.7032789637435225
optimism        = 0.0002772566454816
submitted_rule  = views_max_10_20_r16
honesty_passed  = true
target_reached  = false
gap_to_0.707    = 0.0039982929019591
```

Matches `artifacts/v4/fusion_report_v4.json`.

## New-arm adversarial review

| Arm family | Protocol | Selection-bias finding |
|---|---|---|
| `cat_*_f20` | PASS — fixed iters, no `eval_set`, label-free FE, 20-fold only | OK if rules pre-registered (`c9a4771`). Mild multiplicity vs 10-fold twins. |
| `cat_*_r16` | PASS — same trainer, new seed block `s2700x` | `views_max_r16` / `views_max_10_r16` registered in `f926d12` before these artefacts landed → acceptable. |
| `cat_*_s16` | PASS on fit protocol | **WARN**: `s16` rank ≈ equal mix of base + `r16` (corr ≈ 1.0). Not new signal; doubles rule surface. |
| `cat_*_sf85` | PASS — fixed `train_frac=0.85`, still no early stop | Weaker alone (`~0.69`); only diversity. |
| `cat_alt_d5` | PASS — preset `d5`, fixed 1000 iters | Incremental; no leak pattern. |

Hard red lines checked in `src4/run_world.py`, `worlds_v4.py`, `fuse4.py`, `merge_seeds.py`, `audit_v4.py`:

- no early stopping on scored fold
- no test-label reads
- FE edges on train+test without labels
- fusion via nested selection, not full-OOF cherry-pick as headline

Soft concern (not a hard honesty FAIL): working-tree `fuse4.py` adds `views_max_10_20_r16` / `views_max_s16*` / `views_max_10_20_alt_d5` **after** related arms exist. Nested scoring still bills the selection cost (`optimism ≈ 2.8e-4`). I do **not** upgrade this to cheating, but I reject treating full-OOF `0.70328` as the headline.

## Bayes ceiling (best fusion score)

Score: `views_max_10_20_r16` (full OOF `0.7032789637`), cross-fit / in-sample isotonic on that score:

| Quantity | Value |
|---|---:|
| Base rate | 0.1002009377 |
| CV-calibrated p mean | 0.1002366413 |
| Theoretical Bayes AUC from CV p | 0.7070541231 |
| Theoretical Bayes AUC from in-sample p | 0.7062524265 |
| Redrawn-label AUC using S (mean) | 0.7065229119 |
| Redrawn-label AUC using S (sd) | 0.0069367966 |
| Redrawn-label 95% interval | [0.6933251270, 0.7199931252] |

Interpretation: **0.707 is barely at the edge of the current risk-function ceiling**, not comfortably below it. Hitting nested ≥ 0.707 under honesty is still *possible* only with **new label-free signal** (encoding / loss / world), not by more seed-bagging of the same CatBoost family (mean off-diagonal rank corr among strong/diversity arms ≈ 0.973).

## Risks and veto

1. Endorsed nested `0.7030017071` is **+0.0015** over prior `views_max` `0.7015141215`, still **−0.0040** vs target.
2. `audit_v4.py` rule lag can overstate nested by ~`2.5e-4` if someone quotes it as the headline — veto that quote.
3. `s16` arms are remixes; further `max` pools of correlated clones are diminishing and selection-noisy.
4. No w6/w7 graduating into `artifacts/v4` as strong orthogonal arms yet.
5. Neighbourhood / noise picture from prior audits still applies: labels are far from deterministic in the available signal space.

**Verdict:** honesty **PASS**, target **FAIL**.
I veto any claim that V4 has reached honest nested OOF ≥ 0.707.

**Work-agent continue?** **YES** — continue is allowed under honesty. Required direction: new label-free signal / orthogonal world, then keep nested multi-seed as the only headline. Do not claim target success on full-OOF or on the stale audit nested number. Sync `audit_v4.py` rules to `fuse4.RULES` before the next audit cycle.

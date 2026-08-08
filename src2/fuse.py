"""Fuse the arm predictions and write the submission.

The fusion rule is chosen from a small pre-registered set by *nested* selection:
the OOF rows are split into five outer blocks, the rule is picked on four of
them and applied to the fifth, and the reported AUC is computed on the
prediction assembled that way.  That number carries no selection optimism, so
it is the one the report leads with.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Weight sets over (cat_d5, cat_d6, lgb_te, glm), declared before looking at any
# fused score.  Only these are ever considered.
RULES: dict[str, dict[str, float]] = {
    "cat_d5_only": {"cat_d5": 1.0},
    "cat_pair": {"cat_d5": 0.5, "cat_d6": 0.5},
    "cat_pair_max": {"__max__": 1.0, "cat_d5": 0.5, "cat_d6": 0.5},
    "views_half": {"cat_d5": 0.25, "cat_d6": 0.25, "cat_alt": 0.50},
    "views_thirds": {"cat_d5": 1 / 3, "cat_d6": 1 / 3, "cat_alt": 1 / 3},
    "views_max": {"__max__": 1.0, "cat_d5": 1.0, "cat_d6": 1.0, "cat_alt": 1.0},
    "views_plus_weak": {"cat_d5": 0.30, "cat_d6": 0.30, "cat_alt": 0.30,
                        "glm": 0.05, "lgb_te": 0.05},
    "worlds3_max": {"__max__": 1.0, "cat_d5": 1.0, "cat_d6": 1.0,
                    "cat_alt": 1.0, "cat_alt2": 1.0},
    "worlds3_mean": {"cat_d5": 0.2, "cat_d6": 0.2, "cat_alt": 0.3, "cat_alt2": 0.3},
    "worlds3_gap_max": {"__max__": 1.0, "cat_d5": 1.0, "cat_d6": 1.0,
                        "cat_alt": 1.0, "cat_alt2": 1.0, "gap": 1.0},
    "three_views_gap": {"cat_d5": 0.20, "cat_d6": 0.20, "cat_alt": 0.30, "gap": 0.30},
    "four_views_equal": {"cat_d5": 0.25, "cat_d6": 0.25, "cat_alt": 0.25, "gap": 0.25},
    "four_views_max": {"__max__": 1.0, "cat_d5": 1.0, "cat_d6": 1.0,
                       "cat_alt": 1.0, "gap": 1.0},
    "four_views_plus_weak": {"cat_d5": 0.225, "cat_d6": 0.225, "cat_alt": 0.225,
                             "gap": 0.225, "glm": 0.05, "lgb_te": 0.05},
    "cat_pair_plus_glm": {"cat_d5": 0.425, "cat_d6": 0.425, "glm": 0.15},
    "all_equal": {"cat_d5": 0.25, "cat_d6": 0.25, "glm": 0.25, "lgb_te": 0.25},
}


def _r(v: np.ndarray) -> np.ndarray:
    return rankdata(v) / (len(v) + 1.0)


def apply_rule(rule: dict[str, float], ranks: dict[str, np.ndarray]) -> np.ndarray:
    if "__max__" in rule:
        members = [k for k in rule if k != "__max__"]
        return np.max(np.vstack([ranks[k] for k in members]), axis=0)
    return sum(w * ranks[k] for k, w in rule.items())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("artifacts/v2"))
    ap.add_argument("--submission", type=Path, default=Path("submissions/submission_v2.csv"))
    args = ap.parse_args()

    arms = {}
    for path in sorted(args.dir.glob("arm_*.npz")):
        d = np.load(path)
        arms[path.stem.removeprefix("arm_")] = (d["oof"], d["test"], d["y"])
    if not arms:
        raise SystemExit(f"no arm_*.npz under {args.dir}")
    y = next(iter(arms.values()))[2]

    oof_r = {k: _r(v[0]) for k, v in arms.items()}
    test_r = {k: _r(v[1]) for k, v in arms.items()}
    rules = {n: r for n, r in RULES.items()
             if all(k in oof_r for k in r if k != "__max__")}

    full = {n: float(roc_auc_score(y, apply_rule(r, oof_r))) for n, r in rules.items()}

    # nested selection - no rule ever scores on the block that selected it
    assembled = np.zeros(len(y))
    picks = []
    for outer, (inner_idx, held_idx) in enumerate(
        StratifiedKFold(5, shuffle=True, random_state=99).split(np.zeros(len(y)), y)
    ):
        best, best_auc = None, -1.0
        for n, r in rules.items():
            a = roc_auc_score(y[inner_idx], apply_rule(r, {k: v[inner_idx] for k, v in oof_r.items()}))
            if a > best_auc:
                best, best_auc = n, a
        picks.append(best)
        assembled[held_idx] = _r(apply_rule(rules[best], {k: v[held_idx] for k, v in oof_r.items()}))
    nested_auc = float(roc_auc_score(y, assembled))

    winner = max(set(picks), key=picks.count)
    test_pred = apply_rule(rules[winner], test_r)
    test_pred = (test_pred - test_pred.min()) / (test_pred.max() - test_pred.min() + 1e-12)
    test_pred = 0.001 + 0.998 * test_pred

    sample = pd.read_csv("data/submit_sample.csv")
    test = pd.read_csv("data/test.csv")
    if sample["id"].tolist() != test["id"].tolist():
        raise SystemExit("submission template is not aligned with test.csv")
    sub = sample.copy()
    sub["label"] = test_pred
    args.submission.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.submission, index=False)

    report = {
        "arm_oof_auc": {k: float(roc_auc_score(y, v[0])) for k, v in arms.items()},
        "arm_corr": {f"{a}~{b}": float(np.corrcoef(oof_r[a], oof_r[b])[0, 1])
                     for i, a in enumerate(oof_r) for b in list(oof_r)[i + 1:]},
        "rule_full_oof_auc": full,
        "nested_selection_picks": picks,
        "nested_oof_auc": nested_auc,
        "submitted_rule": winner,
        "submission": str(args.submission),
    }
    (args.dir / "fusion_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

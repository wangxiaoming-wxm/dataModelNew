#!/usr/bin/env python3
"""Collect GPT-5.6 10-fold production arms and fuse with nested selection."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


RULES = {
    "cat_d5_only": {"cat_d5": 1.0},
    "cat_pair_max": {"__max__": 1.0, "cat_d5": 1.0, "cat_d6": 1.0},
    "views_half": {"cat_d5": 0.25, "cat_d6": 0.25, "cat_alt": 0.50},
    "views_thirds": {"cat_d5": 1 / 3, "cat_d6": 1 / 3, "cat_alt": 1 / 3},
    "views_max": {"__max__": 1.0, "cat_d5": 1.0, "cat_d6": 1.0, "cat_alt": 1.0},
    "safe_blend_v2": {"__safe_v2__": 1.0},
}


def _r(v: np.ndarray) -> np.ndarray:
    return rankdata(v) / (len(v) + 1.0)


def apply_rule(rule, ranks, v2_rank=None):
    if "__safe_v2__" in rule:
        # 0.75 * current views_max + 0.25 * frozen v2
        cur = np.maximum.reduce([ranks["cat_d5"], ranks["cat_d6"], ranks["cat_alt"]])
        return 0.75 * _r(cur) + 0.25 * v2_rank
    if "__max__" in rule:
        members = [k for k in rule if k != "__max__"]
        return np.max(np.vstack([ranks[k] for k in members]), axis=0)
    return sum(w * ranks[k] for k, w in rule.items())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("artifacts/gpt56/v3"))
    ap.add_argument("--v2-dir", type=Path, default=Path("artifacts/v2"))
    ap.add_argument("--submission", type=Path, default=Path("submissions/gpt56_s2_10fold.csv"))
    args = ap.parse_args()

    arms = {}
    for path in sorted(args.dir.glob("arm_*.npz")):
        d = np.load(path)
        arms[path.stem.removeprefix("arm_")] = (d["oof"], d["test"], d["y"])
    needed = ["cat_d5", "cat_d6", "cat_alt"]
    missing = [n for n in needed if n not in arms]
    if missing:
        raise SystemExit(f"missing arms: {missing}")
    y = arms["cat_d5"][2]

    v2 = {p.stem[4:]: np.load(p) for p in args.v2_dir.glob("arm_*.npz")}
    v2_rank = _r(np.maximum.reduce([_r(v2[k]["oof"]) for k in ("cat_d5", "cat_d6", "cat_alt")]))
    v2_test = _r(np.maximum.reduce([_r(v2[k]["test"]) for k in ("cat_d5", "cat_d6", "cat_alt")]))

    oof_r = {k: _r(v[0]) for k, v in arms.items()}
    test_r = {k: _r(v[1]) for k, v in arms.items()}
    rules = {n: r for n, r in RULES.items()
             if n == "safe_blend_v2" or all(k in oof_r for k in r if not k.startswith("__"))}

    full = {}
    for n, r in rules.items():
        pred = apply_rule(r, oof_r, v2_rank)
        full[n] = float(roc_auc_score(y, pred))

    assembled = np.zeros(len(y))
    picks = []
    for _, (inner_idx, held_idx) in enumerate(
        StratifiedKFold(5, shuffle=True, random_state=99).split(np.zeros(len(y)), y)
    ):
        best, best_auc = None, -1.0
        for n, r in rules.items():
            a = roc_auc_score(
                y[inner_idx],
                apply_rule(r, {k: v[inner_idx] for k, v in oof_r.items()}, v2_rank[inner_idx]),
            )
            if a > best_auc:
                best, best_auc = n, a
        picks.append(best)
        assembled[held_idx] = _r(
            apply_rule(rules[best], {k: v[held_idx] for k, v in oof_r.items()}, v2_rank[held_idx])
        )
    nested_auc = float(roc_auc_score(y, assembled))
    winner = max(set(picks), key=picks.count)

    test_pred = apply_rule(rules[winner], test_r, v2_test)
    test_pred = (test_pred - test_pred.min()) / (test_pred.max() - test_pred.min() + 1e-12)
    test_pred = 0.001 + 0.998 * test_pred
    sample = pd.read_csv("data/submit_sample.csv")
    sub = sample.copy()
    sub["label"] = test_pred
    args.submission.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.submission, index=False)

    # Also write the pre-registered safe blend regardless of winner.
    safe = apply_rule(RULES["safe_blend_v2"], test_r, v2_test)
    safe = (safe - safe.min()) / (safe.max() - safe.min() + 1e-12)
    safe = 0.001 + 0.998 * safe
    safe_path = args.submission.with_name("gpt56_s2_10fold_safe.csv")
    safe_sub = sample.copy()
    safe_sub["label"] = safe
    safe_sub.to_csv(safe_path, index=False)

    report = {
        "arm_oof_auc": {k: float(roc_auc_score(y, v[0])) for k, v in arms.items()},
        "rule_full_oof_auc": full,
        "nested_selection_picks": picks,
        "nested_oof_auc": nested_auc,
        "submitted_rule": winner,
        "v2_views_max_oof": float(roc_auc_score(y, v2_rank)),
        "delta_vs_v2_nested_proxy": nested_auc - 0.6985627496359704,
        "submission": str(args.submission),
        "safe_submission": str(safe_path),
    }
    (args.dir / "fusion_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

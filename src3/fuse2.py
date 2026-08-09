"""Fusion over an extended set of arms.

Written and committed *before* the w4/w5 arm scores existed, so the rule set is
pre-registered in the branch's sense: no rule here was chosen after seeing what
it would score.  src2/fuse.py's sixteen rules are inherited verbatim for
continuity; the additions are the ones that only become possible once there are
more than three encoding worlds, plus one learned stacker.

Reporting differs from src2/fuse.py in one way that makes the headline harder
rather than easier: the nested selection is repeated over many block seeds and
the *mean* is reported, because a single block seed moves the number by up to
0.002 (measured in src3/audit.py) and quoting one draw of that is a small act of
cherry-picking.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import sys
sys.path.insert(0, "src2")
from fuse import RULES as BASE_RULES  # noqa: E402

STRONG = ("cat_d5", "cat_d6", "cat_alt")
NEW = ("cat_w4", "cat_w5")

EXTRA_RULES: dict[str, dict[str, float]] = {
    "w5_max":        {"__max__": 1.0, **{k: 1.0 for k in STRONG + ("cat_w4", "cat_w5")}},
    "w5_mean":       {k: 1 / 5 for k in STRONG + ("cat_w4", "cat_w5")},
    "w4only_max":    {"__max__": 1.0, **{k: 1.0 for k in STRONG + ("cat_w4",)}},
    "w5only_max":    {"__max__": 1.0, **{k: 1.0 for k in STRONG + ("cat_w5",)}},
    "w4only_mean":   {k: 1 / 4 for k in STRONG + ("cat_w4",)},
    "w5only_mean":   {k: 1 / 4 for k in STRONG + ("cat_w5",)},
    "strong_mean":   {k: 1 / 3 for k in STRONG},
    "strong_heavy":  {"cat_d5": 0.25, "cat_d6": 0.25, "cat_alt": 0.30,
                      "cat_w4": 0.10, "cat_w5": 0.10},
    "all_worlds_max": {"__max__": 1.0,
                       **{k: 1.0 for k in STRONG + ("cat_alt2", "cat_w4", "cat_w5")}},
    "all_worlds_mean": {k: 1 / 6 for k in STRONG + ("cat_alt2", "cat_w4", "cat_w5")},
}

STACKER = "nested_logit_stack"


def _r(v: np.ndarray) -> np.ndarray:
    return rankdata(v) / (len(v) + 1.0)


def apply_rule(rule: dict[str, float], ranks: dict[str, np.ndarray]) -> np.ndarray:
    if "__max__" in rule:
        members = [k for k in rule if k != "__max__"]
        return np.max(np.vstack([ranks[k] for k in members]), axis=0)
    return sum(w * ranks[k] for k, w in rule.items())


def fit_stacker(ranks: dict[str, np.ndarray], y: np.ndarray, idx: np.ndarray):
    names = sorted(ranks)
    A = np.column_stack([ranks[k][idx] for k in names])
    m = LogisticRegression(C=1.0, max_iter=2000)
    m.fit(A, y[idx])
    return names, m


def apply_stacker(model, names, ranks, idx):
    A = np.column_stack([ranks[k][idx] for k in names])
    return model.predict_proba(A)[:, 1]


def nested_run(rules, ranks, y, seed, use_stacker=True):
    assembled = np.zeros(len(y))
    picks = []
    for inner, held in StratifiedKFold(5, shuffle=True, random_state=seed).split(
        np.zeros(len(y)), y
    ):
        best, best_auc, best_kind = None, -np.inf, None
        for n, r in rules.items():
            a = roc_auc_score(y[inner], apply_rule(r, {k: v[inner] for k, v in ranks.items()}))
            if a > best_auc:
                best, best_auc, best_kind = n, a, "rule"
        if use_stacker:
            # the stacker is scored the same way every rule is: fit and judged
            # inside the inner block only, never on the block it will predict
            sub = np.array_split(inner, 2)
            names, m = fit_stacker(ranks, y, sub[0])
            a = roc_auc_score(y[sub[1]], apply_stacker(m, names, ranks, sub[1]))
            if a > best_auc:
                best, best_auc, best_kind = STACKER, a, "stack"
        picks.append(best)
        if best_kind == "stack":
            names, m = fit_stacker(ranks, y, inner)
            assembled[held] = _r(apply_stacker(m, names, ranks, held))
        else:
            assembled[held] = _r(apply_rule(rules[best],
                                            {k: v[held] for k, v in ranks.items()}))
    return float(roc_auc_score(y, assembled)), picks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("artifacts/v3"))
    ap.add_argument("--submission", type=Path, default=Path("submissions/submission_v3.csv"))
    ap.add_argument("--block-seeds", type=int, default=20)
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    arms, y = {}, None
    for p in sorted(args.dir.glob("arm_*.npz")):
        z = np.load(p)
        arms[p.stem.removeprefix("arm_")] = (z["oof"], z["test"])
        y = z["y"] if y is None else y
    if not arms:
        raise SystemExit(f"no arm_*.npz under {args.dir}")

    oof_r = {k: _r(v[0]) for k, v in arms.items()}
    test_r = {k: _r(v[1]) for k, v in arms.items()}
    rules = {n: r for n, r in {**BASE_RULES, **EXTRA_RULES}.items()
             if all(k in oof_r for k in r if k != "__max__")}

    full = {n: float(roc_auc_score(y, apply_rule(r, oof_r))) for n, r in rules.items()}
    nested, all_picks = [], []
    for s in range(90, 90 + args.block_seeds):
        a, picks = nested_run(rules, oof_r, y, s)
        nested.append(a)
        all_picks += picks

    counts = pd.Series(all_picks).value_counts()
    winner = counts.index[0]

    if winner == STACKER:
        names, m = fit_stacker(oof_r, y, np.arange(len(y)))
        A = np.column_stack([test_r[k] for k in names])
        test_pred = m.predict_proba(A)[:, 1]
    else:
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
        "arms": sorted(arms),
        "arm_oof_auc": {k: float(roc_auc_score(y, v[0])) for k, v in arms.items()},
        "arm_rank_corr": {f"{a}~{b}": float(np.corrcoef(oof_r[a], oof_r[b])[0, 1])
                          for i, a in enumerate(sorted(oof_r))
                          for b in sorted(oof_r)[i + 1:]},
        "n_rules": len(rules) + 1,
        "rule_full_oof_auc": dict(sorted(full.items(), key=lambda kv: -kv[1])),
        "nested_oof_mean": float(np.mean(nested)),
        "nested_oof_sd": float(np.std(nested, ddof=1)),
        "nested_oof_min": float(np.min(nested)),
        "nested_oof_max": float(np.max(nested)),
        "pick_counts": counts.to_dict(),
        "submitted_rule": winner,
        "submission": str(args.submission),
    }
    out = args.report or (args.dir / "fusion_report_v3.json")
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

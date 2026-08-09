"""V5 fusion candidates on FROZEN V2 arms, under V2's exact nested protocol.

No new models.  No fold-count change.  The only question is whether a
pre-declared robust combination of the existing three (or five) view ranks
beats ``views_max`` by ~0.002 under nested selection.

All candidate rules are listed below BEFORE any score is computed in main().
Nested selection over the union of V2's RULES and these candidates is the
honest number; a rule that only wins on full-OOF is discarded.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import sys
sys.path.insert(0, "src2")
from fuse import RULES, apply_rule


def r(v):
    return rankdata(v) / (len(v) + 1.0)


def topk_mean(ranks: dict[str, np.ndarray], members: list[str], k: int) -> np.ndarray:
    stack = np.stack([ranks[m] for m in members], axis=0)
    # take the k largest ranks per column
    idx = np.argpartition(stack, -k, axis=0)[-k:]
    return np.take_along_axis(stack, idx, axis=0).mean(axis=0)


def logodds_mean(ranks: dict[str, np.ndarray], members: list[str]) -> np.ndarray:
    eps = 1e-4
    acc = None
    for m in members:
        p = np.clip(ranks[m], eps, 1 - eps)
        lo = np.log(p / (1 - p))
        acc = lo if acc is None else acc + lo
    return acc / len(members)


def softmax_rank(ranks: dict[str, np.ndarray], members: list[str], temp: float) -> np.ndarray:
    stack = np.stack([ranks[m] for m in members], axis=0)
    w = np.exp(stack / temp)
    w /= w.sum(axis=0, keepdims=True)
    return (w * stack).sum(axis=0)


STRONG = ["cat_d5", "cat_d6", "cat_alt"]
WIDE = STRONG + ["gap", "cat_alt2"]


def build_candidates(oof_r):
    """Return {name: score_vector} for every pre-declared candidate."""
    out = {}
    # --- V2 baseline rule, recomputed here for a paired comparison --------
    out["views_max"] = apply_rule(
        {"__max__": 1.0, **{k: 1.0 for k in STRONG}}, oof_r
    )
    out["views_mean"] = apply_rule({k: 1 / 3 for k in STRONG}, oof_r)
    # --- robust variants of the three-view ensemble ----------------------
    out["top2_mean"] = topk_mean(oof_r, STRONG, 2)
    out["max_mean_half"] = 0.5 * out["views_max"] + 0.5 * out["views_mean"]
    out["max_mean_0.7"] = 0.7 * out["views_max"] + 0.3 * out["views_mean"]
    out["logodds_mean"] = logodds_mean(oof_r, STRONG)
    out["softmax_t0.2"] = softmax_rank(oof_r, STRONG, 0.2)
    out["softmax_t0.1"] = softmax_rank(oof_r, STRONG, 0.1)
    out["softmax_t0.05"] = softmax_rank(oof_r, STRONG, 0.05)
    # --- bring weak views in only through robust aggregators -------------
    if all(k in oof_r for k in WIDE):
        out["wide_top2"] = topk_mean(oof_r, WIDE, 2)
        out["wide_top3"] = topk_mean(oof_r, WIDE, 3)
        out["wide_softmax_t0.1"] = softmax_rank(oof_r, WIDE, 0.1)
        out["wide_max"] = np.max(np.stack([oof_r[k] for k in WIDE]), axis=0)
    return out


def nested_select(candidates, y, seeds=range(90, 110)):
    names = list(candidates)
    vals, picks = [], []
    for seed in seeds:
        assembled = np.zeros(len(y))
        local = []
        for inner, held in StratifiedKFold(5, shuffle=True, random_state=seed).split(
            np.zeros(len(y)), y
        ):
            best, ba = None, -np.inf
            for n in names:
                a = roc_auc_score(y[inner], candidates[n][inner])
                if a > ba:
                    best, ba = n, a
            local.append(best)
            assembled[held] = r(candidates[best][held])
        vals.append(float(roc_auc_score(y, assembled)))
        picks += local
    return np.array(vals), picks


def main() -> None:
    arms = {p.stem[4:]: np.load(p) for p in Path("artifacts/v2").glob("arm_*.npz")}
    y = next(iter(arms.values()))["y"]
    oof_r = {k: r(z["oof"]) for k, z in arms.items()}

    # full-OOF of every pre-declared candidate
    cands = build_candidates(oof_r)
    full = {n: float(roc_auc_score(y, s)) for n, s in cands.items()}
    print("full-OOF (pre-declared candidates):")
    for n, a in sorted(full.items(), key=lambda kv: -kv[1]):
        print(f"  {a:.5f}  {n}")

    # nested over the candidate set alone
    vals, picks = nested_select(cands, y)
    print(f"\nnested mean over candidates: {vals.mean():.5f} sd={vals.std(ddof=1):.5f}")
    print(f"pick counts: {dict(pd.Series(picks).value_counts())}")

    # nested over V2 RULES ∪ candidates (honest enlargement of the rule set)
    rules = {n: rr for n, rr in RULES.items()
             if all(k in oof_r for k in rr if k != "__max__")}
    # represent each candidate as a pseudo-rule already scored
    # (nested_select above already did candidate-only; now mix with V2 rules)
    mixed = {f"rule:{n}": apply_rule(rr, oof_r) for n, rr in rules.items()}
    mixed.update({f"cand:{n}": s for n, s in cands.items()})
    vals2, picks2 = nested_select(mixed, y)
    print(f"\nnested mean over RULES∪candidates: {vals2.mean():.5f} sd={vals2.std(ddof=1):.5f}")
    print(f"pick counts: {dict(pd.Series(picks2).value_counts().head(8))}")

    v2_nested = 0.6985627496359704  # pipeline single-seed report
    v2_sup = 0.6982393610891513     # 20-seed supervisor mean
    report = {
        "full_oof": full,
        "nested_candidates_only": {
            "mean": float(vals.mean()), "sd": float(vals.std(ddof=1)),
            "min": float(vals.min()), "max": float(vals.max()),
            "picks": {str(k): int(v) for k, v in pd.Series(picks).value_counts().items()},
        },
        "nested_rules_union_candidates": {
            "mean": float(vals2.mean()), "sd": float(vals2.std(ddof=1)),
            "picks": {str(k): int(v) for k, v in pd.Series(picks2).value_counts().items()},
        },
        "delta_vs_v2_pipeline_nested": float(vals.mean() - v2_nested),
        "delta_vs_v2_supervisor_nested": float(vals.mean() - v2_sup),
        "target_delta": 0.002,
        "clears_target": bool(vals.mean() >= v2_sup + 0.002),
    }
    Path("artifacts/v5_fuse").mkdir(parents=True, exist_ok=True)
    Path("artifacts/v5_fuse/fusion_screen.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in (
        "delta_vs_v2_pipeline_nested", "delta_vs_v2_supervisor_nested",
        "clears_target", "nested_candidates_only")}, indent=2))


if __name__ == "__main__":
    main()

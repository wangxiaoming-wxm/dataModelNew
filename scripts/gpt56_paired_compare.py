#!/usr/bin/env python3
"""Paired comparison of two arm directories with bootstrap CI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def _load(path: Path, arm: str):
    z = np.load(path / f"arm_{arm}.npz")
    return z


def _bootstrap_delta(y, a, b, n_boot: int = 2000, seed: int = 20260809):
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    deltas = []
    for _ in range(n_boot):
        ix = np.r_[rng.choice(pos, size=len(pos), replace=True),
                   rng.choice(neg, size=len(neg), replace=True)]
        deltas.append(roc_auc_score(y[ix], b[ix]) - roc_auc_score(y[ix], a[ix]))
    d = np.asarray(deltas)
    return {
        "delta_mean": float(d.mean()),
        "ci90": [float(np.quantile(d, 0.05)), float(np.quantile(d, 0.95))],
        "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
        "bootstrap_positive_fraction": float((d > 0).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--arm", default="cat_d5")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    pa = _load(args.parent, args.arm)
    ca = _load(args.candidate, args.arm)
    y = pa["y"]
    if not np.array_equal(y, ca["y"]):
        raise SystemExit("y mismatch between parent and candidate")

    parent_oof = pa["oof"]
    cand_oof = ca["oof"]
    report = {
        "parent": str(args.parent),
        "candidate": str(args.candidate),
        "arm": args.arm,
        "parent_auc": float(roc_auc_score(y, parent_oof)),
        "candidate_auc": float(roc_auc_score(y, cand_oof)),
        "delta": float(roc_auc_score(y, cand_oof) - roc_auc_score(y, parent_oof)),
        "spearman": float(spearmanr(parent_oof, cand_oof).statistic),
        "bootstrap": _bootstrap_delta(y, parent_oof, cand_oof, n_boot=args.n_boot),
    }

    # Per-seed paired deltas when raw arrays exist and seed counts match.
    if "oof_prob_by_seed" in pa and "oof_prob_by_seed" in ca:
        ps = pa["oof_prob_by_seed"]
        cs = ca["oof_prob_by_seed"]
        if ps.shape == cs.shape:
            seed_deltas = [
                float(roc_auc_score(y, cs[i]) - roc_auc_score(y, ps[i]))
                for i in range(len(ps))
            ]
            report["per_seed_delta"] = seed_deltas
            report["positive_seed_count"] = int(sum(d > 0 for d in seed_deltas))
            report["n_seeds"] = int(len(seed_deltas))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

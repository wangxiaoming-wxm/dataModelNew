#!/usr/bin/env python3
"""Exploratory frozen rank-blend of v2 and B7 submissions.

Formal use still requires an honest B7 rebuild. This script produces the
pre-registered rank-mean / weighted / max candidates with ID-aligned ranks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rank01(x: np.ndarray) -> np.ndarray:
    return (rankdata(x) - 0.5) / len(x)


def _write_sub(sample: pd.DataFrame, pred: np.ndarray, path: Path) -> None:
    out = sample.copy()
    # keep AUC-preserving open interval similar to fuse.py
    z = (pred - pred.min()) / (pred.max() - pred.min() + 1e-12)
    out["label"] = 0.001 + 0.998 * z
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", type=Path, default=Path("submissions/submission_v2.csv"))
    ap.add_argument("--b7", type=Path, default=Path("submissions/submission_b7_closest_honest.csv"))
    ap.add_argument("--v2-oof-dir", type=Path, default=Path("artifacts/v2"))
    ap.add_argument("--b7-oof", type=Path, default=Path("artifacts/b7_closest/predictions.npz"))
    ap.add_argument("--sample", type=Path, default=Path("data/submit_sample.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("submissions"))
    ap.add_argument("--report", type=Path, default=Path("artifacts/gpt56/s1_frozen_blend/report.json"))
    args = ap.parse_args()

    sample = pd.read_csv(args.sample)
    v2 = pd.read_csv(args.v2)
    b7 = pd.read_csv(args.b7)
    for df, name in [(v2, "v2"), (b7, "b7")]:
        if df["id"].duplicated().any():
            raise SystemExit(f"duplicate ids in {name}")
        if set(df["id"]) != set(sample["id"]):
            raise SystemExit(f"id set mismatch for {name}")
    v2 = sample[["id"]].merge(v2, on="id", how="left", validate="one_to_one")
    b7 = sample[["id"]].merge(b7, on="id", how="left", validate="one_to_one")
    if v2["label"].isna().any() or b7["label"].isna().any():
        raise SystemExit("missing labels after ID align")

    rv = _rank01(v2["label"].to_numpy())
    rb = _rank01(b7["label"].to_numpy())
    rules = {
        "gpt56_s1_frozen_blend.csv": 0.50 * rv + 0.50 * rb,
        "gpt56_s1_v2_75_b7_25.csv": 0.75 * rv + 0.25 * rb,
        "gpt56_s1_rankmax_v2_b7.csv": np.maximum(rv, rb),
    }
    outputs = {}
    for name, pred in rules.items():
        path = args.out_dir / name
        _write_sub(sample, pred, path)
        outputs[name] = {"path": str(path), "sha256": _sha256(path)}

    report = {
        "status": "exploratory_biased_b7_oof",
        "note": "B7 OOF includes outer-fold early stopping; use only as exploratory.",
        "inputs": {
            "v2": _sha256(args.v2),
            "b7": _sha256(args.b7),
            "sample": _sha256(args.sample),
        },
        "test_spearman_v2_b7": float(spearmanr(v2["label"], b7["label"]).statistic),
        "outputs": outputs,
    }

    # Optional OOF diagnostics if artifacts exist.
    if args.v2_oof_dir.exists() and args.b7_oof.exists():
        arms = {p.stem[4:]: np.load(p) for p in args.v2_oof_dir.glob("arm_*.npz")}
        y = arms["cat_d5"]["y"]
        r = lambda x: rankdata(x) / (len(x) + 1.0)
        v = r(np.maximum.reduce([r(arms[k]["oof"]) for k in ("cat_d5", "cat_d6", "cat_alt")]))
        b = r(np.load(args.b7_oof)["oof"])
        report["oof"] = {
            "v2": float(roc_auc_score(y, v)),
            "75_25": float(roc_auc_score(y, 0.75 * v + 0.25 * b)),
            "50_50": float(roc_auc_score(y, 0.50 * v + 0.50 * b)),
            "max": float(roc_auc_score(y, np.maximum(v, b))),
            "warning": "B7 OOF is early-stopping biased; do not use for formal promotion",
        }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Honesty checks for the delivered solution.

1. data files match the recorded SHA256 sums;
2. shuffled-label control - the whole pipeline, including the feature
   engineering that is fitted on train+test, must collapse to AUC ~0.5 when the
   labels are permuted.  This is the check that proves none of the label-free
   transforms (quantile edges, per-source condition scale, frequency counts,
   jitter streams) smuggle target information across folds;
3. the submission is aligned with the organisers' template and in range.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from arms import catboost_frame, fit_predict
from features import fit_edges

EXPECTED_SHA = {
    "train.csv": "494a61073a0438f692914c4868db31df1171e662348e0024e06b120d08d44f28",
    "test.csv": "d6ffd26bd4873fa09f6fac361f59170a880e88e331a01d7a6356bd9184ce55ec",
    "submit_sample.csv": "83cb0263cc5729f61d0e05c68d673dc3f21b41c24bad68afa35159859054c4bf",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", type=Path, default=Path("submissions/submission_v2.csv"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/v2/verify.json"))
    args = ap.parse_args()

    report: dict = {}
    report["data_sha256"] = {n: sha256(Path("data") / n) for n in EXPECTED_SHA}
    report["data_sha256_ok"] = report["data_sha256"] == EXPECTED_SHA

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train["label"].to_numpy()
    raw_all = pd.concat([train.drop(columns=["label"]), test], ignore_index=True)
    edges = fit_edges(raw_all)
    X, cats = catboost_frame(raw_all, edges, stream_offset=1)
    Xtr = X.iloc[: len(train)].reset_index(drop=True)
    Xte = X.iloc[len(train):].reset_index(drop=True)

    y_shuf = y.copy()
    np.random.default_rng(20260808).shuffle(y_shuf)
    oof = np.zeros(len(y_shuf))
    for f, (ti, vi) in enumerate(
        StratifiedKFold(5, shuffle=True, random_state=777).split(Xtr, y_shuf)
    ):
        pv, _ = fit_predict("cat_d5", Xtr.iloc[ti], Xtr.iloc[vi], Xte, cats, y_shuf[ti], 777 + f)
        oof[vi] = pv
    shuffled_auc = float(roc_auc_score(y_shuf, oof))
    report["shuffled_label_auc"] = shuffled_auc
    report["shuffled_label_ok"] = bool(0.47 <= shuffled_auc <= 0.53)

    sample = pd.read_csv("data/submit_sample.csv")
    sub = pd.read_csv(args.submission)
    report["submission"] = {
        "path": str(args.submission),
        "columns_ok": sub.columns.tolist() == ["id", "label"],
        "ids_aligned": sub["id"].tolist() == sample["id"].tolist(),
        "rows": len(sub),
        "finite": bool(np.isfinite(sub["label"]).all()),
        "in_unit_interval": bool(((sub["label"] >= 0) & (sub["label"] <= 1)).all()),
        "min": float(sub["label"].min()),
        "max": float(sub["label"].max()),
    }
    report["all_ok"] = bool(
        report["data_sha256_ok"] and report["shuffled_label_ok"]
        and all(v for k, v in report["submission"].items() if isinstance(v, bool))
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

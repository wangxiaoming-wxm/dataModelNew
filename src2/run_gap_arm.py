"""Score the B6 `gap` feature view as an extra, honestly-validated arm.

The view itself comes from the previous solution (`src/insurance_claim`), but it
is re-run here under this branch's protocol: a fixed tree count and no early
stopping on the outer validation fold.  Its value is diversity - it is an
independently written encoding of the same data, so it decorrelates from the
two views in `src2/features.py`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, "src")
from insurance_claim.train_b6 import build_gap  # noqa: E402

PARAMS = dict(loss_function="Logloss", learning_rate=0.03, depth=6, l2_leaf_reg=10,
              random_strength=0.7, verbose=False, thread_count=4,
              allow_writing_files=False, iterations=500)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[20290, 20291, 20292, 20293])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("artifacts/v2gap"))
    args = ap.parse_args()

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train["label"].to_numpy()
    feats = train.drop(columns=["label"])
    args.out.mkdir(parents=True, exist_ok=True)

    oof_seeds, test_parts, t_start = [], [], time.time()
    for seed in args.seeds:
        t0 = time.time()
        oof = np.zeros(len(y))
        for f, (ti, vi) in enumerate(
            StratifiedKFold(args.folds, shuffle=True, random_state=seed).split(feats, y)
        ):
            Xtr, Xva, Xte, cats = build_gap(
                feats.iloc[ti].reset_index(drop=True),
                feats.iloc[vi].reset_index(drop=True),
                test.copy(),
            )
            m = CatBoostClassifier(**PARAMS, random_seed=seed + f)
            m.fit(Xtr, y[ti], cat_features=cats, verbose=False)
            oof[vi] = m.predict_proba(Xva)[:, 1]
            pt = m.predict_proba(Xte)[:, 1]
            test_parts.append(rankdata(pt) / len(pt))
        oof_seeds.append(rankdata(oof) / len(oof))
        print(f"  gap seed={seed} oof={roc_auc_score(y, oof):.5f} ({time.time() - t0:.0f}s)",
              flush=True)

    oof = np.mean(oof_seeds, axis=0)
    pred = np.mean(test_parts, axis=0)
    np.savez_compressed(args.out / "arm_gap.npz", oof=oof, test=pred, y=y)
    summary = {
        "seeds": args.seeds,
        "bagged_oof_auc": float(roc_auc_score(y, oof)),
        "per_seed_auc": [float(roc_auc_score(y, o)) for o in oof_seeds],
        "elapsed_sec": round(time.time() - t_start, 1),
    }
    (args.out / "gap_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

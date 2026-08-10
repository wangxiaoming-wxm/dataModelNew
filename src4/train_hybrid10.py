"""Hybrid arm: v2 main encoding world + raw x0–x18 (plus-style noise columns).

Hypothesis: plus lifts max3 via x0–x18 orthogonality; merger_ord is the strongest
Ordered world. Combining both feature families under 10-fold ES may yield a
fourth arm that is both strong and complementary.
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src2"))
from arms import CAT_BASE, catboost_frame  # noqa: E402
from features import fit_edges  # noqa: E402

ART = ROOT / "artifacts" / "v4max3pro"
DATA = ROOT / "data"
XCOLS = [f"x{i}" for i in range(19)] + ["x20", "cc", "V", "max_g"]


def hybrid_frame(raw, edges, stream_offset: int):
    X, cats = catboost_frame(raw, edges, stream_offset=stream_offset, n_views=4)
    for c in XCOLS:
        if c in raw.columns:
            X[f"raw_{c}"] = pd.to_numeric(raw[c], errors="coerce").astype(float).fillna(-999.0).to_numpy()
    return X, cats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--stream-offset", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--folds", type=int, default=10)
    args = ap.parse_args()

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["label"].astype(int).values
    raw = pd.concat([train.drop(columns=["label"]), test], ignore_index=True)
    edges = fit_edges(raw)
    X, cats = hybrid_frame(raw, edges, args.stream_offset)
    Xtr = X.iloc[: len(train)].reset_index(drop=True)
    Xte = X.iloc[len(train) :].reset_index(drop=True)

    params = dict(CAT_BASE)
    params.update(
        dict(
            depth=5,
            iterations=1200,
            boosting_type="Ordered",
            thread_count=args.threads,
            eval_metric="AUC",
            od_type="Iter",
            od_wait=150,
        )
    )

    oof = np.zeros(len(y))
    te = np.zeros(len(test))
    t0 = time.time()
    for fold, (ti, vi) in enumerate(
        StratifiedKFold(args.folds, shuffle=True, random_state=args.seed).split(Xtr, y)
    ):
        m = CatBoostClassifier(**dict(params, random_seed=args.seed + fold))
        m.fit(
            Xtr.iloc[ti],
            y[ti],
            eval_set=(Xtr.iloc[vi], y[vi]),
            cat_features=cats,
            use_best_model=True,
            verbose=False,
        )
        oof[vi] = m.predict_proba(Xtr.iloc[vi])[:, 1]
        te += m.predict_proba(Xte)[:, 1] / args.folds
        print(
            f"[hybrid10] seed={args.seed} fold={fold} auc={roc_auc_score(y[vi], oof[vi]):.5f} "
            f"best={m.get_best_iteration()}",
            flush=True,
        )
    auc = float(roc_auc_score(y, oof))
    print(f"[hybrid10] seed={args.seed} OOF={auc:.6f} ({time.time()-t0:.0f}s)", flush=True)
    ART.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ART / f"part_hybrid10_s{args.seed}.npz", oof=oof, test=te, y=y)
    (ART / f"part_hybrid10_s{args.seed}.json").write_text(
        json.dumps({"seed": args.seed, "oof_auc": auc, "n_features": int(X.shape[1])}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

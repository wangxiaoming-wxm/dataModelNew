"""10-fold Ordered arm on the v2 main feature frame (max3 ruler for nested eval).

Hypothesis (gpt56 S2 + plus lesson): plus helps max3 because each model sees
~90% of rows. Apply the same capacity boost to the strongest Ordered world
(merger features), then add it as a 4th/5th max arm — do NOT replace 5-fold max3.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--stream-offset", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=1200)
    ap.add_argument("--es", action="store_true")
    args = ap.parse_args()

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["label"].astype(int).values
    raw = pd.concat([train.drop(columns=["label"]), test], ignore_index=True)
    edges = fit_edges(raw)
    X, cats = catboost_frame(raw, edges, stream_offset=args.stream_offset, n_views=4)
    Xtr = X.iloc[: len(train)].reset_index(drop=True)
    Xte = X.iloc[len(train) :].reset_index(drop=True)

    params = dict(CAT_BASE)
    params.update(
        dict(
            depth=5,
            iterations=args.iterations,
            boosting_type="Ordered",
            thread_count=args.threads,
        )
    )
    if args.es:
        params.update(dict(eval_metric="AUC", od_type="Iter", od_wait=150))

    oof = np.zeros(len(y))
    te = np.zeros(len(test))
    t0 = time.time()
    for fold, (ti, vi) in enumerate(
        StratifiedKFold(args.folds, shuffle=True, random_state=args.seed).split(Xtr, y)
    ):
        m = CatBoostClassifier(**dict(params, random_seed=args.seed + fold))
        if args.es:
            m.fit(
                Xtr.iloc[ti],
                y[ti],
                eval_set=(Xtr.iloc[vi], y[vi]),
                cat_features=cats,
                use_best_model=True,
                verbose=False,
            )
            best = int(m.get_best_iteration() or -1)
        else:
            m.fit(Xtr.iloc[ti], y[ti], cat_features=cats, verbose=False)
            best = args.iterations
        oof[vi] = m.predict_proba(Xtr.iloc[vi])[:, 1]
        te += m.predict_proba(Xte)[:, 1] / args.folds
        print(
            f"[main10_ord] seed={args.seed} fold={fold} auc={roc_auc_score(y[vi], oof[vi]):.5f} best={best}",
            flush=True,
        )
    auc = float(roc_auc_score(y, oof))
    print(f"[main10_ord] seed={args.seed} OOF={auc:.6f} ({time.time()-t0:.0f}s)", flush=True)
    ART.mkdir(parents=True, exist_ok=True)
    tag = f"main10_ord{'_es' if args.es else ''}_s{args.seed}"
    np.savez_compressed(ART / f"part_{tag}.npz", oof=oof, test=te, y=y)
    (ART / f"part_{tag}.json").write_text(
        json.dumps({"seed": args.seed, "oof_auc": auc, "es": args.es, "folds": args.folds}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

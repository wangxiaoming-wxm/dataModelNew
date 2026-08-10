"""10-fold ES Ordered on B5 no-xbin features (capacity upgrade of mn_es family)."""
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
sys.path.insert(0, str(ROOT / "src"))
from insurance_claim.model import TARGET  # noqa: E402
from insurance_claim.train_b5_focus import VIEWS  # noqa: E402

ART = ROOT / "artifacts" / "v4max3pro"
DATA = ROOT / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--folds", type=int, default=10)
    args = ap.parse_args()

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].astype(int)
    builder, b5p = VIEWS["b5"]
    p = {k: v for k, v in b5p.items() if k != "random_seed"}
    p.pop("l2_leaf_reg", None)
    p.update(
        depth=7,
        iterations=1200,
        boosting_type="Ordered",
        thread_count=args.threads,
        eval_metric="AUC",
        od_type="Iter",
        od_wait=150,
        allow_writing_files=False,
        verbose=False,
    )
    features = train.drop(columns=[TARGET])
    oof = np.zeros(len(train))
    te = np.zeros(len(test))
    t0 = time.time()
    for fold, (tri, vai) in enumerate(
        StratifiedKFold(args.folds, shuffle=True, random_state=args.seed).split(features, y)
    ):
        Xtr = features.iloc[tri].reset_index(drop=True)
        Xva = features.iloc[vai].reset_index(drop=True)
        ytr = y.iloc[tri].reset_index(drop=True)
        yva = y.iloc[vai].reset_index(drop=True)
        tr_df, va_df, te_df, cats = builder(Xtr, Xva, test.copy())
        m = CatBoostClassifier(**dict(p, random_seed=args.seed + fold))
        m.fit(tr_df, ytr, eval_set=(va_df, yva), cat_features=cats, use_best_model=True)
        oof[vai] = m.predict_proba(va_df)[:, 1]
        te += m.predict_proba(te_df)[:, 1] / args.folds
        print(
            f"[noxb10] seed={args.seed} fold={fold} auc={roc_auc_score(yva, oof[vai]):.5f} "
            f"best={m.get_best_iteration()}",
            flush=True,
        )
    auc = float(roc_auc_score(y, oof))
    print(f"[noxb10] seed={args.seed} OOF={auc:.6f} ({time.time()-t0:.0f}s)", flush=True)
    ART.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ART / f"part_noxb10_s{args.seed}.npz", oof=oof, test=te, y=y.to_numpy())
    (ART / f"part_noxb10_s{args.seed}.json").write_text(json.dumps({"seed": args.seed, "oof_auc": auc}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

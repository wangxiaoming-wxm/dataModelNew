"""10-fold Ordered plus — keep V10's high training fraction, add Ordered diversity."""
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
from insurance_claim.v10_plus.plus_features import build_plus  # noqa: E402

ART = ROOT / "artifacts" / "v4max3pro"
DATA = ROOT / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--boosting", default="Ordered", choices=["Ordered", "Plain"])
    ap.add_argument("--folds", type=int, default=10)
    args = ap.parse_args()

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].astype(int)
    feats = train.drop(columns=[TARGET])
    oof = np.zeros(len(train))
    pte = np.zeros(len(test))
    t0 = time.time()
    params = dict(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=2500,
        learning_rate=0.02,
        depth=7,
        l2_leaf_reg=20,
        random_strength=1.0,
        od_type="Iter",
        od_wait=150,
        verbose=False,
        thread_count=args.threads,
        allow_writing_files=False,
        boosting_type=args.boosting,
    )
    for fold, (a, b) in enumerate(
        StratifiedKFold(args.folds, shuffle=True, random_state=args.seed).split(feats, y)
    ):
        Xtr = feats.iloc[a].reset_index(drop=True)
        Xva = feats.iloc[b].reset_index(drop=True)
        ytr = y.iloc[a].reset_index(drop=True)
        yva = y.iloc[b].reset_index(drop=True)
        tr, va, te, cats = build_plus(Xtr, Xva, test.copy())
        m = CatBoostClassifier(**dict(params, random_seed=args.seed + fold))
        m.fit(tr, ytr, eval_set=(va, yva), cat_features=cats, use_best_model=True)
        oof[b] = m.predict_proba(va)[:, 1]
        pte += m.predict_proba(te)[:, 1] / args.folds
        print(
            f"[plus10_{args.boosting}] seed={args.seed} fold={fold} "
            f"auc={roc_auc_score(yva, oof[b]):.5f} best={m.get_best_iteration()}",
            flush=True,
        )
    auc = float(roc_auc_score(y, oof))
    print(f"[plus10_{args.boosting}] seed={args.seed} OOF={auc:.6f} ({time.time()-t0:.0f}s)", flush=True)
    ART.mkdir(parents=True, exist_ok=True)
    tag = f"plus10_{args.boosting.lower()}_s{args.seed}"
    np.savez_compressed(ART / f"part_{tag}.npz", oof=oof, test=pte, y=y.to_numpy())
    (ART / f"part_{tag}.json").write_text(json.dumps({"seed": args.seed, "oof_auc": auc}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

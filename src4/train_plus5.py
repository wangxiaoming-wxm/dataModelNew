"""Train V10-plus feature arm under the max3 ruler (stratified 5-fold).

Why: plus is the only inventory arm that lifts max3 nested (+0.0012) thanks to
keeping x0–x18. The frozen V10 plus used 10-fold+ES (optimistic OOF). Here we
retrain at 5-fold so the local nested ruler matches max3; ES is still allowed
for TEST quality (same philosophy as ord_noxb_bag).

boosting_type: Plain or Ordered (CLI).
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
sys.path.insert(0, str(ROOT / "src"))
from insurance_claim.model import TARGET  # noqa: E402
from insurance_claim.v10_plus.plus_features import build_plus  # noqa: E402

ART = ROOT / "artifacts" / "v4max3pro"
DATA = ROOT / "data"


def run_seed(seed: int, folds: int, boosting: str, threads: int, iterations: int):
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].astype(int)
    feats = train.drop(columns=[TARGET])
    oof = np.zeros(len(train), dtype=float)
    pte = np.zeros(len(test), dtype=float)
    fold_meta = []
    t0 = time.time()
    params = dict(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=iterations,
        learning_rate=0.02,
        depth=7,
        l2_leaf_reg=20,
        random_strength=1.0,
        od_type="Iter",
        od_wait=150,
        verbose=False,
        thread_count=threads,
        allow_writing_files=False,
        boosting_type=boosting,
    )
    for fold, (a, b) in enumerate(
        StratifiedKFold(folds, shuffle=True, random_state=seed).split(feats, y)
    ):
        Xtr = feats.iloc[a].reset_index(drop=True)
        Xva = feats.iloc[b].reset_index(drop=True)
        ytr = y.iloc[a].reset_index(drop=True)
        yva = y.iloc[b].reset_index(drop=True)
        tr, va, te, cats = build_plus(Xtr, Xva, test.copy())
        m = CatBoostClassifier(**dict(params, random_seed=seed + fold))
        m.fit(tr, ytr, eval_set=(va, yva), cat_features=cats, use_best_model=True)
        oof[b] = m.predict_proba(va)[:, 1]
        pte += m.predict_proba(te)[:, 1] / folds
        fold_meta.append(
            {
                "fold": fold,
                "valid_auc": float(roc_auc_score(yva, oof[b])),
                "best_iter": int(m.get_best_iteration() or -1),
            }
        )
        print(
            f"[plus_{boosting}_f{folds}] seed={seed} fold={fold} "
            f"auc={fold_meta[-1]['valid_auc']:.5f} best={fold_meta[-1]['best_iter']}",
            flush=True,
        )
    auc = float(roc_auc_score(y, oof))
    print(f"[plus_{boosting}_f{folds}] seed={seed} OOF={auc:.6f} ({time.time()-t0:.0f}s)", flush=True)
    ART.mkdir(parents=True, exist_ok=True)
    tag = f"plus_{boosting.lower()}_f{folds}_s{seed}"
    np.savez_compressed(ART / f"part_{tag}.npz", oof=oof, test=pte, y=y.to_numpy())
    (ART / f"part_{tag}.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "folds": folds,
                "boosting": boosting,
                "oof_auc": auc,
                "folds_meta": fold_meta,
                "elapsed_sec": round(time.time() - t0, 1),
            },
            indent=2,
        )
    )
    return auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--boosting", choices=["Ordered", "Plain"], default="Ordered")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--iterations", type=int, default=2500)
    args = ap.parse_args()
    run_seed(args.seed, args.folds, args.boosting, args.threads, args.iterations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

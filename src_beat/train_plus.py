#!/usr/bin/env python3
"""Expand plus-family arm (V10 plus FE + ES) with new seeds — complementary to max3 in low ranks."""
from __future__ import annotations

import argparse
import json
import os
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

DATA = ROOT / "data"
ART = ROOT / "artifacts" / "beat_max3" / "train"

PARAMS = dict(
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
    allow_writing_files=False,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="plus_new8")
    ap.add_argument("--seeds", type=int, nargs="+", default=[2600, 2601, 2602, 2603, 2604, 2605, 2606, 2607])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--depth", type=int, default=7)
    ap.add_argument("--lr", type=float, default=0.02)
    args = ap.parse_args()

    ART.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].astype(int)
    feats = train.drop(columns=[TARGET])
    params = dict(PARAMS)
    params.update(
        depth=args.depth,
        learning_rate=args.lr,
        thread_count=max(1, (os.cpu_count() or 4) // 2),
    )

    oofs, tes, per = [], [], []
    t0 = time.time()
    for seed in args.seeds:
        part = ART / f"part_{args.tag}_s{seed}.npz"
        if part.exists():
            d = np.load(part)
            oof, te, auc = d["oof"], d["test_pred"], float(d["auc"])
            print(f"[resume] {args.tag} seed {seed}: OOF={auc:.6f}", flush=True)
        else:
            oof = np.zeros(len(train))
            te = np.zeros(len(test))
            t1 = time.time()
            for fold, (a, b) in enumerate(
                StratifiedKFold(args.folds, shuffle=True, random_state=seed).split(feats, y)
            ):
                Xtr = feats.iloc[a].reset_index(drop=True)
                Xva = feats.iloc[b].reset_index(drop=True)
                ytr = y.iloc[a].reset_index(drop=True)
                yva = y.iloc[b].reset_index(drop=True)
                tr, va, te_df, cats = build_plus(Xtr, Xva, test.copy())
                m = CatBoostClassifier(**dict(params, random_seed=seed + fold))
                m.fit(tr, ytr, eval_set=(va, yva), cat_features=cats, use_best_model=True)
                oof[b] = m.predict_proba(va)[:, 1]
                te += m.predict_proba(te_df)[:, 1] / args.folds
                print(
                    f"  seed={seed} fold={fold} auc={roc_auc_score(yva, oof[b]):.5f} "
                    f"best={m.get_best_iteration()}",
                    flush=True,
                )
            auc = float(roc_auc_score(y, oof))
            np.savez(part, oof=oof, test_pred=te, auc=auc, seed=seed)
            print(f"[{args.tag}] seed {seed}: OOF={auc:.6f} ({time.time()-t1:.0f}s)", flush=True)
        oofs.append(oof)
        tes.append(te)
        per.append(auc)

    pooled_oof = np.mean(np.vstack(oofs), 0)
    pooled_te = np.mean(np.vstack(tes), 0)
    out = ART.parent / f"{args.tag}.npz"
    np.savez(out, oof=pooled_oof, test_pred=pooled_te, per_seed=np.array(per), seeds=np.array(args.seeds))
    meta = {
        "tag": args.tag,
        "pooled_oof_auc": float(roc_auc_score(y, pooled_oof)),
        "per_seed": per,
        "seeds": args.seeds,
        "elapsed_sec": round(time.time() - t0, 1),
        "note": "plus-family ES arm; complementary in low-rank region",
    }
    (ART.parent / f"meta_{args.tag}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()

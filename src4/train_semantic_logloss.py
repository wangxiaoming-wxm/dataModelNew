#!/usr/bin/env python3
"""Train the 715-style semantic FE + CatBoostClassifier(Logloss) arm.

Faithful port of explore_d_online.py from the claimed-LB-0.71504 zip:
  - FeatureBuilderSemantic with TOP_CROSS filter + ratio features
  - 5 seeds x 5 folds x 10 bagging, RMSE, ES=120
  - fold-local fit (no label leak into test)

Usage:
  python3 src4/train_semantic_logloss.py --smoke
  python3 src4/train_semantic_logloss.py
  python3 src4/train_semantic_logloss.py --seed 2026   # single seed part
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src4"))
from feat_semantic import FeatureBuilderSemantic  # noqa: E402

DATA = ROOT / "data"
ART = ROOT / "artifacts" / "v4max3pronew"

SEEDS = (2026, 2027, 2028, 2029, 2030)
N_SPLITS = 5
BAG_SEEDS = tuple(range(10))
ITER, LR, DEPTH, L2, RS, ES = 900, 0.03, 6, 10, 0.7, 120

TOP_CROSS = [
    "region__X__source__X__age_range__category_cross",
    "source__X__age_range__X__livability__category_cross",
    "region__X__livability__category_cross",
    "age_range__X__month__X__livability__category_cross",
    "region__X__age_range__X__month__category_cross",
    "region__X__age_range__X__livability__category_cross",
    "region__X__source__X__livability__category_cross",
    "source__X__livability__category_cross",
    "age_range__X__livability__category_cross",
    "region__X__age_range__category_cross",
    "region__X__source__category_cross",
    "source__X__age_range__category_cross",
    "month__X__livability__category_cross",
    "version__X__age_range__X__month__category_cross",
    "source__X__version__X__age_range__category_cross",
    "source__X__month__X__livability__category_cross",
    "region__X__month__X__livability__category_cross",
    "source__X__month__category_cross",
    "source__X__version__category_cross",
    "region__X__month__category_cross",
    "region__X__version__X__age_range__category_cross",
    "region__X__version__X__livability__category_cross",
]
RATIO_PAIRS = [
    ("max_g", "days"),
    ("cc", "days"),
    ("V", "days"),
    ("condition", "days"),
    ("max_g", "cc"),
    ("max_g", "V"),
    ("cc", "V"),
    ("V", "cc"),
    ("max_g", "condition"),
    ("cc", "condition"),
    ("V", "condition"),
    ("days", "age_range"),
    ("condition", "age_range"),
    ("max_g", "livability"),
    ("cc", "livability"),
    ("V", "livability"),
]


def add_ratios(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for num, den in RATIO_PAIRS:
        n = pd.to_numeric(df[num], errors="coerce")
        d = pd.to_numeric(df[den], errors="coerce")
        safe_d = np.where(d > 0, d, np.nan)
        r = n / safe_d
        out[f"ratio_{num}__by__{den}"] = r.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    return out


def build(train_part: pd.DataFrame, test_part: pd.DataFrame, fb: FeatureBuilderSemantic):
    tr = fb.transform(train_part)
    te = fb.transform(test_part)
    tr = pd.concat([tr, add_ratios(train_part)], axis=1)
    te = pd.concat([te, add_ratios(test_part)], axis=1)
    return tr, te


def run_seed(seed: int, train: pd.DataFrame, test: pd.DataFrame, y: np.ndarray, threads: int):
    fb = FeatureBuilderSemantic(selected_cross=set(TOP_CROSS))
    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    test_pred = np.zeros(len(test), dtype=float)
    nb = len(BAG_SEEDS)
    fold_aucs = []
    for fold, (tri, vali) in enumerate(skf.split(train, y)):
        t0 = time.time()
        fb.fit(train.iloc[tri])
        Xtr, Xte = build(train.iloc[tri], test, fb)
        Xva = fb.transform(train.iloc[vali])
        Xva = pd.concat([Xva, add_ratios(train.iloc[vali])], axis=1)
        cat = [c for c in Xtr.columns if c in fb.categorical_features_]
        cidx = [Xtr.columns.get_loc(c) for c in cat]
        for c in cat:
            Xtr[c] = Xtr[c].astype("category")
            Xva[c] = Xva[c].astype("category")
            Xte[c] = Xte[c].astype("category")
        fold_te = np.zeros(len(test), dtype=float)
        for bs in BAG_SEEDS:
            m = CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="AUC",
                iterations=ITER,
                learning_rate=LR,
                depth=DEPTH,
                l2_leaf_reg=L2,
                random_strength=RS,
                bagging_temperature=0.7,
                subsample=0.9,
                border_count=128,
                verbose=0,
                allow_writing_files=False,
                thread_count=threads,
                random_seed=(seed * 100 + bs),
            )
            m.fit(
                Pool(Xtr, y[tri], cat_features=cidx),
                eval_set=Pool(Xva, y[vali], cat_features=cidx),
                early_stopping_rounds=ES,
                verbose=0,
            )
            oof[vali] += m.predict_proba(Xva)[:, 1]
            fold_te += m.predict_proba(Xte)[:, 1]
        oof[vali] /= nb
        test_pred += fold_te / nb
        fa = float(roc_auc_score(y[vali], oof[vali]))
        fold_aucs.append(fa)
        print(
            f"[semantic_logloss] seed={seed} fold={fold} auc={fa:.5f} ({time.time()-t0:.0f}s)",
            flush=True,
        )
    test_pred /= N_SPLITS
    return oof, test_pred, fold_aucs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=None, help="train a single seed part")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--bags", type=int, default=None, help="override bagging count (default 10)")
    ap.add_argument("--tag", type=str, default="semantic_logloss", help="artifact name stem")
    ap.add_argument("--merge-only", action="store_true", help="merge part_*.npz into semantic_logloss.npz")
    args = ap.parse_args()

    global SEEDS, N_SPLITS, BAG_SEEDS, ITER, ES
    ART.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("semantic_logloss")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.bags is not None:
        BAG_SEEDS = tuple(range(int(args.bags)))

    tag = args.tag
    if args.merge_only:
        parts = sorted(ART.glob(f"part_{tag}_s*.npz"))
        if not parts:
            raise SystemExit(f"no part_{tag}_s*.npz found")
        oofs, tes, seeds = [], [], []
        for p in parts:
            d = np.load(p)
            oofs.append(np.asarray(d["oof"], dtype=float))
            tes.append(np.asarray(d["test"], dtype=float))
            seeds.append(int(d["seed"]))
        oof = np.mean(np.vstack(oofs), axis=0)
        te = np.mean(np.vstack(tes), axis=0)
        y = pd.read_csv(DATA / "train.csv")["label"].astype(int).values
        auc = float(roc_auc_score(y, oof))
        out = ART / f"{tag}.npz"
        np.savez_compressed(
            out,
            oof=oof,
            test_pred=te,
            per_seed=np.array([float(roc_auc_score(y, o)) for o in oofs]),
            seeds=np.array(seeds),
            pool=np.array([auc]),
        )
        meta = {
            "arm": tag,
            "protocol": f"es5_logloss_bag{len(BAG_SEEDS)}",
            "n_parts": len(parts),
            "seeds": seeds,
            "pooled_oof_auc": auc,
            "per_seed_auc": [float(roc_auc_score(y, o)) for o in oofs],
            "source": "715 FE + Logloss diversity arm (same protocol as RMSE)",
        }
        (ART / f"{tag}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"merged {len(parts)} parts -> {out} pooled_oof={auc:.5f}", flush=True)
        return 0

    if args.smoke:
        SEEDS = (2026,)
        N_SPLITS = 2
        BAG_SEEDS = (0, 1, 2)
        ITER = 200
        ES = 40
        print("=== SMOKE semantic_logloss ===", flush=True)

    train = pd.read_csv(DATA / "train.csv", dtype={"id": str})
    test = pd.read_csv(DATA / "test.csv", dtype={"id": str})
    y = train["label"].astype(int).values

    seeds = [args.seed] if args.seed is not None else list(SEEDS)
    oof_map, te_map, aucs = {}, {}, []
    for s in seeds:
        o, t, _ = run_seed(s, train, test, y, args.threads)
        oof_map[s], te_map[s] = o, t
        a = float(roc_auc_score(y, o))
        aucs.append(a)
        print(f"[{tag}] seed={s} OOF={a:.5f}", flush=True)
        if not args.smoke:
            part = ART / f"part_{tag}_s{s}.npz"
            np.savez_compressed(part, oof=o, test=t, seed=s, y=y)
            (ART / f"part_{tag}_s{s}.json").write_text(
                json.dumps({"seed": s, "oof_auc": a, "bags": len(BAG_SEEDS)}, indent=2),
                encoding="utf-8",
            )
            print(f"  wrote {part}", flush=True)

    pooled = np.mean(np.vstack(list(oof_map.values())), axis=0)
    te = np.mean(np.vstack(list(te_map.values())), axis=0)
    pa = float(roc_auc_score(y, pooled))
    print(f">>> pooled OOF={pa:.5f}  mean±std={np.mean(aucs):.5f}±{np.std(aucs):.5f}", flush=True)

    if args.smoke:
        smoke_path = ART / f"{tag}_smoke.npz"
        np.savez_compressed(smoke_path, oof=pooled, test_pred=te, y=y)
        print(f"smoke artifact -> {smoke_path}", flush=True)
        return 0

    if args.seed is None:
        out = ART / f"{tag}.npz"
        np.savez_compressed(
            out,
            oof=pooled,
            test_pred=te,
            per_seed=np.array(aucs),
            seeds=np.array(seeds),
            pool=np.array([pa]),
        )
        meta = {
            "arm": tag,
            "protocol": f"es5_logloss_bag{len(BAG_SEEDS)}",
            "seeds": seeds,
            "n_splits": N_SPLITS,
            "bagging": len(BAG_SEEDS),
            "params": {
                "iter": ITER,
                "lr": LR,
                "depth": DEPTH,
                "l2": L2,
                "random_strength": RS,
                "es": ES,
                "loss": "Logloss",
            },
            "pooled_oof_auc": pa,
            "per_seed_auc": aucs,
            "source": "715 FE + Logloss diversity arm (same protocol as RMSE)",
        }
        (ART / f"{tag}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

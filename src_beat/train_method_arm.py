#!/usr/bin/env python3
"""Methodology arm: B5 backbone + CoFEH-distilled ops + fold-local features_goldmine.

search-first stack:
  - features_goldmine (PyPI) for candidate FE
  - CoFEH spirit: distill high-gain ops into deterministic columns
  - Made-With-ML: residual-aware complementary arm (ES CatBoost Ordered)
  - Gate via src_beat/supervise.py (max3 base frozen)
"""
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
from insurance_claim.train_b5_focus import VIEWS  # noqa: E402

DATA = ROOT / "data"
ART = ROOT / "artifacts" / "beat_max3" / "train"
DROP_NOISE = [f"x{i}" for i in range(19)]


def lean_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.drop(columns=[c for c in ["id", "label"] + DROP_NOISE if c in out.columns], errors="ignore")
    for c in out.columns:
        if out[c].dtype == object or str(out[c].dtype).startswith("string"):
            out[c] = out[c].astype(str)
    return out


def fit_group_maps(tr: pd.DataFrame) -> dict:
    days = pd.to_numeric(tr["days"], errors="coerce")
    cc = pd.to_numeric(tr["cc"], errors="coerce")
    cond = pd.to_numeric(tr["condition"], errors="coerce")
    src = tr["source"].astype(str)
    reg = tr["region"].astype(str)
    return {
        "source_cc_mean": pd.Series(cc.values, index=src).groupby(level=0).mean().to_dict(),
        "source_days_mean": pd.Series(days.values, index=src).groupby(level=0).mean().to_dict(),
        "source_cond_mean": pd.Series(cond.values, index=src).groupby(level=0).mean().to_dict(),
        "region_days_mean": pd.Series(days.values, index=reg).groupby(level=0).mean().to_dict(),
        "global_cc": float(np.nanmean(cc)),
        "global_days": float(np.nanmean(days)),
        "global_cond": float(np.nanmean(cond)),
    }


def cofeh_numeric(df: pd.DataFrame, maps: dict) -> pd.DataFrame:
    days = pd.to_numeric(df["days"], errors="coerce")
    cond = pd.to_numeric(df["condition"], errors="coerce")
    age = pd.to_numeric(df["age_range"], errors="coerce")
    cc = pd.to_numeric(df["cc"], errors="coerce")
    V = pd.to_numeric(df["V"], errors="coerce")
    src = df["source"].astype(str)
    reg = df["region"].astype(str)
    out = pd.DataFrame(index=df.index)
    out["cf_age_mul_days"] = age * days
    out["cf_age_div_days"] = age / days.clip(lower=1.0)
    out["cf_days_x_cond"] = days * cond
    out["cf_cond_over_days"] = cond / days.clip(lower=1.0)
    out["cf_log_days"] = np.log1p(days.clip(lower=0))
    out["cf_log_cc"] = np.log1p(cc.clip(lower=0))
    out["cf_log_V"] = np.log1p(V.clip(lower=0))
    out["cf_gde_cc_src"] = cc - src.map(maps["source_cc_mean"]).fillna(maps["global_cc"])
    out["cf_gde_days_src"] = days - src.map(maps["source_days_mean"]).fillna(maps["global_days"])
    out["cf_gde_cond_src"] = cond - src.map(maps["source_cond_mean"]).fillna(maps["global_cond"])
    out["cf_gde_days_reg"] = days - reg.map(maps["region_days_mean"]).fillna(maps["global_days"])
    out["cf_rule_days_cond"] = ((days <= 3000) & (cond > 0.09)).astype(float)
    return out.astype(float).fillna(0.0)


def augment(tr, va, te, Xtr, Xva, Xte, ytr, mode: str, seed: int):
    """Append cofeh (+ optional goldmine) columns onto B5 matrices."""
    tr0, va0, te0 = lean_frame(Xtr), lean_frame(Xva), lean_frame(Xte)
    maps = fit_group_maps(tr0)
    for frame, src in ((tr, tr0), (va, va0), (te, te0)):
        ops = cofeh_numeric(src, maps)
        for c in ops.columns:
            frame[c] = ops[c].to_numpy()

    if mode == "goldmine":
        from features_goldmine import GoldenFeatures

        gf = GoldenFeatures(
            random_state=seed,
            verbose=0,
            selectivity="strict",
            max_selected_features=20,
            exclude_strategies=["categorical_oof_target"],
        )
        g_tr = gf.fit_transform(tr0, np.asarray(ytr))
        g_va = gf.transform(va0).reindex(columns=g_tr.columns, fill_value=0.0)
        g_te = gf.transform(te0).reindex(columns=g_tr.columns, fill_value=0.0)
        for g in (g_tr, g_va, g_te):
            g.replace([np.inf, -np.inf], np.nan, inplace=True)
            g.fillna(0.0, inplace=True)
        for c in g_tr.columns:
            tr[f"gm_{c}"] = g_tr[c].to_numpy()
            va[f"gm_{c}"] = g_va[c].to_numpy()
            te[f"gm_{c}"] = g_te[c].to_numpy()
    return tr, va, te


def run_seed(mode, builder, features, test, y, seed, n_splits):
    oof = np.zeros(len(features))
    te_pred = np.zeros(len(test))
    ncpu = os.cpu_count() or 4
    params = dict(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=1400,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=10,
        random_strength=0.7,
        boosting_type="Ordered",
        od_type="Iter",
        od_wait=150,
        verbose=False,
        allow_writing_files=False,
        thread_count=max(1, ncpu // 2),
    )
    for fold, (tri, vai) in enumerate(
        StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(features, y)
    ):
        Xtr = features.iloc[tri].reset_index(drop=True)
        Xva = features.iloc[vai].reset_index(drop=True)
        ytr = y.iloc[tri].reset_index(drop=True)
        yva = y.iloc[vai].reset_index(drop=True)
        tr, va, te, cats = builder(Xtr, Xva, test.copy())
        tr, va, te = augment(tr, va, te, Xtr, Xva, test.copy(), ytr, mode, seed + fold)
        # ensure va/te have all columns
        va = va.reindex(columns=tr.columns)
        te = te.reindex(columns=tr.columns)
        m = CatBoostClassifier(**dict(params, random_seed=seed + fold))
        m.fit(tr, ytr, eval_set=(va, yva), cat_features=cats, use_best_model=True)
        oof[vai] = m.predict_proba(va)[:, 1]
        te_pred += m.predict_proba(te)[:, 1] / n_splits
        print(
            f"  [{mode}] seed={seed} fold={fold} auc={roc_auc_score(yva, oof[vai]):.5f} "
            f"best={m.get_best_iteration()} n={tr.shape[1]}",
            flush=True,
        )
    return oof, te_pred, float(roc_auc_score(y, oof))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cofeh", "goldmine"], default="cofeh")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[2700, 2701, 2702, 2703])
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    tag = args.tag or f"{args.mode}_arm"
    ART.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["label"].astype(int)
    features = train.drop(columns=["label"])
    builder, _ = VIEWS["b5"]

    oofs, tes, per = [], [], []
    t0 = time.time()
    for seed in args.seeds:
        part = ART / f"part_{tag}_s{seed}.npz"
        if part.exists():
            d = np.load(part)
            oof, te_p, auc = d["oof"], d["test_pred"], float(d["auc"])
            print(f"[resume] {tag} seed {seed}: OOF={auc:.6f}", flush=True)
        else:
            t1 = time.time()
            oof, te_p, auc = run_seed(args.mode, builder, features, test, y, seed, args.folds)
            np.savez(part, oof=oof, test_pred=te_p, auc=auc, seed=seed)
            print(f"[{tag}] seed {seed}: OOF={auc:.6f} ({time.time()-t1:.0f}s)", flush=True)
        oofs.append(oof)
        tes.append(te_p)
        per.append(auc)

    pooled_oof = np.mean(np.vstack(oofs), 0)
    pooled_te = np.mean(np.vstack(tes), 0)
    out = ART.parent / f"{tag}.npz"
    np.savez(out, oof=pooled_oof, test_pred=pooled_te, per_seed=np.array(per), seeds=np.array(args.seeds))
    meta = {
        "tag": tag,
        "mode": args.mode,
        "backbone": "b5",
        "methodology": [
            "features_goldmine",
            "CoFEH-distilled-ops",
            "Made-With-ML-residual",
            "search-first",
            "verification-gate",
        ],
        "pooled_oof_auc": float(roc_auc_score(y, pooled_oof)),
        "per_seed": per,
        "seeds": args.seeds,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (ART.parent / f"meta_{tag}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()

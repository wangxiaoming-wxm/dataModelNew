#!/usr/bin/env python3
"""Strategy-compliant probes from 下一步策略_20260811.md §6 场景三.

Admission gate (hard):
  single-arm 5-fold OOF AUC > 0.690 AND Spearman corr vs merger_ord8 < 0.88
If fail → do NOT enter max fusion.

Never stack high-corr twin arms (noxb10/w12 kitchen-sink) — that is the v4ext failure mode.
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
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from insurance_claim.train_b5_focus import VIEWS  # noqa: E402

DATA = ROOT / "data"
ART = ROOT / "artifacts" / "beat_max3"
OUT = ART / "probes"
N_SPLITS = 5


def cond_r_ratio(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    med = out.groupby("source")["condition"].transform("median").replace(0, np.nan)
    cr = (pd.to_numeric(out["condition"], errors="coerce") / med).fillna(1.0)
    days = pd.to_numeric(out["days"], errors="coerce")
    out["cond_r"] = cr
    out["ratio"] = days / cr.clip(lower=1e-9)
    out["rate"] = days * (1.0 - out.groupby("source")["condition"].rank(pct=True))
    return out


def add_exp1_ratio_bins(tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame):
    """Fold-local ratio quantile bins + source×ratio_bin (as cats, no TE)."""
    tr, va, te = cond_r_ratio(tr), cond_r_ratio(va), cond_r_ratio(te)
    qs = np.quantile(tr["ratio"].to_numpy(float), [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95])
    for name, frame in ("tr", tr), ("va", va), ("te", te):
        frame["ratio_bin"] = np.digitize(frame["ratio"].to_numpy(float), qs).astype(str)
        frame["src_ratio_bin"] = frame["source"].astype(str) + "|" + frame["ratio_bin"]
        # nonlinear transforms (label-free)
        frame["log_ratio"] = np.log(frame["ratio"].clip(lower=1e-9))
        frame["sqrt_ratio"] = np.sqrt(frame["ratio"].clip(lower=0))
        frame["inv_ratio"] = 1.0 / frame["ratio"].clip(lower=1e-9)
        # within-source z of ratio (fit mean/std on train)
    mu = tr.groupby("source")["ratio"].mean()
    sd = tr.groupby("source")["ratio"].std().replace(0, np.nan).fillna(1.0)
    for frame in (tr, va, te):
        src = frame["source"]
        frame["ratio_z_src"] = (frame["ratio"] - src.map(mu)) / src.map(sd).fillna(1.0)
    return tr, va, te


def add_exp2_cliff(tr, va, te):
    tr, va, te = cond_r_ratio(tr), cond_r_ratio(va), cond_r_ratio(te)
    for frame in (tr, va, te):
        cond = pd.to_numeric(frame["condition"], errors="coerce")
        frame["cond_cliff"] = (cond < 0.05).astype(int).astype(str)
        frame["cond_cliff_src"] = frame["cond_cliff"] + "|" + frame["source"].astype(str)
        frame["cond_band"] = pd.cut(
            cond.fillna(cond.median()),
            bins=[-np.inf, 0.05, 0.10, 0.20, np.inf],
            labels=["c0", "c1", "c2", "c3"],
        ).astype(str)
        frame["cond_band_src"] = frame["cond_band"] + "|" + frame["source"].astype(str)
    return tr, va, te


def prepare_cat(tr, va, te):
    cat_cols = []
    for c in tr.columns:
        if tr[c].dtype == object or str(tr[c].dtype).startswith("string"):
            cat_cols.append(c)
    force = [
        "region", "source", "month", "version", "grades", "code",
        "ratio_bin", "src_ratio_bin", "cond_cliff", "cond_cliff_src", "cond_band", "cond_band_src",
    ]
    for c in force:
        if c in tr.columns and c not in cat_cols:
            cat_cols.append(c)
    tr, va, te = tr.copy(), va.copy(), te.copy()
    for c in cat_cols:
        for d in (tr, va, te):
            d[c] = d[c].astype(str).fillna("__NA__")
    for c in tr.columns:
        if c in cat_cols:
            continue
        tr[c] = pd.to_numeric(tr[c], errors="coerce")
        med = float(tr[c].median()) if tr[c].notna().any() else 0.0
        tr[c] = tr[c].fillna(med)
        va[c] = pd.to_numeric(va[c], errors="coerce").fillna(med)
        te[c] = pd.to_numeric(te[c], errors="coerce").fillna(med)
    va = va.reindex(columns=tr.columns)
    te = te.reindex(columns=tr.columns)
    return tr, va, te, cat_cols


def build_matrix(mode: str, Xtr, Xva, Xte):
    builder, _ = VIEWS["b5"]
    tr, va, te, cats = builder(Xtr, Xva, Xte)
    # also keep raw for feature add via parallel frames
    rtr, rva, rte = Xtr.copy(), Xva.copy(), Xte.copy()
    if mode == "exp1":
        rtr, rva, rte = add_exp1_ratio_bins(rtr, rva, rte)
        extra = ["ratio", "cond_r", "rate", "log_ratio", "sqrt_ratio", "inv_ratio", "ratio_z_src", "ratio_bin", "src_ratio_bin"]
    elif mode == "exp2":
        rtr, rva, rte = add_exp2_cliff(rtr, rva, rte)
        extra = ["ratio", "cond_r", "cond_cliff", "cond_cliff_src", "cond_band", "cond_band_src"]
    else:
        raise ValueError(mode)
    rtr2, rva2, rte2, cats2 = prepare_cat(rtr[extra], rva[extra], rte[extra])
    for c in rtr2.columns:
        name = f"probe_{c}"
        tr[name] = rtr2[c].to_numpy()
        va[name] = rva2[c].to_numpy()
        te[name] = rte2[c].to_numpy()
        if c in cats2:
            cats.append(name)
    va = va.reindex(columns=tr.columns)
    te = te.reindex(columns=tr.columns)
    return tr, va, te, cats


def run_probe(mode: str, seeds: list[int]) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["label"].astype(int)
    features = train.drop(columns=["label"])
    mo8 = np.load(ART / "merger_ord8.npz")["oof"]
    ca8 = np.load(ART / "v2_cat_alt8.npz")["oof"]

    ncpu = os.cpu_count() or 4
    params = dict(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=2000,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=10,
        boosting_type="Ordered",
        od_type="Iter",
        od_wait=200,
        verbose=False,
        allow_writing_files=False,
        thread_count=max(1, ncpu // 2),
    )

    oofs, tes, per = [], [], []
    t0 = time.time()
    for seed in seeds:
        part = OUT / f"part_{mode}_s{seed}.npz"
        if part.exists():
            d = np.load(part)
            oof, te_p, auc = d["oof"], d["test_pred"], float(d["auc"])
            print(f"[resume] {mode} s{seed} {auc:.5f}", flush=True)
        else:
            oof = np.zeros(len(train))
            te_p = np.zeros(len(test))
            for fold, (a, b) in enumerate(StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed).split(features, y)):
                Xtr = features.iloc[a].reset_index(drop=True)
                Xva = features.iloc[b].reset_index(drop=True)
                ytr = y.iloc[a].reset_index(drop=True)
                yva = y.iloc[b].reset_index(drop=True)
                tr, va, te, cats = build_matrix(mode, Xtr, Xva, test.copy())
                m = CatBoostClassifier(**dict(params, random_seed=seed + fold))
                m.fit(tr, ytr, eval_set=(va, yva), cat_features=cats, use_best_model=True)
                oof[b] = m.predict_proba(va)[:, 1]
                te_p += m.predict_proba(te)[:, 1] / N_SPLITS
                print(f"  {mode} s{seed} f{fold} auc={roc_auc_score(yva, oof[b]):.5f} best={m.get_best_iteration()}", flush=True)
            auc = float(roc_auc_score(y, oof))
            np.savez(part, oof=oof, test_pred=te_p, auc=auc)
            print(f"[{mode}] seed {seed} OOF={auc:.5f}", flush=True)
        oofs.append(oof)
        tes.append(te_p)
        per.append(auc)

    oof = np.mean(np.vstack(oofs), 0)
    te = np.mean(np.vstack(tes), 0)
    auc = float(roc_auc_score(y, oof))
    corr_mo = float(spearmanr(oof, mo8).correlation)
    corr_ca = float(spearmanr(oof, ca8).correlation)
    gate = {"auc_gt_0.690": auc > 0.690, "corr_mo8_lt_0.88": corr_mo < 0.88, "corr_ca8_lt_0.88": corr_ca < 0.88}
    admit = all(gate.values())
    np.savez(ART / f"probe_{mode}.npz", oof=oof, test_pred=te, per_seed=np.array(per), seeds=np.array(seeds))
    report = {
        "mode": mode,
        "pooled_oof_auc": auc,
        "per_seed": per,
        "corr_spearman_mo8": corr_mo,
        "corr_spearman_ca8": corr_ca,
        "gate": gate,
        "admit_to_max": admit,
        "elapsed_sec": round(time.time() - t0, 1),
        "note": "Strategy §6 exp; refuse max entry if gate fails",
    }
    (OUT / f"report_{mode}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return report


def run_exp3_lowratio_corrector(seeds: list[int]) -> dict:
    """Residual corrector on low-ratio positives subgroup (strategy exp3)."""
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["label"].astype(int).to_numpy()
    # fused max3 oof ranks
    def rk(a):
        from scipy.stats import rankdata

        return rankdata(a) / len(a)

    mo = rk(np.load(ART / "merger_ord8.npz")["oof"])
    ca = rk(np.load(ART / "v2_cat_alt8.npz")["oof"])
    od = rk(np.load(ART / "ord_noxb_bag.npz")["oof"])
    fused = np.maximum.reduce([mo, ca, od])
    # residual target
    resid = y.astype(float) - fused
    tr = cond_r_ratio(train)
    te = cond_r_ratio(test)
    q10 = float(np.quantile(tr["ratio"], 0.10))
    low = (tr["ratio"] <= q10).to_numpy()
    print(f"[exp3] low-ratio n={low.sum()} q10={q10:.3f}", flush=True)

    # features: lean numeric + cats
    feats = ["days", "condition", "age_range", "ratio", "cond_r", "cc", "V", "livability", "x19", "x20"]
    cats = ["region", "source", "month", "version", "grades", "w1", "w2"]

    oof = np.zeros(len(train))
    pte = np.zeros(len(test))
    ncpu = os.cpu_count() or 4
    params = dict(
        loss_function="RMSE",
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=20,
        od_type="Iter",
        od_wait=100,
        verbose=False,
        allow_writing_files=False,
        thread_count=max(1, ncpu // 2),
    )
    # single seed bag for probe speed but still quality
    seed = seeds[0]
    for fold, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(train, y)):
        # train corrector only on low-ratio rows in fold-train
        a_low = np.array([i for i in a if low[i]])
        if len(a_low) < 50:
            continue
        Xtr = train.iloc[a_low].reset_index(drop=True)
        ytr = resid[a_low]
        Xva = train.iloc[b].reset_index(drop=True)
        # build frames
        def pack(df):
            d = cond_r_ratio(df)
            out = d[feats + cats].copy()
            for c in cats:
                out[c] = out[c].astype(str)
            for c in feats:
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
            return out

        tr_df, va_df, te_df = pack(Xtr), pack(Xva), pack(test)
        from catboost import CatBoostRegressor

        m = CatBoostRegressor(**dict(params, random_seed=seed + fold))
        m.fit(tr_df, ytr, eval_set=(va_df, resid[b]), cat_features=cats, use_best_model=True)
        oof[b] = m.predict(va_df)
        pte += m.predict(te_df) / 5
        print(f"  exp3 f{fold} best={m.get_best_iteration()} n_low={len(a_low)}", flush=True)

    # corrector as ranking signal: fused + oof residual
    corr_signal = fused + oof
    # evaluate as arm vs label
    auc = float(roc_auc_score(y, corr_signal))
    # also raw residual auc (should be weak)
    auc_raw = float(roc_auc_score(y, oof))
    mo8 = np.load(ART / "merger_ord8.npz")["oof"]
    corr_mo = float(spearmanr(corr_signal, mo8).correlation)
    gate = {"auc_gt_0.690": auc > 0.690, "corr_mo8_lt_0.88": corr_mo < 0.88, "raw_auc_gt_0.55": auc_raw > 0.55}
    admit = gate["auc_gt_0.690"] and gate["corr_mo8_lt_0.88"] and gate["raw_auc_gt_0.55"]
    np.savez(ART / "probe_exp3.npz", oof=corr_signal, test_pred=rk(np.maximum.reduce([
        rk(np.load(ART / "merger_ord8.npz")["test_pred"]),
        rk(np.load(ART / "v2_cat_alt8.npz")["test_pred"]),
        rk(np.load(ART / "ord_noxb_bag.npz")["test_pred"]),
    ])) + pte, residual_oof=oof, residual_test=pte)
    # fix test: need consistent - save residual separately; fusion handled later
    np.savez(ART / "probe_exp3_resid.npz", oof=oof, test_pred=pte)
    report = {
        "mode": "exp3",
        "corrected_auc": auc,
        "residual_auc": auc_raw,
        "corr_spearman_mo8": corr_mo,
        "gate": gate,
        "admit_to_max": admit,
        "q10_ratio": q10,
    }
    (OUT / "report_exp3.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", default=["exp1", "exp2", "exp3"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[2900, 2901, 2902, 2903])
    args = ap.parse_args()
    results = {}
    for e in args.exps:
        if e == "exp3":
            results[e] = run_exp3_lowratio_corrector(args.seeds)
        else:
            results[e] = run_probe(e, args.seeds)
    (OUT / "summary.json").write_text(json.dumps(results, indent=2))
    admitted = [k for k, v in results.items() if v.get("admit_to_max")]
    print("ADMITTED", admitted)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""best_v1 dual-encoding arms from 714/最新参考 (claimed LB 0.71464).

Faithful port of explore_best.py with:
  - data paths fixed to data/train.csv, data/test.csv
  - edges / freq fit on TRAIN only (no test leakage)
  - fold-local rebuild optional via --fold-local-edges
  - incremental seed parts under artifacts/beat_max3/best_v1/

Fusion: max(rank(main), rank(alt)) → submission_best_v1.csv
Also screens vs v4_max3 for ship gate.
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
from catboost import CatBoostRegressor, Pool
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ART = ROOT / "artifacts" / "beat_max3" / "best_v1"
SUB = ROOT / "submissions"

BIN_COLS = ["t1", "t2", "r1", "r2", "c1", "c2", "w1", "w2"]
GRADE_MAP = {"s": 1, "ss": 2, "sss": 3}
DAYS_FX = np.array([700, 2500, 5000, 7000, 9000, 10000], dtype=float)
QUANTS = (5, 10, 20, 40)
ALT_QUANTS = (7, 13, 25)


def _qbins(v, e):
    return np.digitize(np.asarray(v, dtype=float), e)


def fit_edges_main(df: pd.DataFrame) -> dict:
    scale = df.groupby("source")["condition"].median()
    cond = pd.to_numeric(df["condition"], errors="coerce")
    days = pd.to_numeric(df["days"], errors="coerce")
    cond_r = (cond / df["source"].map(scale)).fillna(1.0)
    ratio = days / cond_r.clip(lower=1e-9)
    edges = {"__scale__": scale}
    for n in QUANTS:
        qs = np.linspace(0, 1, n + 1)[1:-1]
        edges[f"d_{n}"] = np.quantile(days.dropna(), qs)
        edges[f"c_{n}"] = np.quantile(cond.dropna(), qs)
        edges[f"cr_{n}"] = np.quantile(cond_r, qs)
        edges[f"ra_{n}"] = np.quantile(ratio, qs)
    return edges


def build_main(df: pd.DataFrame, edges: dict, freq_ref: pd.DataFrame | None = None):
    out = pd.DataFrame(index=df.index)
    days = pd.to_numeric(df["days"], errors="coerce")
    cond = pd.to_numeric(df["condition"], errors="coerce")
    scale = edges["__scale__"]
    cond_r = (cond / df["source"].map(scale)).fillna(1.0)
    ratio = days / cond_r.clip(lower=1e-9)
    out["days"] = days
    out["log_days"] = np.log1p(days.clip(lower=0))
    out["condition"] = cond
    out["log_condition"] = np.log1p(cond.clip(lower=0))
    out["condition_missing"] = cond.isna().astype(int)
    out["cond_r"] = cond_r.astype(float)
    out["log_cond_r"] = np.log(cond_r.clip(lower=1e-9))
    out["ratio"] = ratio.astype(float)
    out["log_ratio"] = np.log(ratio.clip(lower=1e-9))
    out["ratio_p75"] = (days / cond_r.clip(lower=1e-9) ** 0.75).astype(float)
    out["cond_x_days"] = (cond * days).astype(float)
    out["cond_over_days"] = (cond / (days.abs() + 1.0)).astype(float)
    out["age_range"] = df["age_range"].astype(float)
    out["days_over_age"] = (days / df["age_range"].astype(float)).astype(float)
    out["grade_ord"] = df["grades"].map(GRADE_MAP).astype(float)
    for c in BIN_COLS:
        out[c] = df[c].astype(int)
    out["bin_sum"] = out[BIN_COLS].sum(axis=1)
    cats = []
    out["region"] = df["region"].astype(str)
    out["source"] = df["source"].astype(str)
    out["month"] = df["month"].astype(str)
    out["version"] = df["version"].astype(str)
    out["grades_c"] = df["grades"].astype(str)
    out["age_cat"] = df["age_range"].astype(str)
    out["bin_pat"] = out[BIN_COLS].astype(str).agg("".join, axis=1)
    out["days_fx"] = np.digitize(days.to_numpy(dtype=float), DAYS_FX).astype(str)
    cats += ["region", "source", "month", "version", "grades_c", "age_cat", "bin_pat", "days_fx"]
    for n in QUANTS:
        out[f"d{n}"] = _qbins(days, edges[f"d_{n}"]).astype(str)
        out[f"r{n}"] = _qbins(ratio, edges[f"ra_{n}"]).astype(str)
        cats += [f"d{n}", f"r{n}"]
    for n in (5, 10, 20):
        out[f"c{n}"] = _qbins(cond.fillna(-1), edges[f"c_{n}"]).astype(str)
        out[f"cr{n}"] = _qbins(cond_r, edges[f"cr_{n}"]).astype(str)
        cats += [f"c{n}", f"cr{n}"]

    def cross(n, *p):
        s = out[p[0]].astype(str)
        for x in p[1:]:
            s = s + "|" + out[x].astype(str)
        out[n] = s
        cats.append(n)

    cross("rs", "region", "source")
    cross("d10r", "d10", "region")
    cross("d10s", "d10", "source")
    cross("d20r", "d20", "region")
    cross("d20s", "d20", "source")
    cross("d10a", "d10", "age_cat")
    cross("d10c10", "d10", "c10")
    cross("c10r", "c10", "region")
    cross("c10s", "c10", "source")
    cross("ra", "region", "age_cat")
    cross("sa", "source", "age_cat")
    cross("d10p", "d10", "bin_pat")
    cross("rp", "region", "bin_pat")
    cross("d5rs", "d5", "region", "source")
    cross("r10r", "r10", "region")
    cross("r10s", "r10", "source")
    cross("r10a", "r10", "age_cat")
    cross("r20r", "r20", "region")
    cross("r10p", "r10", "bin_pat")
    cross("cr10r", "cr10", "region")
    cross("cr10a", "cr10", "age_cat")
    cross("c5s", "c5", "source")
    cross("c20s", "c20", "source")
    cross("cr5s", "cr5", "source")
    cross("cr10s", "cr10", "source")
    cross("cr20s", "cr20", "source")
    cross("cr5r", "cr5", "region")
    cross("cr20r", "cr20", "region")
    cross("c5r", "c5", "region")
    cross("d5c5", "d5", "c5")
    cross("d20c20", "d20", "c20")
    cross("d5cr5", "d5", "cr5")
    cross("d10cr10", "d10", "cr10")
    cross("d10c10r", "d10", "c10", "region")
    cross("d10c10s", "d10", "c10", "source")
    cross("d10c10a", "d10", "c10", "age_cat")
    cross("sc10a", "source", "c10", "age_cat")
    cross("rc10a", "region", "c10", "age_cat")
    cross("rsa", "region", "source", "age_cat")
    cross("dfs", "days_fx", "source")
    cross("dfc10", "days_fx", "c10")
    cross("dfcr10", "days_fx", "cr10")
    cross("dfr", "days_fx", "region")
    cross("r5rs", "r5", "region", "source")

    ref = freq_ref if freq_ref is not None else out
    for c in ("region", "source", "bin_pat", "rs", "d10r", "c10s", "month", "version"):
        vc = ref[c].value_counts()
        out[f"f_{c}"] = out[c].map(vc).fillna(1).astype(float)

    out["x19c"] = df["x19"].astype(str)
    out["x20c"] = df["x20"].astype(str)
    out["lvc"] = df["livability"].astype(str)
    out["t3c"] = df["t3"].astype(str)
    out["cdc"] = df["code"].astype(str)
    cats += ["x19c", "x20c", "lvc", "t3c", "cdc"]
    out["cc"] = df["cc"].astype(float)
    out["max_g"] = df["max_g"].astype(float)
    out["V"] = df["V"].astype(float)
    cross("x20s", "x20c", "source")
    cross("x20r", "x20c", "region")
    cross("x20a", "x20c", "age_cat")
    cross("x19l", "x19c", "lvc")
    cross("lva", "lvc", "age_cat")
    cross("rl", "region", "lvc")
    cross("t3d5", "t3c", "d5")
    cross("sx20a", "source", "x20c", "age_cat")
    cross("rx20a", "region", "x20c", "age_cat")
    cross("rsx19", "region", "source", "x19c")
    return out, cats


def fit_edges_alt(df: pd.DataFrame) -> dict:
    rk = df.groupby("source")["condition"].rank(pct=True).fillna(0.5)
    days = pd.to_numeric(df["days"], errors="coerce")
    rate = days * (1.0 - rk)
    edges = {}
    for n in ALT_QUANTS:
        qs = np.linspace(0, 1, n + 1)[1:-1]
        edges[f"d_{n}"] = np.quantile(days.dropna(), qs)
        edges[f"k_{n}"] = np.quantile(rk, qs)
        edges[f"e_{n}"] = np.quantile(rate, qs)
    return edges


def build_alt(df: pd.DataFrame, edges: dict, freq_ref: pd.DataFrame | None = None):
    out = pd.DataFrame(index=df.index)
    days = pd.to_numeric(df["days"], errors="coerce")
    cond = pd.to_numeric(df["condition"], errors="coerce")
    rk = df.groupby("source")["condition"].rank(pct=True).fillna(0.5)
    rate = days * (1.0 - rk)
    out["days"] = days
    out["sqrt_days"] = np.sqrt(days.clip(lower=0))
    out["condition"] = cond
    out["cond_rk"] = rk
    out["rate"] = rate
    out["log_rate"] = np.log1p(rate.clip(lower=0))
    out["rate_over_age"] = rate / df["age_range"].astype(float)
    out["condition_missing"] = cond.isna().astype(int)
    out["age_range"] = df["age_range"].astype(float)
    out["grade_ord"] = df["grades"].map(GRADE_MAP).astype(float)
    for c in BIN_COLS:
        out[c] = df[c].astype(int)
    out["bin_sum"] = out[BIN_COLS].sum(axis=1)
    cats = []
    out["region"] = df["region"].astype(str)
    out["source"] = df["source"].astype(str)
    out["month"] = df["month"].astype(str)
    out["version"] = df["version"].astype(str)
    out["grades_c"] = df["grades"].astype(str)
    out["age_cat"] = df["age_range"].astype(str)
    out["bin_pat"] = out[BIN_COLS].astype(str).agg("".join, axis=1)
    cats += ["region", "source", "month", "version", "grades_c", "age_cat", "bin_pat"]
    for n in ALT_QUANTS:
        out[f"d{n}"] = _qbins(days, edges[f"d_{n}"]).astype(str)
        out[f"k{n}"] = _qbins(rk, edges[f"k_{n}"]).astype(str)
        out[f"e{n}"] = _qbins(rate, edges[f"e_{n}"]).astype(str)
        cats += [f"d{n}", f"k{n}", f"e{n}"]

    def cross(n, *p):
        s = out[p[0]].astype(str)
        for x in p[1:]:
            s = s + "|" + out[x].astype(str)
        out[n] = s
        cats.append(n)

    cross("Ak7s", "k7", "source")
    cross("Ak13s", "k13", "source")
    cross("Ak25s", "k25", "source")
    cross("Ak13r", "k13", "region")
    cross("Ak7a", "k7", "age_cat")
    cross("Ad13r", "d13", "region")
    cross("Ad13s", "d13", "source")
    cross("Ad7a", "d7", "age_cat")
    cross("Ad25r", "d25", "region")
    cross("Ae13r", "e13", "region")
    cross("Ae13s", "e13", "source")
    cross("Ae7a", "e7", "age_cat")
    cross("Ae7p", "e7", "bin_pat")
    cross("Ad7k7", "d7", "k7")
    cross("Ad13k13", "d13", "k13")
    cross("Ars", "region", "source")
    cross("Ara", "region", "age_cat")
    cross("Asa", "source", "age_cat")
    cross("Ad7rs", "d7", "region", "source")
    cross("Ak7ra", "k7", "region", "age_cat")
    cross("Ae7rs", "e7", "region", "source")
    cross("Ad7p", "d7", "bin_pat")
    cross("Arp", "region", "bin_pat")

    ref = freq_ref if freq_ref is not None else out
    for c in ("region", "source", "bin_pat", "Ars", "Ak13s", "Ad13r"):
        vc = ref[c].value_counts()
        out[f"f_{c}"] = out[c].map(vc).fillna(1).astype(float)

    out["x19c"] = df["x19"].astype(str)
    out["x20c"] = df["x20"].astype(str)
    out["lvc"] = df["livability"].astype(str)
    out["t3c"] = df["t3"].astype(str)
    out["cdc"] = df["code"].astype(str)
    cats += ["x19c", "x20c", "lvc", "t3c", "cdc"]
    cross("x20s", "x20c", "source")
    cross("x20r", "x20c", "region")
    cross("x20a", "x20c", "age_cat")
    cross("rl", "region", "lvc")
    return out, cats


def nested_auc(oof, y, n_blocks=5):
    out = np.zeros(len(y))
    for b in np.array_split(np.arange(len(y)), n_blocks):
        out[b] = rankdata(oof[b]) / len(b)
    return float(roc_auc_score(y, out))


def run_arm(
    arm_name: str,
    build_fn,
    fit_edges_fn,
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    seeds: list[int],
    bag_seeds: tuple[int, ...],
    ordered: bool,
    depth: int,
    iter_cnt: int,
    l2: float,
    n_splits: int,
    threads: int,
    fold_local_edges: bool,
):
    ART.mkdir(parents=True, exist_ok=True)
    oof_seeds, te_parts, per = [], [], []
    raw_tr = train.drop(columns=["label"])
    # global edges on train only (default; matches recipe spirit without test leak)
    edges_global = fit_edges_fn(raw_tr)
    if not fold_local_edges:
        # build once with train-freq
        Xtr_tmp, cats = build_fn(raw_tr, edges_global)
        Xte_tmp, _ = build_fn(test, edges_global, freq_ref=Xtr_tmp)
        for c in cats:
            Xtr_tmp[c] = Xtr_tmp[c].astype(str)
            Xte_tmp[c] = Xte_tmp[c].astype(str)
        Xtr_g, Xte_g, cats_g = Xtr_tmp, Xte_tmp, cats

    for seed in seeds:
        part = ART / f"part_{arm_name}_s{seed}.npz"
        if part.exists():
            d = np.load(part)
            oof_r = d["oof_rank"]
            te_r = d["test_rank"]
            auc = float(d["auc"])
            print(f"[resume] {arm_name} s{seed} {auc:.5f}", flush=True)
        else:
            t0 = time.time()
            skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
            oof = np.zeros(len(y))
            te_seed = np.zeros(len(test))
            for f, (tri, vali) in enumerate(skf.split(raw_tr, y)):
                if fold_local_edges:
                    edges = fit_edges_fn(raw_tr.iloc[tri])
                    Xtr_all, cats = build_fn(raw_tr, edges)
                    Xte_all, _ = build_fn(test, edges, freq_ref=Xtr_all.iloc[tri])
                    for c in cats:
                        Xtr_all[c] = Xtr_all[c].astype(str)
                        Xte_all[c] = Xte_all[c].astype(str)
                    Xtr_f = Xtr_all.iloc[tri]
                    Xva_f = Xtr_all.iloc[vali]
                    Xte_f = Xte_all
                else:
                    cats = cats_g
                    Xtr_f = Xtr_g.iloc[tri]
                    Xva_f = Xtr_g.iloc[vali]
                    Xte_f = Xte_g
                fold_te = np.zeros(len(test))
                for bs in bag_seeds:
                    kw = dict(
                        loss_function="RMSE",
                        eval_metric="RMSE",
                        iterations=iter_cnt,
                        learning_rate=0.03,
                        depth=depth,
                        l2_leaf_reg=l2,
                        random_strength=0.7,
                        verbose=0,
                        allow_writing_files=False,
                        thread_count=threads,
                        random_seed=(seed * 100 + bs),
                    )
                    if ordered:
                        kw["boosting_type"] = "Ordered"
                    m = CatBoostRegressor(**kw)
                    m.fit(Pool(Xtr_f, y[tri], cat_features=cats), verbose=False)
                    oof[vali] += m.predict(Xva_f)
                    fold_te += m.predict(Xte_f)
                oof[vali] /= len(bag_seeds)
                te_seed += fold_te / len(bag_seeds)
                print(
                    f"  [{arm_name}] s{seed} f{f} fold_auc={roc_auc_score(y[vali], oof[vali]):.5f}",
                    flush=True,
                )
            te_seed /= n_splits
            auc = float(roc_auc_score(y, oof))
            oof_r = rankdata(oof) / len(oof)
            te_r = rankdata(te_seed) / len(te_seed)
            np.savez(part, oof=oof, test_pred=te_seed, oof_rank=oof_r, test_rank=te_r, auc=auc)
            print(f"[{arm_name}] seed {seed}: OOF={auc:.5f} ({time.time()-t0:.0f}s)", flush=True)
        oof_seeds.append(oof_r)
        te_parts.append(te_r)
        per.append(auc)

    oof_pool = np.mean(np.vstack(oof_seeds), 0)
    te_pool = np.mean(np.vstack(te_parts), 0)
    return oof_pool, te_pool, float(roc_auc_score(y, oof_pool)), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--bags", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold-local-edges", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    if args.smoke:
        seeds = [2026]
        bags = (0,)
        folds = 2
    else:
        seeds = args.seeds or [2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033]
        bags = tuple(range(args.bags))
        folds = args.folds

    threads = args.threads or max(1, (os.cpu_count() or 4) // 2)
    ART.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["label"].astype(int).values
    tid = test["id"]

    print(f"=== best_v1 seeds={seeds} bags={bags} folds={folds} fold_local={args.fold_local_edges} ===", flush=True)
    t0 = time.time()

    o1, t1, a1, p1 = run_arm(
        "main",
        build_main,
        fit_edges_main,
        train,
        test,
        y,
        seeds,
        bags,
        True,
        5,
        800,
        10,
        folds,
        threads,
        args.fold_local_edges,
    )
    print(f"臂1 pooled OOF={a1:.5f} per={p1}", flush=True)

    o2, t2, a2, p2 = run_arm(
        "alt",
        build_alt,
        fit_edges_alt,
        train,
        test,
        y,
        seeds,
        bags,
        False,
        6,
        800,
        6,
        folds,
        threads,
        args.fold_local_edges,
    )
    print(f"臂2 pooled OOF={a2:.5f} per={p2}", flush=True)

    fuse_o = np.maximum(o1, o2)
    fuse_t = np.maximum(t1, t2)
    fuse_auc = float(roc_auc_score(y, fuse_o))
    fuse_nested = nested_auc(fuse_o, y)
    corr = float(spearmanr(o1, o2).correlation)
    print(f"max2 OOF={fuse_auc:.5f} nested={fuse_nested:.5f} corr={corr:.4f}", flush=True)

    np.savez(
        ART.parent / "best_v1.npz",
        oof=fuse_o,
        test_pred=fuse_t,
        main_oof=o1,
        alt_oof=o2,
        main_te=t1,
        alt_te=t2,
    )
    np.savez(ART.parent / "best_v1_main.npz", oof=o1, test_pred=t1)
    np.savez(ART.parent / "best_v1_alt.npz", oof=o2, test_pred=t2)

    pd.DataFrame({"id": tid, "label": np.clip(fuse_t, 0.001, 0.999)}).to_csv(
        SUB / "submission_best_v1.csv", index=False
    )
    pd.DataFrame({"id": tid, "label": np.clip(fuse_t, 0.001, 0.999)}).to_csv(
        ART.parent / "submission_best_v1.csv", index=False
    )

    # screen vs max3
    report = {
        "main_oof": a1,
        "alt_oof": a2,
        "fuse_oof": fuse_auc,
        "fuse_nested": fuse_nested,
        "corr_main_alt": corr,
        "seeds": seeds,
        "bags": list(bags),
        "folds": folds,
        "fold_local_edges": args.fold_local_edges,
        "elapsed_min": round((time.time() - t0) / 60, 2),
        "claimed_lb": 0.71464,
        "note": "714 best_v1 port; edges/freq on train-only",
    }
    if (ART.parent / "merger_ord8.npz").exists():
        mo = np.load(ART.parent / "merger_ord8.npz")["oof"]
        ca = np.load(ART.parent / "v2_cat_alt8.npz")["oof"]
        od = np.load(ART.parent / "ord_noxb_bag.npz")["oof"]

        def rk(a):
            return rankdata(np.asarray(a, float)) / len(a)

        base = np.maximum.reduce([rk(mo), rk(ca), rk(od)])
        base_n = nested_auc(base, y)
        meta = np.maximum(base, fuse_o)
        report["max3_nested"] = base_n
        report["best_v1_nested"] = fuse_nested
        report["delta_vs_max3_nested"] = fuse_nested - base_n
        report["meta_max_nested"] = nested_auc(meta, y)
        report["spearman_best_vs_max3"] = float(spearmanr(fuse_o, base).correlation)
        # also write meta submission
        mo_t = np.load(ART.parent / "merger_ord8.npz")["test_pred"]
        ca_t = np.load(ART.parent / "v2_cat_alt8.npz")["test_pred"]
        od_t = np.load(ART.parent / "ord_noxb_bag.npz")["test_pred"]
        base_t = np.maximum.reduce([rk(mo_t), rk(ca_t), rk(od_t)])
        meta_t = np.maximum(base_t, fuse_t)
        pd.DataFrame({"id": tid, "label": meta_t}).to_csv(SUB / "submission_max3_x_bestv1.csv", index=False)

    (ART.parent / "report_best_v1.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""SUPER714-Plus：在 best_v1 上做可提交差异化，并争取推高 AUC。

相对冠军 best_v1 的刻意差异（预测必不同）：
1. 跨世界连续特征：main 注入 rate；alt 注入 ratio/cond_r
2. alt 分箱改为 (6,12,24)
3. 10 seeds × 4 bags × 1000 trees（冠军为 8×3×800）
4. 种子起点 3100（冠军为 2026）
5. main depth=6（Ordered，冠军为 5）；alt depth=6 保持但分箱/特征已变

融合仍用 max(rank)。主提交写入 submissions/submission_super714_plus.csv。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from scipy.stats import pearsonr, rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]

BIN_COLS = ["t1", "t2", "r1", "r2", "c1", "c2", "w1", "w2"]
GRADE_MAP = {"s": 1, "ss": 2, "sss": 3}
DAYS_FX = np.array([700, 2500, 5000, 7000, 9000, 10000], dtype=float)
QUANTS = (5, 10, 20, 40)
ALT_QUANTS = (6, 12, 24)  # best_v1 为 (7,13,25)
SEEDS = tuple(range(3100, 3110))  # 10 seeds
BAG_SEEDS = (0, 1, 2, 3)  # 4 bags
N_SPLITS = 5
LR = 0.03
ITERATIONS = 1000


def resolve_data_dir(explicit: str | None = None) -> Path:
    for candidate in (explicit, os.environ.get("DATA_DIR"), str(ROOT / "data")):
        if not candidate:
            continue
        directory = Path(candidate).expanduser().resolve()
        if (directory / "train.csv").is_file() and (directory / "test.csv").is_file():
            return directory
    raise FileNotFoundError("找不到 train.csv/test.csv")


def _qbins(v, e):
    return np.digitize(np.asarray(v, dtype=float), e)


def fit_edges_main(df):
    scale = df.groupby("source")["condition"].median()
    cond = pd.to_numeric(df["condition"])
    days = pd.to_numeric(df["days"])
    cond_r = (cond / df["source"].map(scale)).fillna(1.0)
    ratio = days / cond_r.clip(lower=1e-9)
    rk = df.groupby("source")["condition"].rank(pct=True).fillna(0.5)
    rate = days * (1.0 - rk)
    edges = {"__scale__": scale}
    for n in QUANTS:
        qs = np.linspace(0, 1, n + 1)[1:-1]
        edges[f"d_{n}"] = np.quantile(days.dropna(), qs)
        edges[f"c_{n}"] = np.quantile(cond.dropna(), qs)
        edges[f"cr_{n}"] = np.quantile(cond_r, qs)
        edges[f"ra_{n}"] = np.quantile(ratio, qs)
    edges["__rate_ref__"] = True  # marker
    return edges


def build_main(df, edges):
    out = pd.DataFrame(index=df.index)
    days = pd.to_numeric(df["days"])
    cond = pd.to_numeric(df["condition"])
    scale = edges["__scale__"]
    cond_r = (cond / df["source"].map(scale)).fillna(1.0)
    ratio = days / cond_r.clip(lower=1e-9)
    rk = df.groupby("source")["condition"].rank(pct=True).fillna(0.5)
    rate = days * (1.0 - rk)
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
    # PLUS: 跨世界连续特征
    out["rate"] = rate.astype(float)
    out["log_rate"] = np.log1p(rate.clip(lower=0))
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
    for c in ("region", "source", "bin_pat", "rs", "d10r", "c10s", "month", "version"):
        out[f"f_{c}"] = out[c].map(out[c].value_counts()).astype(float)
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


def fit_edges_alt(df):
    rk = df.groupby("source")["condition"].rank(pct=True).fillna(0.5)
    days = pd.to_numeric(df["days"])
    rate = days * (1.0 - rk)
    scale = df.groupby("source")["condition"].median()
    cond = pd.to_numeric(df["condition"])
    cond_r = (cond / df["source"].map(scale)).fillna(1.0)
    ratio = days / cond_r.clip(lower=1e-9)
    edges = {"__scale__": scale}
    for n in ALT_QUANTS:
        qs = np.linspace(0, 1, n + 1)[1:-1]
        edges[f"d_{n}"] = np.quantile(days.dropna(), qs)
        edges[f"k_{n}"] = np.quantile(rk, qs)
        edges[f"e_{n}"] = np.quantile(rate, qs)
    return edges


def build_alt(df, edges):
    out = pd.DataFrame(index=df.index)
    days = pd.to_numeric(df["days"])
    cond = pd.to_numeric(df["condition"])
    rk = df.groupby("source")["condition"].rank(pct=True).fillna(0.5)
    rate = days * (1.0 - rk)
    scale = edges["__scale__"]
    cond_r = (cond / df["source"].map(scale)).fillna(1.0)
    ratio = days / cond_r.clip(lower=1e-9)
    out["days"] = days
    out["sqrt_days"] = np.sqrt(days.clip(lower=0))
    out["condition"] = cond
    out["cond_rk"] = rk
    out["rate"] = rate
    out["log_rate"] = np.log1p(rate.clip(lower=0))
    out["rate_over_age"] = rate / df["age_range"].astype(float)
    # PLUS: 跨世界
    out["cond_r"] = cond_r.astype(float)
    out["ratio"] = ratio.astype(float)
    out["log_ratio"] = np.log(ratio.clip(lower=1e-9))
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

    # 命名随新分箱
    cross("Ak6s", "k6", "source")
    cross("Ak12s", "k12", "source")
    cross("Ak24s", "k24", "source")
    cross("Ak12r", "k12", "region")
    cross("Ak6a", "k6", "age_cat")
    cross("Ad12r", "d12", "region")
    cross("Ad12s", "d12", "source")
    cross("Ad6a", "d6", "age_cat")
    cross("Ad24r", "d24", "region")
    cross("Ae12r", "e12", "region")
    cross("Ae12s", "e12", "source")
    cross("Ae6a", "e6", "age_cat")
    cross("Ae6p", "e6", "bin_pat")
    cross("Ad6k6", "d6", "k6")
    cross("Ad12k12", "d12", "k12")
    cross("Ars", "region", "source")
    cross("Ara", "region", "age_cat")
    cross("Asa", "source", "age_cat")
    cross("Ad6rs", "d6", "region", "source")
    cross("Ak6ra", "k6", "region", "age_cat")
    cross("Ae6rs", "e6", "region", "source")
    cross("Ad6p", "d6", "bin_pat")
    cross("Arp", "region", "bin_pat")
    for c in ("region", "source", "bin_pat", "Ars", "Ak12s", "Ad12r"):
        out[f"f_{c}"] = out[c].map(out[c].value_counts()).astype(float)
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


def run_arm(arm_name, build_fn, edges, train, test, y, ordered, depth, l2, seeds, n_splits, bag_seeds, iterations):
    raw_all = pd.concat([train.drop(columns=["label"]), test], ignore_index=True)
    x_all, cats = build_fn(raw_all, edges)
    for c in cats:
        x_all[c] = x_all[c].astype(str)
    xtr = x_all.iloc[: len(train)].reset_index(drop=True)
    xte = x_all.iloc[len(train) :].reset_index(drop=True)
    oof_seeds, te_parts = [], []
    for seed in seeds:
        t0 = time.time()
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
        oof = np.zeros(len(y))
        te_seed = np.zeros(len(test))
        for tri, vali in skf.split(xtr, y):
            fold_te = np.zeros(len(test))
            for bs in bag_seeds:
                kw = dict(
                    loss_function="RMSE",
                    eval_metric="RMSE",
                    iterations=iterations,
                    learning_rate=LR,
                    depth=depth,
                    l2_leaf_reg=l2,
                    random_strength=0.7,
                    verbose=0,
                    allow_writing_files=False,
                    thread_count=-1,
                    random_seed=(seed * 100 + bs),
                )
                if ordered:
                    kw["boosting_type"] = "Ordered"
                model = CatBoostRegressor(**kw)
                model.fit(Pool(xtr.iloc[tri], y[tri], cat_features=cats), verbose=False)
                oof[vali] += model.predict(xtr.iloc[vali])
                fold_te += model.predict(xte)
            oof[vali] /= len(bag_seeds)
            te_seed += fold_te / len(bag_seeds)
        te_seed /= n_splits
        oof_seeds.append(rankdata(oof) / len(oof))
        te_parts.append(rankdata(te_seed) / len(te_seed))
        print(
            f"  [{arm_name}] seed {seed}: OOF={roc_auc_score(y, oof):.5f} ({time.time() - t0:.0f}s)",
            flush=True,
        )
    oof_pool = np.mean(oof_seeds, axis=0)
    te_pool = np.mean(te_parts, axis=0)
    return oof_pool, te_pool, float(roc_auc_score(y, oof_pool))


def setup_log(path: Path) -> logging.Logger:
    logger = logging.getLogger("super714_plus")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for handler in (logging.FileHandler(path, mode="w", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="SUPER714-Plus 差异化训练")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--data-dir")
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    art = ROOT / "artifacts" / "super714_plus"
    sub = ROOT / "submissions"
    art.mkdir(parents=True, exist_ok=True)
    sub.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    log = setup_log(art / f"train{suffix}.log")

    seeds = (3100,) if args.smoke else SEEDS
    n_splits = 2 if args.smoke else N_SPLITS
    bag_seeds = (0,) if args.smoke else BAG_SEEDS
    iterations = 200 if args.smoke else ITERATIONS

    log.info(
        "=== SUPER714-Plus smoke=%s seeds=%s bags=%s iter=%s ===",
        args.smoke,
        list(seeds),
        list(bag_seeds),
        iterations,
    )
    t0 = time.time()
    train = pd.read_csv(data_dir / "train.csv", dtype={"id": str})
    test = pd.read_csv(data_dir / "test.csv", dtype={"id": str})
    sample = pd.read_csv(data_dir / "submit_sample.csv", dtype={"id": str})
    y = train["label"].astype(int).values
    raw_all = pd.concat([train.drop(columns=["label"]), test])
    edges_main = fit_edges_main(raw_all)
    edges_alt = fit_edges_alt(raw_all)

    log.info("--- main: +rate 特征, Ordered d6, l2=10 ---")
    o1, t1, a1 = run_arm(
        "main_plus", build_main, edges_main, train, test, y,
        True, 6, 10, seeds, n_splits, bag_seeds, iterations,
    )
    log.info("main_plus pooled OOF = %.5f", a1)
    # 臂级 checkpoint：主臂极慢，中断后可手工恢复
    np.save(art / f"plus_main{suffix}.npy", {"oof": o1, "test": t1, "auc": a1})
    log.info("checkpoint: %s", art / f"plus_main{suffix}.npy")

    log.info("--- alt: +ratio/cond_r, 分箱(6,12,24), Plain d6, l2=5 ---")
    o2, t2, a2 = run_arm(
        "alt_plus", build_alt, edges_alt, train, test, y,
        False, 6, 5, seeds, n_splits, bag_seeds, iterations,
    )
    log.info("alt_plus pooled OOF = %.5f", a2)
    np.save(art / f"plus_alt{suffix}.npy", {"oof": o2, "test": t2, "auc": a2})
    log.info("checkpoint: %s", art / f"plus_alt{suffix}.npy")

    fuse_oof = np.maximum(o1, o2)
    fuse_te = np.maximum(t1, t2)
    fuse_auc = float(roc_auc_score(y, fuse_oof))
    corr = float(pearsonr(o1, o2).statistic)
    log.info("max2 OOF = %.5f | corr=%.4f", fuse_auc, corr)

    # W62 预注册权重（best_v1 臂线上 0.71503）；另存 OOF 网格最优
    w62_oof = 0.62 * o1 + 0.38 * o2
    w62_te = 0.62 * t1 + 0.38 * t2
    w62_auc = float(roc_auc_score(y, w62_oof))
    best_w, best_auc = 0.62, w62_auc
    for w in np.round(np.arange(0.50, 0.801, 0.01), 2):
        auc_w = float(roc_auc_score(y, w * o1 + (1.0 - w) * o2))
        if auc_w > best_auc:
            best_auc, best_w = auc_w, float(w)
    best_te = best_w * t1 + (1.0 - best_w) * t2
    log.info("w62 OOF = %.5f | wbest(w=%.2f) OOF = %.5f", w62_auc, best_w, best_auc)

    # 对照冻结冠军
    champ_path = ROOT / "artifacts" / "super714" / "best_v1_oof.npy"
    champ_auc = None
    spear_test = None
    if champ_path.is_file():
        from scipy.stats import spearmanr

        champ = np.load(champ_path, allow_pickle=True).item()
        champ_auc = float(roc_auc_score(y, champ["fuse"]))
        champ_test = np.load(ROOT / "artifacts" / "super714" / "best_v1_test.npy", allow_pickle=True).item()
        spear_test = float(spearmanr(fuse_te, champ_test["fuse"]).statistic)
        log.info(
            "vs best_v1: ΔOOF=%+.5f (champ=%.5f) | test Spearman=%.4f",
            fuse_auc - champ_auc,
            champ_auc,
            spear_test,
        )

    if list(sample.columns) != ["id", "label"]:
        raise ValueError("submit_sample 列错误")

    def _write_sub(path: Path, scores: np.ndarray) -> str:
        submission = sample[["id"]].copy()
        submission["label"] = np.clip(scores, 0.001, 0.999)
        submission.to_csv(path, index=False)
        return sha256(path)

    out_sub = sub / f"submission_super714_plus{suffix}.csv"
    out_w62 = sub / f"submission_super714_plus_w62{suffix}.csv"
    out_best = sub / f"submission_super714_plus_wbest{suffix}.csv"
    sha_max2 = _write_sub(out_sub, fuse_te)
    sha_w62 = _write_sub(out_w62, w62_te)
    sha_best = _write_sub(out_best, best_te)

    np.save(art / f"plus_oof{suffix}.npy", {"main": o1, "alt": o2, "fuse": fuse_oof, "w62": w62_oof})
    np.save(art / f"plus_test{suffix}.npy", {"main": t1, "alt": t2, "fuse": fuse_te, "w62": w62_te})
    metrics = {
        "mode": "smoke" if args.smoke else "full",
        "recipe": {
            "seeds": list(seeds),
            "bags": list(bag_seeds),
            "iterations": iterations,
            "main": "Ordered d6 l2=10 +rate",
            "alt": "Plain d6 l2=5 +ratio/cond_r bins(6,12,24)",
            "fusion": "max2(rank) + w62(0.62/0.38) + oof-wbest",
        },
        "auc": {
            "main": a1,
            "alt": a2,
            "max2": fuse_auc,
            "w62": w62_auc,
            "wbest": best_auc,
            "wbest_w_main": best_w,
        },
        "pearson_main_alt": corr,
        "champ_max2_auc": champ_auc,
        "delta_vs_champ": (None if champ_auc is None else fuse_auc - champ_auc),
        "test_spearman_vs_champ": spear_test,
        "submission": str(out_sub.relative_to(ROOT)),
        "submission_sha256": sha_max2,
        "submission_w62": str(out_w62.relative_to(ROOT)),
        "submission_w62_sha256": sha_w62,
        "submission_wbest": str(out_best.relative_to(ROOT)),
        "submission_wbest_sha256": sha_best,
        "elapsed_minutes": (time.time() - t0) / 60,
        "lb_anchor": {"best_v1_max2": 0.71453, "best_v1_w62": 0.71503},
    }
    if champ_path.is_file():
        champ_sub = ROOT / "submissions" / "submission_super714.csv"
        if champ_sub.is_file():
            metrics["different_from_champ_submission"] = sha_max2 != sha256(champ_sub)
    (art / f"metrics{suffix}.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log.info("已保存 %s / %s / %s", out_sub, out_w62, out_best)
    log.info("总耗时 %.1f 分钟", (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

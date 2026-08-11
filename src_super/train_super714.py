"""SUPER714 训练入口。

基线严格保留 best_v1（线上 AUC 0.71453）：
臂1 cond_r + RMSE + Ordered/d5，臂2 rate + RMSE + Plain/d6，
每臂 8 seeds × 5 folds × 3 bags，最终融合 ``max2(rank)``。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from scipy.stats import rankdata, pearsonr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]


def resolve_data_dir(explicit: str | None = None) -> Path:
    """定位数据目录；优先 CLI，其次 ``DATA_DIR``，最后仓库 ``data/``。"""
    candidates = [
        explicit,
        os.environ.get("DATA_DIR"),
        str(ROOT / "data"),
        "/Volumes/pssd/app/ml/正式比赛/data",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        directory = Path(candidate).expanduser().resolve()
        if (directory / "train.csv").is_file() and (directory / "test.csv").is_file():
            return directory
    raise FileNotFoundError(
        "找不到 train.csv/test.csv；请设置 DATA_DIR 或传入 --data-dir。"
        f" 已尝试: {[item for item in candidates if item]}"
    )

BIN_COLS = ["t1","t2","r1","r2","c1","c2","w1","w2"]
GRADE_MAP = {"s":1,"ss":2,"sss":3}
DAYS_FX = np.array([700,2500,5000,7000,9000,10000], dtype=float)
QUANTS = (5,10,20,40); ALT_QUANTS = (7,13,25)

def _qbins(v,e): return np.digitize(np.asarray(v,dtype=float),e)

# ---- 臂1: cond_r世界 ----
def fit_edges_main(df):
    scale = df.groupby("source")["condition"].median()
    cond = pd.to_numeric(df["condition"]); days = pd.to_numeric(df["days"])
    cond_r = (cond/df["source"].map(scale)).fillna(1.0)
    ratio = days/cond_r.clip(lower=1e-9)
    edges = {"__scale__":scale}
    for n in QUANTS:
        qs = np.linspace(0,1,n+1)[1:-1]
        edges[f"d_{n}"]=np.quantile(days.dropna(),qs); edges[f"c_{n}"]=np.quantile(cond.dropna(),qs)
        edges[f"cr_{n}"]=np.quantile(cond_r,qs); edges[f"ra_{n}"]=np.quantile(ratio,qs)
    return edges

def build_main(df, edges):
    out=pd.DataFrame(index=df.index)
    days=pd.to_numeric(df["days"]); cond=pd.to_numeric(df["condition"])
    scale=edges["__scale__"]
    cond_r=(cond/df["source"].map(scale)).fillna(1.0); ratio=days/cond_r.clip(lower=1e-9)
    out["days"]=days; out["log_days"]=np.log1p(days.clip(lower=0))
    out["condition"]=cond; out["log_condition"]=np.log1p(cond.clip(lower=0))
    out["condition_missing"]=cond.isna().astype(int)
    out["cond_r"]=cond_r.astype(float); out["log_cond_r"]=np.log(cond_r.clip(lower=1e-9))
    out["ratio"]=ratio.astype(float); out["log_ratio"]=np.log(ratio.clip(lower=1e-9))
    out["ratio_p75"]=(days/cond_r.clip(lower=1e-9)**0.75).astype(float)
    out["cond_x_days"]=(cond*days).astype(float); out["cond_over_days"]=(cond/(days.abs()+1.0)).astype(float)
    out["age_range"]=df["age_range"].astype(float); out["days_over_age"]=(days/df["age_range"].astype(float)).astype(float)
    out["grade_ord"]=df["grades"].map(GRADE_MAP).astype(float)
    for c in BIN_COLS: out[c]=df[c].astype(int)
    out["bin_sum"]=out[BIN_COLS].sum(axis=1)
    cats=[]; out["region"]=df["region"].astype(str); out["source"]=df["source"].astype(str)
    out["month"]=df["month"].astype(str); out["version"]=df["version"].astype(str)
    out["grades_c"]=df["grades"].astype(str); out["age_cat"]=df["age_range"].astype(str)
    out["bin_pat"]=out[BIN_COLS].astype(str).agg("".join,axis=1)
    out["days_fx"]=np.digitize(days.to_numpy(dtype=float),DAYS_FX).astype(str)
    cats+=["region","source","month","version","grades_c","age_cat","bin_pat","days_fx"]
    for n in QUANTS:
        out[f"d{n}"]=_qbins(days,edges[f"d_{n}"]).astype(str)
        out[f"r{n}"]=_qbins(ratio,edges[f"ra_{n}"]).astype(str)
        cats+=[f"d{n}",f"r{n}"]
    for n in (5,10,20):
        out[f"c{n}"]=_qbins(cond.fillna(-1),edges[f"c_{n}"]).astype(str)
        out[f"cr{n}"]=_qbins(cond_r,edges[f"cr_{n}"]).astype(str)
        cats+=[f"c{n}",f"cr{n}"]
    def cross(n,*p):
        s=out[p[0]].astype(str)
        for x in p[1:]: s=s+"|"+out[x].astype(str)
        out[n]=s; cats.append(n)
    cross("rs","region","source"); cross("d10r","d10","region"); cross("d10s","d10","source")
    cross("d20r","d20","region"); cross("d20s","d20","source"); cross("d10a","d10","age_cat")
    cross("d10c10","d10","c10"); cross("c10r","c10","region"); cross("c10s","c10","source")
    cross("ra","region","age_cat"); cross("sa","source","age_cat")
    cross("d10p","d10","bin_pat"); cross("rp","region","bin_pat")
    cross("d5rs","d5","region","source"); cross("r10r","r10","region"); cross("r10s","r10","source")
    cross("r10a","r10","age_cat"); cross("r20r","r20","region"); cross("r10p","r10","bin_pat")
    cross("cr10r","cr10","region"); cross("cr10a","cr10","age_cat")
    cross("c5s","c5","source"); cross("c20s","c20","source")
    cross("cr5s","cr5","source"); cross("cr10s","cr10","source"); cross("cr20s","cr20","source")
    cross("cr5r","cr5","region"); cross("cr20r","cr20","region"); cross("c5r","c5","region")
    cross("d5c5","d5","c5"); cross("d20c20","d20","c20")
    cross("d5cr5","d5","cr5"); cross("d10cr10","d10","cr10")
    cross("d10c10r","d10","c10","region"); cross("d10c10s","d10","c10","source")
    cross("d10c10a","d10","c10","age_cat"); cross("sc10a","source","c10","age_cat")
    cross("rc10a","region","c10","age_cat"); cross("rsa","region","source","age_cat")
    cross("dfs","days_fx","source"); cross("dfc10","days_fx","c10")
    cross("dfcr10","days_fx","cr10"); cross("dfr","days_fx","region"); cross("r5rs","r5","region","source")
    for c in ("region","source","bin_pat","rs","d10r","c10s","month","version"):
        out[f"f_{c}"]=out[c].map(out[c].value_counts()).astype(float)
    out["x19c"]=df["x19"].astype(str); out["x20c"]=df["x20"].astype(str)
    out["lvc"]=df["livability"].astype(str); out["t3c"]=df["t3"].astype(str); out["cdc"]=df["code"].astype(str)
    cats+=["x19c","x20c","lvc","t3c","cdc"]
    out["cc"]=df["cc"].astype(float); out["max_g"]=df["max_g"].astype(float); out["V"]=df["V"].astype(float)
    cross("x20s","x20c","source"); cross("x20r","x20c","region"); cross("x20a","x20c","age_cat")
    cross("x19l","x19c","lvc"); cross("lva","lvc","age_cat"); cross("rl","region","lvc")
    cross("t3d5","t3c","d5"); cross("sx20a","source","x20c","age_cat")
    cross("rx20a","region","x20c","age_cat"); cross("rsx19","region","source","x19c")
    return out,cats

# ---- 臂2: rate世界 ----
def fit_edges_alt(df):
    rk = df.groupby("source")["condition"].rank(pct=True).fillna(0.5)
    days = pd.to_numeric(df["days"]); rate = days*(1.0-rk)
    edges = {}
    for n in ALT_QUANTS:
        qs = np.linspace(0,1,n+1)[1:-1]
        edges[f"d_{n}"]=np.quantile(days.dropna(),qs); edges[f"k_{n}"]=np.quantile(rk,qs)
        edges[f"e_{n}"]=np.quantile(rate,qs)
    return edges

def build_alt(df, edges):
    out=pd.DataFrame(index=df.index)
    days=pd.to_numeric(df["days"]); cond=pd.to_numeric(df["condition"])
    rk=df.groupby("source")["condition"].rank(pct=True).fillna(0.5); rate=days*(1.0-rk)
    out["days"]=days; out["sqrt_days"]=np.sqrt(days.clip(lower=0))
    out["condition"]=cond; out["cond_rk"]=rk; out["rate"]=rate
    out["log_rate"]=np.log1p(rate.clip(lower=0)); out["rate_over_age"]=rate/df["age_range"].astype(float)
    out["condition_missing"]=cond.isna().astype(int); out["age_range"]=df["age_range"].astype(float)
    out["grade_ord"]=df["grades"].map(GRADE_MAP).astype(float)
    for c in BIN_COLS: out[c]=df[c].astype(int)
    out["bin_sum"]=out[BIN_COLS].sum(axis=1)
    cats=[]; out["region"]=df["region"].astype(str); out["source"]=df["source"].astype(str)
    out["month"]=df["month"].astype(str); out["version"]=df["version"].astype(str)
    out["grades_c"]=df["grades"].astype(str); out["age_cat"]=df["age_range"].astype(str)
    out["bin_pat"]=out[BIN_COLS].astype(str).agg("".join,axis=1)
    cats+=["region","source","month","version","grades_c","age_cat","bin_pat"]
    for n in ALT_QUANTS:
        out[f"d{n}"]=_qbins(days,edges[f"d_{n}"]).astype(str)
        out[f"k{n}"]=_qbins(rk,edges[f"k_{n}"]).astype(str)
        out[f"e{n}"]=_qbins(rate,edges[f"e_{n}"]).astype(str)
        cats+=[f"d{n}",f"k{n}",f"e{n}"]
    def cross(n,*p):
        s=out[p[0]].astype(str)
        for x in p[1:]: s=s+"|"+out[x].astype(str); out[n]=s; cats.append(n)
    cross("Ak7s","k7","source"); cross("Ak13s","k13","source"); cross("Ak25s","k25","source")
    cross("Ak13r","k13","region"); cross("Ak7a","k7","age_cat")
    cross("Ad13r","d13","region"); cross("Ad13s","d13","source"); cross("Ad7a","d7","age_cat")
    cross("Ad25r","d25","region"); cross("Ae13r","e13","region"); cross("Ae13s","e13","source")
    cross("Ae7a","e7","age_cat"); cross("Ae7p","e7","bin_pat")
    cross("Ad7k7","d7","k7"); cross("Ad13k13","d13","k13")
    cross("Ars","region","source"); cross("Ara","region","age_cat"); cross("Asa","source","age_cat")
    cross("Ad7rs","d7","region","source"); cross("Ak7ra","k7","region","age_cat")
    cross("Ae7rs","e7","region","source"); cross("Ad7p","d7","bin_pat"); cross("Arp","region","bin_pat")
    for c in ("region","source","bin_pat","Ars","Ak13s","Ad13r"):
        out[f"f_{c}"]=out[c].map(out[c].value_counts()).astype(float)
    out["x19c"]=df["x19"].astype(str); out["x20c"]=df["x20"].astype(str)
    out["lvc"]=df["livability"].astype(str); out["t3c"]=df["t3"].astype(str); out["cdc"]=df["code"].astype(str)
    cats+=["x19c","x20c","lvc","t3c","cdc"]
    cross("x20s","x20c","source"); cross("x20r","x20c","region"); cross("x20a","x20c","age_cat")
    cross("rl","region","lvc")
    return out,cats

# ============================================================
SEEDS = (2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033)
N_SPLITS = 5
BAG_SEEDS = (0, 1, 2)
LR = 0.03


def run_arm(
    arm_name,
    build_fn,
    edges,
    train,
    test,
    y,
    ordered,
    depth,
    iter_cnt,
    l2,
    *,
    seeds=SEEDS,
    n_splits=N_SPLITS,
    bag_seeds=BAG_SEEDS,
):
    """训练一个 best_v1 臂并返回跨 seed 的 rank 平均。"""
    raw_all = pd.concat([train.drop(columns=["label"]), test], ignore_index=True)
    X_all, cats = build_fn(raw_all, edges)
    for c in cats: X_all[c] = X_all[c].astype(str)
    Xtr = X_all.iloc[:len(train)].reset_index(drop=True)
    Xte = X_all.iloc[len(train):].reset_index(drop=True)
    oof_seeds, te_parts = [], []
    for seed in seeds:
        t0 = time.time()
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
        oof = np.zeros(len(y)); te_seed = np.zeros(len(test))
        for f, (tri, vali) in enumerate(skf.split(Xtr, y)):
            fold_te = np.zeros(len(test))
            for bs in bag_seeds:
                kw = dict(loss_function="RMSE", eval_metric="RMSE",
                          iterations=iter_cnt, learning_rate=LR, depth=depth, l2_leaf_reg=l2,
                          random_strength=0.7, verbose=0, allow_writing_files=False,
                          thread_count=-1, random_seed=(seed*100+bs))
                if ordered: kw["boosting_type"] = "Ordered"
                m = CatBoostRegressor(**kw)
                m.fit(Pool(Xtr.iloc[tri], y[tri], cat_features=cats), verbose=False)
                oof[vali] += m.predict(Xtr.iloc[vali])
                fold_te += m.predict(Xte)
            oof[vali] /= len(bag_seeds); te_seed += fold_te / len(bag_seeds)
        te_seed /= n_splits
        oof_seeds.append(rankdata(oof)/len(oof))
        te_parts.append(rankdata(te_seed)/len(te_seed))
        a = roc_auc_score(y, oof)
        print(f"  [{arm_name}] seed {seed}: OOF={a:.5f} ({time.time()-t0:.0f}s)", flush=True)
    oof_pool = np.mean(oof_seeds, axis=0); te_pool = np.mean(te_parts, axis=0)
    return oof_pool, te_pool, float(roc_auc_score(y, oof_pool))


def setup_log(log_path: Path) -> logging.Logger:
    """同时写终端与文件，避免复用进程时重复 handler。"""
    logger = logging.getLogger("super714")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for handler in (
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 SUPER714/best_v1 双臂基线")
    parser.add_argument("--smoke", action="store_true", help="2 折×1 seed×1 bag 通路检查")
    parser.add_argument("--data-dir", help="含 train.csv/test.csv 的目录；覆盖 DATA_DIR")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = resolve_data_dir(args.data_dir)
    artifact_dir = ROOT / "artifacts" / "super714"
    submission_dir = ROOT / "submissions"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_smoke" if args.smoke else ""
    log = setup_log(artifact_dir / f"train{suffix}.log")
    seeds = (2026,) if args.smoke else SEEDS
    n_splits = 2 if args.smoke else N_SPLITS
    bag_seeds = (0,) if args.smoke else BAG_SEEDS

    log.info("=== SUPER714 best_v1 基线（线上锚点 0.71453）smoke=%s ===", args.smoke)
    t_start = time.time()
    train = pd.read_csv(data_dir / "train.csv", dtype={"id": str})
    test = pd.read_csv(data_dir / "test.csv", dtype={"id": str})
    y = train["label"].astype(int).values
    raw_all = pd.concat([train.drop(columns=["label"]), test])
    edges_main = fit_edges_main(raw_all); edges_alt = fit_edges_alt(raw_all)

    log.info("--- 臂1: cond_r + Ordered+d5+800+l2=10 ---")
    o1, t1, a1 = run_arm(
        "main", build_main, edges_main, train, test, y, True, 5, 800, 10,
        seeds=seeds, n_splits=n_splits, bag_seeds=bag_seeds,
    )
    log.info("  臂1 pooled OOF = %.5f", a1)

    log.info("--- 臂2: rate + Plain+d6+800+l2=6 ---")
    o2, t2, a2 = run_arm(
        "alt", build_alt, edges_alt, train, test, y, False, 6, 800, 6,
        seeds=seeds, n_splits=n_splits, bag_seeds=bag_seeds,
    )
    log.info("  臂2 pooled OOF = %.5f", a2)

    fuse_oof = np.maximum(o1, o2); fuse_te = np.maximum(t1, t2)
    fuse_auc = float(roc_auc_score(y, fuse_oof))
    correlation = float(pearsonr(o1, o2)[0])
    log.info("  max2融合 OOF = %.5f", fuse_auc)
    log.info("  (臂1=%.5f, 臂2=%.5f)", a1, a2)
    log.info("  corr=%.4f", correlation)

    sub = pd.DataFrame({"id":test["id"], "label":fuse_te.clip(0.001,0.999)})
    sub_path = submission_dir / f"submission_super714{suffix}.csv"
    sub.to_csv(sub_path, index=False)
    np.save(
        artifact_dir / f"trained_oof{suffix}.npy",
        {"main": o1, "alt": o2, "fuse": fuse_oof},
    )
    np.save(
        artifact_dir / f"trained_test{suffix}.npy",
        {"main": t1, "alt": t2, "fuse": fuse_te},
    )
    metrics = {
        "mode": "smoke" if args.smoke else "full",
        "data_dir": str(data_dir),
        "recipe": {
            "seeds": list(seeds),
            "n_splits": n_splits,
            "bag_seeds": list(bag_seeds),
            "iterations": 800,
        },
        "auc": {"main": a1, "alt": a2, "max2": fuse_auc},
        "pearson_main_alt": correlation,
        "selected_fusion": "max2",
        "submission": str(sub_path.relative_to(ROOT)),
    }
    (artifact_dir / f"metrics{suffix}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    log.info("已保存 %s", sub_path)
    log.info("DATA_DIR=%s", data_dir)
    log.info("总耗时: %.1f 分钟", (time.time() - t_start) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

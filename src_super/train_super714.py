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
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    from .features_te import FEATURE_NAME, build_source_days_te
except ImportError:  # 兼容 ``python src_super/train_super714.py``
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
    from features_te import FEATURE_NAME, build_source_days_te

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


def run_te_arm(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    edges: dict,
    *,
    seeds=SEEDS,
    n_splits=N_SPLITS,
    bag_seeds=BAG_SEEDS,
) -> tuple[np.ndarray, np.ndarray, float, list[float]]:
    """训练唯一候选臂：main 特征加双层诚实 source×days-bin TE。"""
    raw_train = train.drop(columns=["label"]).reset_index(drop=True)
    raw_test = test.reset_index(drop=True)
    raw_all = pd.concat([raw_train, raw_test], ignore_index=True)
    x_all, cat_features = build_main(raw_all, edges)
    for column in cat_features:
        x_all[column] = x_all[column].astype(str)
    x_train = x_all.iloc[: len(train)].reset_index(drop=True)
    x_test = x_all.iloc[len(train) :].reset_index(drop=True)

    oof_ranks: list[np.ndarray] = []
    test_ranks: list[np.ndarray] = []
    seed_aucs: list[float] = []
    for seed in seeds:
        started = time.time()
        splitter = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
        seed_oof = np.zeros(len(train), dtype=float)
        seed_test = np.zeros(len(test), dtype=float)
        for train_idx, valid_idx in splitter.split(x_train, y):
            te_fold = build_source_days_te(
                fit_frame=raw_train.iloc[train_idx],
                y_fit=y[train_idx],
                valid_frame=raw_train.iloc[valid_idx],
                other_frames=(raw_test,),
                n_bins=10,
                smoothing=20.0,
                inner_splits=4,
                inner_seed=seed,
            )
            x_fit = x_train.iloc[train_idx].copy()
            x_valid = x_train.iloc[valid_idx].copy()
            x_fold_test = x_test.copy()
            x_fit[FEATURE_NAME] = te_fold.fit.to_numpy()
            x_valid[FEATURE_NAME] = te_fold.valid.to_numpy()
            x_fold_test[FEATURE_NAME] = te_fold.others[0].to_numpy()

            fold_test = np.zeros(len(test), dtype=float)
            for bag_seed in bag_seeds:
                model = CatBoostRegressor(
                    loss_function="RMSE",
                    eval_metric="RMSE",
                    iterations=800,
                    learning_rate=LR,
                    depth=5,
                    l2_leaf_reg=10,
                    random_strength=0.7,
                    boosting_type="Ordered",
                    verbose=0,
                    allow_writing_files=False,
                    thread_count=-1,
                    random_seed=seed * 100 + bag_seed,
                )
                model.fit(
                    Pool(x_fit, y[train_idx], cat_features=cat_features),
                    verbose=False,
                )
                seed_oof[valid_idx] += model.predict(x_valid)
                fold_test += model.predict(x_fold_test)
            seed_oof[valid_idx] /= len(bag_seeds)
            seed_test += fold_test / len(bag_seeds)

        seed_test /= n_splits
        seed_auc = float(roc_auc_score(y, seed_oof))
        seed_aucs.append(seed_auc)
        oof_ranks.append(rankdata(seed_oof) / len(seed_oof))
        test_ranks.append(rankdata(seed_test) / len(seed_test))
        print(
            f"  [main_te] seed {seed}: OOF={seed_auc:.5f} "
            f"({time.time() - started:.0f}s)",
            flush=True,
        )

    pooled_oof = np.mean(oof_ranks, axis=0)
    pooled_test = np.mean(test_ranks, axis=0)
    pooled_auc = float(roc_auc_score(y, pooled_oof))
    return pooled_oof, pooled_test, pooled_auc, seed_aucs


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
    parser = argparse.ArgumentParser(description="训练 SUPER714 的折内 TE 候选臂")
    parser.add_argument("--smoke", action="store_true", help="2 折×1 seed×1 bag 通路检查")
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="重新训练 best_v1 双臂，不训练 TE；默认直接复用冻结基线",
    )
    parser.add_argument("--data-dir", help="含 train.csv/test.csv 的目录；覆盖 DATA_DIR")
    return parser.parse_args()


def load_frozen_baseline(
    y: np.ndarray,
    test_rows: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]:
    """读取并复核不可变 best_v1 基座。"""
    artifact_dir = ROOT / "artifacts" / "super714"
    oof = np.load(artifact_dir / "best_v1_oof.npy", allow_pickle=True).item()
    test_pred = np.load(artifact_dir / "best_v1_test.npy", allow_pickle=True).item()
    if set(oof) != {"main", "alt", "fuse"} or set(test_pred) != {"main", "alt", "fuse"}:
        raise ValueError("冻结 best_v1 产物键必须为 main/alt/fuse")
    if any(len(np.asarray(value)) != len(y) for value in oof.values()):
        raise ValueError("冻结 best_v1 OOF 长度错误")
    if any(len(np.asarray(value)) != test_rows for value in test_pred.values()):
        raise ValueError("冻结 best_v1 test 长度错误")

    scores = {
        key: float(roc_auc_score(y, np.asarray(oof[key])))
        for key in ("main", "alt", "fuse")
    }
    expected = {"main": 0.699917, "alt": 0.697704, "fuse": 0.701275}
    for key, expected_score in expected.items():
        if abs(scores[key] - expected_score) > 1e-5:
            raise ValueError(
                f"冻结基座 {key} AUC={scores[key]:.8f}，偏离 {expected_score:.6f}"
            )
    return oof, test_pred, scores


def save_submission(
    sample: pd.DataFrame,
    test: pd.DataFrame,
    prediction: np.ndarray,
    path: Path,
) -> None:
    """按官方样例行序写提交并执行硬校验。"""
    if list(sample.columns) != ["id", "label"]:
        raise ValueError("submit_sample.csv 列必须严格为 id,label")
    if not sample["id"].astype(str).reset_index(drop=True).equals(
        test["id"].astype(str).reset_index(drop=True)
    ):
        raise ValueError("submit_sample.csv 与 test.csv 的 id/行序不一致")
    values = np.asarray(prediction, dtype=float)
    if len(values) != len(sample) or not np.isfinite(values).all():
        raise ValueError("提交预测长度错误或包含非有限值")
    submission = sample[["id"]].copy()
    submission["label"] = np.clip(values, 0.001, 0.999)
    submission.to_csv(path, index=False)


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

    log.info(
        "=== SUPER714（冻结 best_v1 + 折内 TE）smoke=%s baseline_only=%s ===",
        args.smoke,
        args.baseline_only,
    )
    t_start = time.time()
    train = pd.read_csv(data_dir / "train.csv", dtype={"id": str})
    test = pd.read_csv(data_dir / "test.csv", dtype={"id": str})
    sample = pd.read_csv(data_dir / "submit_sample.csv", dtype={"id": str})
    y = train["label"].astype(int).values
    raw_all = pd.concat([train.drop(columns=["label"]), test])
    edges_main = fit_edges_main(raw_all)

    frozen_oof, frozen_test, frozen_scores = load_frozen_baseline(y, len(test))
    main_oof = np.asarray(frozen_oof["main"])
    alt_oof = np.asarray(frozen_oof["alt"])
    max2_oof = np.maximum(main_oof, alt_oof)
    main_test = np.asarray(frozen_test["main"])
    alt_test = np.asarray(frozen_test["alt"])
    max2_test = np.maximum(main_test, alt_test)
    base_spearman = float(spearmanr(main_oof, alt_oof).statistic)
    log.info(
        "冻结基座: main=%.6f alt=%.6f max2=%.6f Spearman=%.5f",
        frozen_scores["main"],
        frozen_scores["alt"],
        frozen_scores["fuse"],
        base_spearman,
    )

    recipe = {
        "seeds": list(seeds),
        "n_splits": n_splits,
        "bag_seeds": list(bag_seeds),
        "iterations": 800,
    }
    if args.baseline_only:
        edges_alt = fit_edges_alt(raw_all)
        log.info("--- 重训臂1: cond_r + Ordered+d5+800+l2=10 ---")
        trained_main_oof, trained_main_test, main_auc = run_arm(
            "main", build_main, edges_main, train, test, y, True, 5, 800, 10,
            seeds=seeds, n_splits=n_splits, bag_seeds=bag_seeds,
        )
        log.info("--- 重训臂2: rate + Plain+d6+800+l2=6 ---")
        trained_alt_oof, trained_alt_test, alt_auc = run_arm(
            "alt", build_alt, edges_alt, train, test, y, False, 6, 800, 6,
            seeds=seeds, n_splits=n_splits, bag_seeds=bag_seeds,
        )
        trained_max2_oof = np.maximum(trained_main_oof, trained_alt_oof)
        trained_max2_test = np.maximum(trained_main_test, trained_alt_test)
        max2_auc = float(roc_auc_score(y, trained_max2_oof))
        correlation = float(pearsonr(trained_main_oof, trained_alt_oof).statistic)
        np.save(
            artifact_dir / f"trained_oof{suffix}.npy",
            {
                "main": trained_main_oof,
                "alt": trained_alt_oof,
                "fuse": trained_max2_oof,
            },
        )
        np.save(
            artifact_dir / f"trained_test{suffix}.npy",
            {
                "main": trained_main_test,
                "alt": trained_alt_test,
                "fuse": trained_max2_test,
            },
        )
        submission_path = submission_dir / f"submission_super714_baseline{suffix}.csv"
        save_submission(sample, test, trained_max2_test, submission_path)
        metrics = {
            "mode": "baseline_smoke" if args.smoke else "baseline_full",
            "data_dir": str(data_dir),
            "recipe": recipe,
            "auc": {"main": main_auc, "alt": alt_auc, "max2": max2_auc},
            "pearson_main_alt": correlation,
            "selected_fusion": "max2",
            "submission": str(submission_path.relative_to(ROOT)),
        }
    else:
        log.info("--- 候选臂: main + 双层诚实 TE(source×days_bin10) ---")
        te_oof, te_test, te_auc, seed_aucs = run_te_arm(
            train,
            test,
            y,
            edges_main,
            seeds=seeds,
            n_splits=n_splits,
            bag_seeds=bag_seeds,
        )
        max3_oof = np.maximum.reduce([main_oof, alt_oof, te_oof])
        max3_test = np.maximum.reduce([main_test, alt_test, te_test])
        max3_auc = float(roc_auc_score(y, max3_oof))
        corr_main = float(spearmanr(te_oof, main_oof).statistic)
        gain = max3_auc - frozen_scores["fuse"]
        gates = {
            "te_auc_gt_0.697": te_auc > 0.697,
            "spearman_main_lt_0.90": corr_main < 0.90,
            "max3_gain_gt_0.001": gain > 0.001,
        }
        accepted = (not args.smoke) and all(gates.values())
        selected_fusion = "max3" if accepted else "max2"
        selected_test = max3_test if accepted else max2_test
        log.info(
            "TE=%.6f Spearman(TE,main)=%.5f max3=%.6f gain=%+.6f",
            te_auc,
            corr_main,
            max3_auc,
            gain,
        )
        log.info("门槛=%s；选择=%s", gates, selected_fusion)

        np.savez(
            artifact_dir / f"main_te{suffix}.npz",
            oof=te_oof,
            test_pred=te_test,
            per_seed=np.asarray(seed_aucs),
            seeds=np.asarray(seeds),
            y=y,
        )
        submission_path = submission_dir / f"submission_super714{suffix}.csv"
        save_submission(sample, test, selected_test, submission_path)
        if accepted:
            save_submission(
                sample,
                test,
                max3_test,
                submission_dir / "submission_super714_te_max3.csv",
            )
        metrics = {
            "mode": "te_smoke" if args.smoke else "te_full",
            "data_dir": str(data_dir),
            "recipe": recipe,
            "frozen_auc": frozen_scores,
            "frozen_main_alt_spearman": base_spearman,
            "te_seed_auc": seed_aucs,
            "te_auc": te_auc,
            "te_main_spearman": corr_main,
            "max3_auc": max3_auc,
            "max3_gain": gain,
            "gates": gates,
            "accepted": accepted,
            "selected_fusion": selected_fusion,
            "submission": str(submission_path.relative_to(ROOT)),
        }

    metrics = {
        **metrics,
        "elapsed_minutes": (time.time() - t_start) / 60,
    }
    (artifact_dir / f"metrics{suffix}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    log.info("已保存 %s", submission_path)
    log.info("DATA_DIR=%s", data_dir)
    log.info("总耗时: %.1f 分钟", (time.time() - t_start) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

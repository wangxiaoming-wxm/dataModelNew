"""vz20: 训练各"臂"并缓存 per-fold held-out 预测.

诚实协议 (held-out, 预注册权重, 无 outer-valid 调参):
  - 外层 StratifiedKFold, 使用全新 outer seed (未用于任何调参).
  - 每个外折: 所有特征统计只在 outer_train 上拟合 (label-free), 应用到 outer_valid / test.
  - 每个臂 = CatBoost RMSE, K seeds 的 rank 平均.
  - byte07 TE: 在 outer_train 上算 byte->rate, 映射到 outer_valid / test (无 label 泄漏).

臂定义:
  A1  : vz19 build_main, Ordered d5 l2=10 rsm=1.0   (vz19 arm1)
  A2  : vz19 build_alt,  Plain   d6 l2=6  rsm=0.3   (vz19 arm2, 已知强正则)
  REF1: vz19 build_main, Plain   d6 l2=6  rsm=1.0   (独立 ref 管线, 换 boosting/seed)
  REF2: vz19 build_alt,  Ordered d5 l2=10 rsm=1.0
  R1  : rich ratio_rich, Ordered d5 l2=10 rsm=1.0   (来自 rebuild V2, nested 证明 +0.006)
  R2  : rich rate_rich,  Plain   d6 l2=6  rsm=1.0

产物缓存到 artifacts/vz20/cache/{profile}_{arm}_fold{f}_{valid|test}.npy
以及 folds_{profile}.npz (valid 索引, y).
"""
from __future__ import annotations
import os, sys, time, argparse, importlib.util
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostRegressor, Pool

HERE = os.path.dirname(os.path.abspath(__file__))
VZ20 = os.path.dirname(HERE)
CACHE = os.path.join(VZ20, "artifacts", "vz20", "cache")
os.makedirs(CACHE, exist_ok=True)

# ---- 全新 outer seed: 未用于 vz19(2040-2047) 或 rebuild(2026) 的任何调参 ----
OUTER_SEED = 90210
BAG_SEEDS = [11, 12, 13, 14]          # my_cb / rich 臂
REF_BAG_SEEDS = [101, 102, 103, 104]  # ref 臂 (独立 seed)

# ---- 载入 vz19 特征模块 (features.py 就在本目录) ----
import features as vz19_feat  # noqa: E402

# ---- 载入 rebuild rich 特征模块 (ref_rebuild/features.py) ----
_rb_path = os.path.join(VZ20, "ref_rebuild", "features.py")
_spec = importlib.util.spec_from_file_location("rebuild_features", _rb_path)
rebuild_features = importlib.util.module_from_spec(_spec)
sys.modules["rebuild_features"] = rebuild_features
_spec.loader.exec_module(rebuild_features)
RebuildFeatureBuilder = rebuild_features.RebuildFeatureBuilder

ARMS = {
    "A1":   dict(world="main",       boosting="Ordered", depth=5, l2=10, rsm=1.0, ref=False),
    "A2":   dict(world="alt",        boosting="Plain",   depth=6, l2=6,  rsm=0.3, ref=False),
    "REF1": dict(world="main",       boosting="Plain",   depth=6, l2=6,  rsm=1.0, ref=True),
    "REF2": dict(world="alt",        boosting="Ordered", depth=5, l2=10, rsm=1.0, ref=True),
    "R1":   dict(world="ratio_rich", boosting="Ordered", depth=5, l2=10, rsm=1.0, ref=False),
    "R2":   dict(world="rate_rich",  boosting="Plain",   depth=6, l2=6,  rsm=1.0, ref=False),
    "R3":   dict(world="ratio_freq", boosting="Ordered", depth=5, l2=10, rsm=1.0, ref=False),
    "R4":   dict(world="rate_freq",  boosting="Plain",   depth=6, l2=6,  rsm=1.0, ref=False),
}


def id_byte(s, idx):
    try:
        return int(str(s)[2 * idx:2 * idx + 2], 16)
    except (ValueError, KeyError):
        return 0


def byte_te_map(keys_fit, y_fit, keys_apply):
    rate = pd.Series(y_fit).groupby(keys_fit).mean()
    return pd.Series(keys_apply).map(rate).fillna(float(np.mean(y_fit))).values


def build_world(world, fit_frame, apply_frame):
    """在 fit_frame 上拟合特征统计, 返回 (X_fit, X_apply, cats). 全部 label-free."""
    if world in ("main", "alt"):
        if world == "main":
            edges = vz19_feat.fit_edges_main(fit_frame)
            Xf, cats = vz19_feat.build_main(fit_frame, edges)
            Xa, _ = vz19_feat.build_main(apply_frame, edges)
        else:
            edges = vz19_feat.fit_edges_alt(fit_frame)
            Xf, cats = vz19_feat.build_alt(fit_frame, edges)
            Xa, _ = vz19_feat.build_alt(apply_frame, edges)
        for c in cats:
            Xf[c] = Xf[c].astype(str)
            Xa[c] = Xa[c].astype(str)
        return Xf, Xa, cats
    # rich worlds via RebuildFeatureBuilder
    builder = RebuildFeatureBuilder(world)
    fm_fit = builder.fit_transform(fit_frame)
    fm_app = builder.transform(apply_frame)
    return fm_fit.frame, fm_app.frame, list(fm_fit.cat_columns)


def train_arm_predict(world, cfg, Xf, cats, yf, Xv, Xt, seeds, trees, tc=-1):
    """K seeds rank 平均, 一次训练同时预测 valid 与 test, 返回 (rank_valid, rank_test)."""
    pv, pt = [], []
    pool_fit = Pool(Xf, yf, cat_features=cats)
    for sd in seeds:
        kw = dict(loss_function="RMSE", eval_metric="RMSE", iterations=trees,
                  learning_rate=0.03, depth=cfg["depth"], l2_leaf_reg=cfg["l2"],
                  random_strength=0.7, rsm=cfg["rsm"], one_hot_max_size=2,
                  verbose=0, allow_writing_files=False, thread_count=tc,
                  boosting_type=cfg["boosting"], random_seed=sd)
        m = CatBoostRegressor(**kw)
        m.fit(pool_fit, verbose=False)
        vv = m.predict(Xv); tt = m.predict(Xt)
        pv.append(rankdata(vv) / len(vv))
        pt.append(rankdata(tt) / len(tt))
    return np.mean(pv, axis=0), np.mean(pt, axis=0)


def run(profile, outer_splits, nseed, trees, arms, threads):
    tr = pd.read_csv(os.path.join(VZ20, "..", "data", "train.csv"), dtype={"id": str})
    te = pd.read_csv(os.path.join(VZ20, "..", "data", "test.csv"), dtype={"id": str})
    y = tr["label"].astype(int).values
    tr_raw = tr.drop(columns=["label"]).reset_index(drop=True)

    skf = StratifiedKFold(outer_splits, shuffle=True, random_state=OUTER_SEED)
    folds = list(skf.split(tr_raw, y))

    # 保存 fold 结构
    np.savez(os.path.join(CACHE, f"folds_{profile}.npz"),
             y=y, **{f"valid_{i}": vi for i, (_, vi) in enumerate(folds)})

    # byte07 缓存 (每折)
    b0 = np.array([id_byte(x, 0) for x in tr["id"]])
    b7 = np.array([id_byte(x, 7) for x in tr["id"]])
    b0_te_k = np.array([id_byte(x, 0) for x in te["id"]])
    b7_te_k = np.array([id_byte(x, 7) for x in te["id"]])
    for i, (tri, vali) in enumerate(folds):
        vpath = os.path.join(CACHE, f"{profile}_BYTE07_fold{i}_valid.npy")
        tpath = os.path.join(CACHE, f"{profile}_BYTE07_fold{i}_test.npy")
        if not (os.path.exists(vpath) and os.path.exists(tpath)):
            b0v = byte_te_map(b0[tri], y[tri], b0[vali])
            b7v = byte_te_map(b7[tri], y[tri], b7[vali])
            b07v = (rankdata(b0v) + rankdata(b7v)) / 2 / len(vali)
            b0t = byte_te_map(b0[tri], y[tri], b0_te_k)
            b7t = byte_te_map(b7[tri], y[tri], b7_te_k)
            b07t = (rankdata(b0t) + rankdata(b7t)) / 2 / len(te)
            np.save(vpath, b07v)
            np.save(tpath, b07t)

    for arm in arms:
        cfg = ARMS[arm]
        seeds = (REF_BAG_SEEDS if cfg["ref"] else BAG_SEEDS)[:nseed]
        for i, (tri, vali) in enumerate(folds):
            vpath = os.path.join(CACHE, f"{profile}_{arm}_fold{i}_valid.npy")
            tpath = os.path.join(CACHE, f"{profile}_{arm}_fold{i}_test.npy")
            if os.path.exists(vpath) and os.path.exists(tpath):
                vp = np.load(vpath)
                print(f"  skip {arm} fold{i}: valid AUC={roc_auc_score(y[vali], vp):.5f}", flush=True)
                continue
            t0 = time.time()
            fit_frame = tr_raw.iloc[tri].reset_index(drop=True)
            valid_frame = tr_raw.iloc[vali].reset_index(drop=True)
            Xf, Xv, cats = build_world(cfg["world"], fit_frame, valid_frame)
            _, Xt, _ = build_world(cfg["world"], fit_frame, te)
            vp, tp = train_arm_predict(cfg["world"], cfg, Xf, cats, y[tri], Xv, Xt, seeds, trees, threads)
            np.save(vpath, vp)
            np.save(tpath, tp)
            print(f"  {arm} fold{i}: valid AUC={roc_auc_score(y[vali], vp):.5f} "
                  f"({time.time()-t0:.0f}s, {len(seeds)}seed x{trees})", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="smoke")
    p.add_argument("--outer-splits", type=int, default=2)
    p.add_argument("--nseed", type=int, default=1)
    p.add_argument("--trees", type=int, default=500)
    p.add_argument("--arms", default="A1,A2,R1,R2")
    p.add_argument("--threads", type=int, default=-1)
    args = p.parse_args()
    arms = [a for a in args.arms.split(",") if a]
    print(f"=== vz20 arms: profile={args.profile} outer={args.outer_splits} "
          f"nseed={args.nseed} trees={args.trees} arms={arms} OUTER_SEED={OUTER_SEED} ===", flush=True)
    run(args.profile, args.outer_splits, args.nseed, args.trees, arms, args.threads)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""SUPER714-Jitter：best_v1 双臂 + 可控 jitter 视图（冲 +0.01 量级）。

动机：主办方冗余列（x19/x20/t3…）本质是对 source/condition/region 的噪声重编码，
曾带来约 +0.01 AUC。本方案在 best_v1 特征上显式加入 id-稳定的 jitter 分箱视图，
超参与 best_v1 完全一致（Ordered d5 / Plain d6，8×3×800），融合 AM40/W62/max2。

硬门禁：最优融合 OOF > 冻结 AM40（0.701811）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src_super"))
from jitter import add_jitter_views  # noqa: E402
from train_super714 import (  # noqa: E402
    BAG_SEEDS,
    LR,
    N_SPLITS,
    SEEDS,
    build_alt,
    build_main,
    fit_edges_alt,
    fit_edges_main,
    resolve_data_dir,
    run_arm,
)

AM40_OOF = 0.7018113510376338
W_MAIN, W_ALT = 0.62, 0.38
ALPHA_MAX = 0.40


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_sub(sample: pd.DataFrame, scores: np.ndarray, path: Path) -> str:
    out = sample[["id"]].copy()
    out["label"] = np.clip(np.asarray(scores, float), 0.001, 0.999)
    out.to_csv(path, index=False)
    return sha256(path)


def am40(main: np.ndarray, alt: np.ndarray) -> np.ndarray:
    linear = W_MAIN * main + W_ALT * alt
    return ALPHA_MAX * np.maximum(main, alt) + (1.0 - ALPHA_MAX) * linear


def build_main_jitter(df: pd.DataFrame, edges: dict):
    out, cats = build_main(df, edges)
    days = pd.to_numeric(df["days"])
    cond = pd.to_numeric(df["condition"])
    scale = edges["__scale__"]
    cond_r = (cond / df["source"].map(scale)).fillna(1.0)
    add_jitter_views(out, cats, df, cond_r, days, n_views=4, n_bins=10, n_sub=8, stream_offset=0)
    return out, cats


def build_alt_jitter(df: pd.DataFrame, edges: dict):
    out, cats = build_alt(df, edges)
    days = pd.to_numeric(df["days"])
    # alt 世界用 rate 轴做 jitter 的“条件”代理：用 cond_rk 映射到伪 cond_r 尺度
    rk = df.groupby("source")["condition"].rank(pct=True).fillna(0.5)
    # 用 (1-rk) 的尺度代替 cond_r，使 jitter 扰动落在 rate 相关轴上
    proxy = (1.0 - rk).clip(lower=1e-3)
    add_jitter_views(out, cats, df, proxy, days, n_views=4, n_bins=10, n_sub=8, stream_offset=1)
    return out, cats


def run_clf_arm(build_fn, edges, train, test, y, seeds, n_splits, bag_seeds, iterations):
    """同特征 Logloss 分类臂，制造与 RMSE 回归臂的排序分歧。"""
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
                model = CatBoostClassifier(
                    loss_function="Logloss",
                    eval_metric="AUC",
                    iterations=iterations,
                    learning_rate=LR,
                    depth=5,
                    l2_leaf_reg=10,
                    random_strength=0.7,
                    verbose=0,
                    allow_writing_files=False,
                    thread_count=-1,
                    random_seed=seed * 100 + bs + 17,
                    auto_class_weights="Balanced",
                )
                model.fit(Pool(xtr.iloc[tri], y[tri], cat_features=cats), verbose=False)
                oof[vali] += model.predict_proba(xtr.iloc[vali])[:, 1]
                fold_te += model.predict_proba(xte)[:, 1]
            oof[vali] /= len(bag_seeds)
            te_seed += fold_te / len(bag_seeds)
        te_seed /= n_splits
        oof_seeds.append(rankdata(oof) / len(oof))
        te_parts.append(rankdata(te_seed) / len(te_seed))
        print(f"  [clf_jitter] seed {seed}: OOF={roc_auc_score(y, oof):.5f} ({time.time()-t0:.0f}s)", flush=True)
    oof_pool = np.mean(oof_seeds, axis=0)
    te_pool = np.mean(te_parts, axis=0)
    return oof_pool, te_pool, float(roc_auc_score(y, oof_pool))


def main() -> int:
    parser = argparse.ArgumentParser(description="best_v1 + jitter 冲分训练")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-clf", action="store_true", help="跳过 Logloss 第三臂")
    parser.add_argument("--data-dir")
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    art = ROOT / "artifacts" / "super714_jitter"
    sub = ROOT / "submissions"
    art.mkdir(parents=True, exist_ok=True)
    sub.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    seeds = (2026,) if args.smoke else SEEDS
    n_splits = 2 if args.smoke else N_SPLITS
    bag_seeds = (0,) if args.smoke else BAG_SEEDS
    iterations = 200 if args.smoke else 800

    train = pd.read_csv(data_dir / "train.csv", dtype={"id": str})
    test = pd.read_csv(data_dir / "test.csv", dtype={"id": str})
    sample = pd.read_csv(data_dir / "submit_sample.csv", dtype={"id": str})
    y = train["label"].astype(int).to_numpy()
    raw_all = pd.concat([train.drop(columns=["label"]), test])

    print(
        f"=== SUPER714-Jitter smoke={args.smoke} seeds={list(seeds)} "
        f"bags={list(bag_seeds)} iter={iterations} clf={not args.no_clf} ===",
        flush=True,
    )
    t0 = time.time()
    edges_main = fit_edges_main(raw_all)
    edges_alt = fit_edges_alt(raw_all)

    print("--- main+jitter: Ordered d5 l2=10 ---", flush=True)
    o1, t1, a1 = run_arm(
        "main_jit", build_main_jitter, edges_main, train, test, y,
        True, 5, iterations, 10,
        seeds=seeds, n_splits=n_splits, bag_seeds=bag_seeds,
    )
    np.save(art / f"main{suffix}.npy", {"oof": o1, "test": t1, "auc": a1})
    print(f"main_jit pool OOF={a1:.5f}", flush=True)

    print("--- alt+jitter: Plain d6 l2=6 ---", flush=True)
    o2, t2, a2 = run_arm(
        "alt_jit", build_alt_jitter, edges_alt, train, test, y,
        False, 6, iterations, 6,
        seeds=seeds, n_splits=n_splits, bag_seeds=bag_seeds,
    )
    np.save(art / f"alt{suffix}.npy", {"oof": o2, "test": t2, "auc": a2})
    print(f"alt_jit pool OOF={a2:.5f}", flush=True)

    arms_o = {"main": o1, "alt": o2}
    arms_t = {"main": t1, "alt": t2}
    a3 = None
    if not args.no_clf:
        print("--- clf+jitter: Logloss d5 (diversity) ---", flush=True)
        o3, t3, a3 = run_clf_arm(
            build_main_jitter, edges_main, train, test, y,
            seeds, n_splits, bag_seeds, iterations,
        )
        np.save(art / f"clf{suffix}.npy", {"oof": o3, "test": t3, "auc": a3})
        arms_o["clf"] = o3
        arms_t["clf"] = t3
        print(
            f"clf_jit pool OOF={a3:.5f} spearman vs main={spearmanr(o3,o1).statistic:.4f}",
            flush=True,
        )

    cand_oof = {
        "max2": np.maximum(o1, o2),
        "w62": W_MAIN * o1 + W_ALT * o2,
        "am40": am40(o1, o2),
        "rankmean": 0.5 * o1 + 0.5 * o2,
    }
    cand_te = {
        "max2": np.maximum(t1, t2),
        "w62": W_MAIN * t1 + W_ALT * t2,
        "am40": am40(t1, t2),
        "rankmean": 0.5 * t1 + 0.5 * t2,
    }
    if a3 is not None:
        cand_oof["max3"] = np.maximum.reduce([o1, o2, o3])
        cand_te["max3"] = np.maximum.reduce([t1, t2, t3])
        for w in (0.10, 0.15, 0.20):
            name = f"am40_p_clf{int(w*100)}"
            cand_oof[name] = (1 - w) * am40(o1, o2) + w * o3
            cand_te[name] = (1 - w) * am40(t1, t2) + w * t3

    best_w, best_auc = W_MAIN, -1.0
    for w in np.round(np.arange(0.50, 0.801, 0.01), 2):
        auc = float(roc_auc_score(y, w * o1 + (1.0 - w) * o2))
        if auc > best_auc:
            best_auc, best_w = auc, float(w)
    cand_oof["wbest"] = best_w * o1 + (1.0 - best_w) * o2
    cand_te["wbest"] = best_w * t1 + (1.0 - best_w) * t2

    # 与冻结 AM40 再混合（jitter 臂若弱可回退）
    frozen = np.load(ROOT / "artifacts" / "super714" / "best_v1_oof.npy", allow_pickle=True).item()
    frozen_te = np.load(ROOT / "artifacts" / "super714" / "best_v1_test.npy", allow_pickle=True).item()
    fam40_o = am40(frozen["main"], frozen["alt"])
    fam40_t = am40(frozen_te["main"], frozen_te["alt"])
    for w in (0.3, 0.5, 0.7):
        name = f"mix_frozen_am40_{int(w*100)}"
        cand_oof[name] = w * fam40_o + (1 - w) * cand_oof["am40"]
        cand_te[name] = w * fam40_t + (1 - w) * cand_te["am40"]

    scores = {k: float(roc_auc_score(y, v)) for k, v in cand_oof.items()}
    champ = max(scores, key=scores.get)
    beat = scores[champ] > AM40_OOF + 1e-12

    paths = {}
    for name, te in cand_te.items():
        p = sub / f"submission_jitter_{name}{suffix}.csv"
        paths[name] = {
            "path": str(p.relative_to(ROOT)),
            "sha256": write_sub(sample, te, p),
            "oof_auc": scores[name],
        }

    beat_path = sub / f"submission_champion{suffix}.csv"
    if beat:
        shutil.copyfile(sub / f"submission_jitter_{champ}{suffix}.csv", beat_path)
        gate = "PASS"
    else:
        # 回退：仍写出最优 jitter 融合供分析，不覆盖为 champion
        shutil.copyfile(sub / f"submission_jitter_{champ}{suffix}.csv", sub / f"submission_jitter_best{suffix}.csv")
        gate = "FAIL"

    np.save(art / f"jitter_oof{suffix}.npy", {**arms_o, **cand_oof})
    np.save(art / f"jitter_test{suffix}.npy", {**arms_t, **cand_te})
    metrics = {
        "mode": "smoke" if args.smoke else "full",
        "recipe": {
            "base": "best_v1 dual world + controlled jitter views",
            "seeds": list(seeds),
            "bags": list(bag_seeds),
            "iterations": iterations,
            "main": "Ordered d5 l2=10 + jitter×3",
            "alt": "Plain d6 l2=6 + jitter×3",
            "clf": None if a3 is None else "Logloss d5 + jitter on main features",
        },
        "arm_oof": {"main": a1, "alt": a2, "clf": a3},
        "corr_main_alt": float(pearsonr(o1, o2).statistic),
        "fusions": paths,
        "champion": champ,
        "champion_oof": scores[champ],
        "frozen_am40_oof": AM40_OOF,
        "delta_vs_am40": scores[champ] - AM40_OOF,
        "gate_beat_am40": beat,
        "gate": gate,
        "wbest_w_main": best_w,
        "elapsed_minutes": (time.time() - t0) / 60,
    }
    (art / f"metrics{suffix}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "arm_oof": metrics["arm_oof"],
        "champion": champ,
        "champion_oof": scores[champ],
        "delta_vs_am40": scores[champ] - AM40_OOF,
        "gate": gate,
    }, indent=2), flush=True)
    for k, v in sorted(scores.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {v:.8f}  {k}", flush=True)
    print(f"elapsed {metrics['elapsed_minutes']:.1f} min | gate={gate}", flush=True)
    return 0 if beat or args.smoke else 2


if __name__ == "__main__":
    raise SystemExit(main())

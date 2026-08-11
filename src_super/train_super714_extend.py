#!/usr/bin/env python3
"""best_v1 同构扩展：冻结 8 seed 臂 + 新 seed 续训，再做 W62 加权。

动机：
- Plus（改 depth/分箱/跨世界）OOF 掉到 0.696，已拒绝
- W62 在冻结臂上线上 0.71503
- 下一步只加 seed 多样性，不改特征/超参/分箱
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src_super"))
from train_super714 import (  # noqa: E402
    BAG_SEEDS,
    N_SPLITS,
    build_alt,
    build_main,
    fit_edges_alt,
    fit_edges_main,
    resolve_data_dir,
    run_arm,
)

# 在 best_v1 的 2026–2033 之后续训
EXT_SEEDS = (2034, 2035, 2036, 2037)
N_FROZEN_SEEDS = 8
W62_MAIN, W62_ALT = 0.62, 0.38


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


def merge_pool(frozen: np.ndarray, new_parts: list[np.ndarray], n_frozen: int) -> np.ndarray:
    """frozen 已是 n_frozen 个 seed-rank 的均值；与新 seed-rank 再平均。"""
    total = n_frozen + len(new_parts)
    return (n_frozen * np.asarray(frozen, float) + np.sum(new_parts, axis=0)) / total


def main() -> int:
    parser = argparse.ArgumentParser(description="best_v1 同构 seed 扩展 + W62")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--data-dir")
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    art = ROOT / "artifacts" / "super714_extend"
    sub = ROOT / "submissions"
    art.mkdir(parents=True, exist_ok=True)
    sub.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""

    seeds = (2034,) if args.smoke else EXT_SEEDS
    n_splits = 2 if args.smoke else N_SPLITS
    bag_seeds = (0,) if args.smoke else BAG_SEEDS
    iterations = 200 if args.smoke else 800
    n_frozen = 1 if args.smoke else N_FROZEN_SEEDS

    train = pd.read_csv(data_dir / "train.csv", dtype={"id": str})
    test = pd.read_csv(data_dir / "test.csv", dtype={"id": str})
    sample = pd.read_csv(data_dir / "submit_sample.csv", dtype={"id": str})
    y = train["label"].astype(int).to_numpy()
    raw_all = pd.concat([train.drop(columns=["label"]), test])

    frozen_oof = np.load(ROOT / "artifacts" / "super714" / "best_v1_oof.npy", allow_pickle=True).item()
    frozen_te = np.load(ROOT / "artifacts" / "super714" / "best_v1_test.npy", allow_pickle=True).item()

    print(
        f"=== SUPER714-Extend smoke={args.smoke} new_seeds={list(seeds)} "
        f"bags={list(bag_seeds)} iter={iterations} ===",
        flush=True,
    )
    t0 = time.time()
    edges_main = fit_edges_main(raw_all)
    edges_alt = fit_edges_alt(raw_all)

    print("--- extend main: cond_r Ordered d5 l2=10 ---", flush=True)
    # run_arm 返回的是 new seeds 的 pool；我们需要逐 seed 以便与冻结均值正确合并。
    # 这里直接用 run_arm 得到 new pool，再按 seed 数做加权合并（new pool = mean(new seeds)）。
    o_new_main, t_new_main, a_new_main = run_arm(
        "main_ext", build_main, edges_main, train, test, y,
        True, 5, iterations, 10,
        seeds=seeds, n_splits=n_splits, bag_seeds=bag_seeds,
    )
    print(f"new main pool OOF={a_new_main:.5f}", flush=True)

    print("--- extend alt: rate Plain d6 l2=6 ---", flush=True)
    o_new_alt, t_new_alt, a_new_alt = run_arm(
        "alt_ext", build_alt, edges_alt, train, test, y,
        False, 6, iterations, 6,
        seeds=seeds, n_splits=n_splits, bag_seeds=bag_seeds,
    )
    print(f"new alt pool OOF={a_new_alt:.5f}", flush=True)

    n_new = len(seeds)
    main_o = (n_frozen * np.asarray(frozen_oof["main"], float) + n_new * o_new_main) / (n_frozen + n_new)
    alt_o = (n_frozen * np.asarray(frozen_oof["alt"], float) + n_new * o_new_alt) / (n_frozen + n_new)
    main_t = (n_frozen * np.asarray(frozen_te["main"], float) + n_new * t_new_main) / (n_frozen + n_new)
    alt_t = (n_frozen * np.asarray(frozen_te["alt"], float) + n_new * t_new_alt) / (n_frozen + n_new)

    max2_o = np.maximum(main_o, alt_o)
    max2_t = np.maximum(main_t, alt_t)
    w62_o = W62_MAIN * main_o + W62_ALT * alt_o
    w62_t = W62_MAIN * main_t + W62_ALT * alt_t

    best_w, best_auc = W62_MAIN, float(roc_auc_score(y, w62_o))
    for w in np.round(np.arange(0.50, 0.801, 0.01), 2):
        a = float(roc_auc_score(y, w * main_o + (1.0 - w) * alt_o))
        if a > best_auc:
            best_auc, best_w = a, float(w)
    best_t = best_w * main_t + (1.0 - best_w) * alt_t

    metrics = {
        "mode": "smoke" if args.smoke else "full",
        "recipe": {
            "frozen_seeds": n_frozen,
            "new_seeds": list(seeds),
            "bags": list(bag_seeds),
            "iterations": iterations,
            "main": "Ordered d5 l2=10 (best_v1 identical)",
            "alt": "Plain d6 l2=6 (best_v1 identical)",
            "merge": f"({n_frozen}*frozen + {n_new}*new_pool)/{n_frozen + n_new}",
        },
        "auc": {
            "new_main": a_new_main,
            "new_alt": a_new_alt,
            "merged_main": float(roc_auc_score(y, main_o)),
            "merged_alt": float(roc_auc_score(y, alt_o)),
            "merged_max2": float(roc_auc_score(y, max2_o)),
            "merged_w62": float(roc_auc_score(y, w62_o)),
            "merged_wbest": best_auc,
            "wbest_w_main": best_w,
            "frozen_max2": float(roc_auc_score(y, frozen_oof["fuse"])),
            "frozen_w62": float(
                roc_auc_score(y, W62_MAIN * frozen_oof["main"] + W62_ALT * frozen_oof["alt"])
            ),
        },
        "pearson_merged_main_alt": float(pearsonr(main_o, alt_o).statistic),
        "elapsed_minutes": (time.time() - t0) / 60,
    }
    metrics["auc"]["delta_w62_vs_frozen_w62"] = (
        metrics["auc"]["merged_w62"] - metrics["auc"]["frozen_w62"]
    )

    p_max2 = sub / f"submission_super714_extend{suffix}.csv"
    p_w62 = sub / f"submission_super714_extend_w62{suffix}.csv"
    p_best = sub / f"submission_super714_extend_wbest{suffix}.csv"
    metrics["submissions"] = {
        "max2": {"path": str(p_max2.relative_to(ROOT)), "sha256": write_sub(sample, max2_t, p_max2)},
        "w62": {"path": str(p_w62.relative_to(ROOT)), "sha256": write_sub(sample, w62_t, p_w62)},
        "wbest": {"path": str(p_best.relative_to(ROOT)), "sha256": write_sub(sample, best_t, p_best)},
    }
    champ = sub / "submission_w62.csv"
    if champ.is_file():
        other = pd.read_csv(champ)["label"].to_numpy(dtype=float)
        metrics["spearman_extend_w62_vs_submission_w62"] = float(spearmanr(w62_t, other).statistic)
        metrics["different_from_submission_w62"] = sha256(p_w62) != sha256(champ)

    np.save(
        art / f"extend_oof{suffix}.npy",
        {"main": main_o, "alt": alt_o, "fuse": max2_o, "w62": w62_o, "new_main": o_new_main, "new_alt": o_new_alt},
    )
    np.save(
        art / f"extend_test{suffix}.npy",
        {"main": main_t, "alt": alt_t, "fuse": max2_t, "w62": w62_t, "new_main": t_new_main, "new_alt": t_new_alt},
    )
    (art / f"metrics{suffix}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics["auc"], indent=2), flush=True)
    print(f"saved {p_max2.name} / {p_w62.name} / {p_best.name}", flush=True)
    print(f"elapsed {metrics['elapsed_minutes']:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

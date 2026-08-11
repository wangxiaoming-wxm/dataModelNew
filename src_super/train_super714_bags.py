#!/usr/bin/env python3
"""SUPER714-Bags：best_v1 同构，仅 bags 3→6，再融合并硬门禁超过冻结 W62。

协议与 best_v1 完全一致，只改 BAG_SEEDS=(0,1,2,3,4,5)。
融合候选：max2 / W62 / AM40 / OOF-wbest。
仅当最优候选 OOF > 冻结 W62 时，将 champion 软链/写出为 submission_beat_w62.csv。
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
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src_super"))
from train_super714 import (  # noqa: E402
    N_SPLITS,
    build_alt,
    build_main,
    fit_edges_alt,
    fit_edges_main,
    resolve_data_dir,
    run_arm,
)

SEEDS = (2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033)
BAG_SEEDS = (0, 1, 2, 3, 4, 5)
ITERATIONS = 800
W_MAIN, W_ALT = 0.62, 0.38
ALPHA_MAX = 0.40
W62_OOF = 0.7015936597140784


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


def main() -> int:
    parser = argparse.ArgumentParser(description="best_v1 同构 6-bag 重训")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--data-dir")
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    art = ROOT / "artifacts" / "super714_bags"
    sub = ROOT / "submissions"
    art.mkdir(parents=True, exist_ok=True)
    sub.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""

    seeds = (2026,) if args.smoke else SEEDS
    n_splits = 2 if args.smoke else N_SPLITS
    bag_seeds = (0, 1) if args.smoke else BAG_SEEDS
    iterations = 200 if args.smoke else ITERATIONS

    train = pd.read_csv(data_dir / "train.csv", dtype={"id": str})
    test = pd.read_csv(data_dir / "test.csv", dtype={"id": str})
    sample = pd.read_csv(data_dir / "submit_sample.csv", dtype={"id": str})
    y = train["label"].astype(int).to_numpy()
    raw_all = pd.concat([train.drop(columns=["label"]), test])

    print(
        f"=== SUPER714-Bags smoke={args.smoke} seeds={list(seeds)} "
        f"bags={list(bag_seeds)} iter={iterations} ===",
        flush=True,
    )
    t0 = time.time()
    edges_main = fit_edges_main(raw_all)
    edges_alt = fit_edges_alt(raw_all)

    print("--- main: cond_r Ordered d5 l2=10 | 6 bags ---", flush=True)
    o1, t1, a1 = run_arm(
        "main_bags", build_main, edges_main, train, test, y,
        True, 5, iterations, 10,
        seeds=seeds, n_splits=n_splits, bag_seeds=bag_seeds,
    )
    np.save(art / f"bags_main{suffix}.npy", {"oof": o1, "test": t1, "auc": a1})
    print(f"main pool OOF={a1:.5f}", flush=True)

    print("--- alt: rate Plain d6 l2=6 | 6 bags ---", flush=True)
    o2, t2, a2 = run_arm(
        "alt_bags", build_alt, edges_alt, train, test, y,
        False, 6, iterations, 6,
        seeds=seeds, n_splits=n_splits, bag_seeds=bag_seeds,
    )
    np.save(art / f"bags_alt{suffix}.npy", {"oof": o2, "test": t2, "auc": a2})
    print(f"alt pool OOF={a2:.5f}", flush=True)

    cand_oof = {
        "max2": np.maximum(o1, o2),
        "w62": W_MAIN * o1 + W_ALT * o2,
        "am40": am40(o1, o2),
    }
    cand_te = {
        "max2": np.maximum(t1, t2),
        "w62": W_MAIN * t1 + W_ALT * t2,
        "am40": am40(t1, t2),
    }
    best_w, best_auc = W_MAIN, -1.0
    for w in np.round(np.arange(0.50, 0.801, 0.01), 2):
        auc = float(roc_auc_score(y, w * o1 + (1.0 - w) * o2))
        if auc > best_auc:
            best_auc, best_w = auc, float(w)
    cand_oof["wbest"] = best_w * o1 + (1.0 - best_w) * o2
    cand_te["wbest"] = best_w * t1 + (1.0 - best_w) * t2

    scores = {k: float(roc_auc_score(y, v)) for k, v in cand_oof.items()}
    champion_name = max(scores, key=scores.get)
    champion_auc = scores[champion_name]
    frozen_w62 = W62_OOF
    # 也对照冻结臂 AM40
    frozen = np.load(ROOT / "artifacts" / "super714" / "best_v1_oof.npy", allow_pickle=True).item()
    frozen_am40 = float(roc_auc_score(y, am40(frozen["main"], frozen["alt"])))

    beat_frozen_w62 = champion_auc > frozen_w62 + 1e-12
    beat_frozen_am40 = champion_auc > frozen_am40 + 1e-12

    paths = {}
    for name, te in cand_te.items():
        paths[name] = {
            "path": f"submissions/submission_super714_bags_{name}{suffix}.csv",
            "sha256": write_sub(sample, te, sub / f"submission_super714_bags_{name}{suffix}.csv"),
            "oof_auc": scores[name],
        }

    champ_path = sub / f"submission_beat_w62{suffix}.csv"
    if beat_frozen_w62:
        src = sub / f"submission_super714_bags_{champion_name}{suffix}.csv"
        shutil.copyfile(src, champ_path)
        champ_sha = sha256(champ_path)
        gate = "PASS"
    else:
        # 不覆盖 champion；若存在旧文件则保留
        champ_sha = sha256(champ_path) if champ_path.is_file() else None
        gate = "FAIL"

    w62_sub = sub / "submission_w62.csv"
    spear = None
    if beat_frozen_w62 and w62_sub.is_file():
        spear = float(
            spearmanr(
                pd.read_csv(champ_path)["label"],
                pd.read_csv(w62_sub)["label"],
            ).statistic
        )

    np.save(art / f"bags_oof{suffix}.npy", {"main": o1, "alt": o2, **cand_oof})
    np.save(art / f"bags_test{suffix}.npy", {"main": t1, "alt": t2, **cand_te})
    metrics = {
        "mode": "smoke" if args.smoke else "full",
        "recipe": {
            "seeds": list(seeds),
            "bags": list(bag_seeds),
            "iterations": iterations,
            "main": "Ordered d5 l2=10 (best_v1 identical)",
            "alt": "Plain d6 l2=6 (best_v1 identical)",
            "change_vs_best_v1": "bags 3→6 only",
        },
        "arm_oof": {"main": a1, "alt": a2, "pearson": float(pearsonr(o1, o2).statistic)},
        "fusions": paths,
        "wbest_w_main": best_w,
        "champion": champion_name,
        "champion_oof": champion_auc,
        "frozen_w62_oof": frozen_w62,
        "frozen_am40_oof": frozen_am40,
        "delta_vs_frozen_w62": champion_auc - frozen_w62,
        "delta_vs_frozen_am40": champion_auc - frozen_am40,
        "gate_beat_frozen_w62": beat_frozen_w62,
        "gate_beat_frozen_am40": beat_frozen_am40,
        "gate": gate,
        "submission_beat_w62": str(champ_path.relative_to(ROOT)) if beat_frozen_w62 else None,
        "submission_beat_w62_sha256": champ_sha,
        "spearman_vs_submission_w62": spear,
        "elapsed_minutes": (time.time() - t0) / 60,
    }
    (art / f"metrics{suffix}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps({k: metrics[k] for k in (
        "arm_oof", "champion", "champion_oof", "frozen_w62_oof",
        "delta_vs_frozen_w62", "gate", "wbest_w_main",
    )}, indent=2, ensure_ascii=False), flush=True)
    for name, info in paths.items():
        print(f"  {name}: OOF={info['oof_auc']:.8f} -> {info['path']}", flush=True)
    print(f"elapsed {metrics['elapsed_minutes']:.1f} min | gate={gate}", flush=True)
    return 0 if beat_frozen_w62 or args.smoke else 2


if __name__ == "__main__":
    raise SystemExit(main())

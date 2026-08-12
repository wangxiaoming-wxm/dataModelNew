#!/usr/bin/env python3
"""同构扰动：只把 main 的 l2 10→14，alt 冻结；融合门禁超过 AM40。"""
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
    BAG_SEEDS,
    N_SPLITS,
    SEEDS,
    build_main,
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--data-dir")
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    art = ROOT / "artifacts" / "iso_main_l2"
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
    frozen_oof = np.load(ROOT / "artifacts" / "super714" / "best_v1_oof.npy", allow_pickle=True).item()
    frozen_te = np.load(ROOT / "artifacts" / "super714" / "best_v1_test.npy", allow_pickle=True).item()
    alt = np.asarray(frozen_oof["alt"], float)
    te_alt = np.asarray(frozen_te["alt"], float)

    print(
        f"=== ISO main l2=14 smoke={args.smoke} seeds={list(seeds)} bags={list(bag_seeds)} ===",
        flush=True,
    )
    t0 = time.time()
    edges = fit_edges_main(raw_all)
    o1, t1, a1 = run_arm(
        "main_l2_14", build_main, edges, train, test, y,
        True, 5, iterations, 14,
        seeds=seeds, n_splits=n_splits, bag_seeds=bag_seeds,
    )
    print(f"main_l2_14 pool OOF={a1:.5f}", flush=True)

    cand_oof = {
        "max2": np.maximum(o1, alt),
        "w62": W_MAIN * o1 + W_ALT * alt,
        "am40": am40(o1, alt),
    }
    cand_te = {
        "max2": np.maximum(t1, te_alt),
        "w62": W_MAIN * t1 + W_ALT * te_alt,
        "am40": am40(t1, te_alt),
    }
    best_w, best_auc = W_MAIN, -1.0
    for w in np.round(np.arange(0.50, 0.801, 0.01), 2):
        auc = float(roc_auc_score(y, w * o1 + (1.0 - w) * alt))
        if auc > best_auc:
            best_auc, best_w = auc, float(w)
    cand_oof["wbest"] = best_w * o1 + (1.0 - best_w) * alt
    cand_te["wbest"] = best_w * t1 + (1.0 - best_w) * te_alt
    scores = {k: float(roc_auc_score(y, v)) for k, v in cand_oof.items()}
    champ = max(scores, key=scores.get)
    beat = scores[champ] > AM40_OOF + 1e-12

    paths = {}
    for name, te in cand_te.items():
        p = sub / f"submission_iso_l2_{name}{suffix}.csv"
        paths[name] = {"path": str(p.relative_to(ROOT)), "sha256": write_sub(sample, te, p), "oof_auc": scores[name]}

    beat_path = sub / f"submission_beat_am40_iso{suffix}.csv"
    if beat:
        shutil.copyfile(sub / f"submission_iso_l2_{champ}{suffix}.csv", beat_path)
        gate = "PASS"
    else:
        gate = "FAIL"

    np.save(art / f"iso_oof{suffix}.npy", {"main": o1, "alt": alt, **cand_oof})
    np.save(art / f"iso_test{suffix}.npy", {"main": t1, "alt": te_alt, **cand_te})
    metrics = {
        "mode": "smoke" if args.smoke else "full",
        "recipe": {"main": "Ordered d5 l2=14 (only change)", "alt": "frozen best_v1", "seeds": list(seeds), "bags": list(bag_seeds)},
        "main_oof": a1,
        "frozen_main_oof": float(roc_auc_score(y, frozen_oof["main"])),
        "spearman_new_vs_frozen_main": float(spearmanr(o1, frozen_oof["main"]).statistic),
        "pearson_main_alt": float(pearsonr(o1, alt).statistic),
        "fusions": paths,
        "champion": champ,
        "champion_oof": scores[champ],
        "frozen_am40_oof": AM40_OOF,
        "delta_vs_am40": scores[champ] - AM40_OOF,
        "gate": gate,
        "wbest_w_main": best_w,
        "elapsed_minutes": (time.time() - t0) / 60,
    }
    (art / f"metrics{suffix}.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("main_oof", "champion", "champion_oof", "delta_vs_am40", "gate")}, indent=2))
    for k, v in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"  {v:.8f}  {k}")
    print(f"elapsed {metrics['elapsed_minutes']:.1f} min | gate={gate}", flush=True)
    return 0 if beat or args.smoke else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Plus 双臂后处理：max2 + 预注册 W62 权重 + OOF 权重网格。

依赖 train_super714_plus 产物：
  artifacts/super714_plus/plus_{oof,test}.npy  (keys: main, alt, fuse)

输出：
  submissions/submission_super714_plus.csv          # max2（与训练脚本一致）
  submissions/submission_super714_plus_w62.csv      # 0.62*main+0.38*alt
  submissions/submission_super714_plus_wbest.csv    # OOF 网格最优权重
  artifacts/super714_plus/fuse_weights_metrics.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
W62_MAIN = 0.62
W62_ALT = 0.38


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_data_dir(explicit: str | None) -> Path:
    for candidate in (explicit, os.environ.get("DATA_DIR"), ROOT / "data"):
        if not candidate:
            continue
        directory = Path(candidate).expanduser().resolve()
        if (directory / "train.csv").is_file() and (directory / "test.csv").is_file():
            return directory
    raise FileNotFoundError("找不到 data/train.csv 与 test.csv")


def write_sub(sample: pd.DataFrame, scores: np.ndarray, path: Path) -> str:
    out = sample[["id"]].copy()
    out["label"] = np.clip(np.asarray(scores, float), 0.001, 0.999)
    out.to_csv(path, index=False)
    return sha256(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plus 双臂加权融合后处理")
    parser.add_argument("--data-dir")
    parser.add_argument("--suffix", default="", help="如 _smoke")
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    art = ROOT / "artifacts" / "super714_plus"
    sub = ROOT / "submissions"
    sub.mkdir(parents=True, exist_ok=True)
    suffix = args.suffix

    oof_path = art / f"plus_oof{suffix}.npy"
    te_path = art / f"plus_test{suffix}.npy"
    if not oof_path.is_file() or not te_path.is_file():
        raise FileNotFoundError(f"缺少 Plus 产物：{oof_path} / {te_path}")

    train = pd.read_csv(data_dir / "train.csv", usecols=["id", "label"], dtype={"id": str})
    sample = pd.read_csv(data_dir / "submit_sample.csv", dtype={"id": str})
    y = train["label"].astype(int).to_numpy()
    oof = np.load(oof_path, allow_pickle=True).item()
    te = np.load(te_path, allow_pickle=True).item()
    main_o = np.asarray(oof["main"], float)
    alt_o = np.asarray(oof["alt"], float)
    main_t = np.asarray(te["main"], float)
    alt_t = np.asarray(te["alt"], float)

    max2_o = np.maximum(main_o, alt_o)
    max2_t = np.maximum(main_t, alt_t)
    max2_auc = float(roc_auc_score(y, max2_o))

    w62_o = W62_MAIN * main_o + W62_ALT * alt_o
    w62_t = W62_MAIN * main_t + W62_ALT * alt_t
    w62_auc = float(roc_auc_score(y, w62_o))

    grid = []
    best_w, best_auc = W62_MAIN, -1.0
    for w in np.round(np.arange(0.50, 0.801, 0.01), 2):
        s = w * main_o + (1.0 - w) * alt_o
        auc = float(roc_auc_score(y, s))
        grid.append({"w_main": float(w), "oof_auc": auc})
        if auc > best_auc:
            best_auc = auc
            best_w = float(w)
    best_t = best_w * main_t + (1.0 - best_w) * alt_t

    path_max2 = sub / f"submission_super714_plus{suffix}.csv"
    path_w62 = sub / f"submission_super714_plus_w62{suffix}.csv"
    path_best = sub / f"submission_super714_plus_wbest{suffix}.csv"
    sha_max2 = write_sub(sample, max2_t, path_max2)
    sha_w62 = write_sub(sample, w62_t, path_w62)
    sha_best = write_sub(sample, best_t, path_best)

    # 对照冻结 best_v1 / 线上 W62
    refs = {}
    champ_oof = ROOT / "artifacts" / "super714" / "best_v1_oof.npy"
    if champ_oof.is_file():
        champ = np.load(champ_oof, allow_pickle=True).item()
        refs["best_v1_max2_oof"] = float(roc_auc_score(y, champ["fuse"]))
        refs["delta_max2_vs_best_v1"] = max2_auc - refs["best_v1_max2_oof"]
    for name, p in (
        ("submission_super714", sub / "submission_super714.csv"),
        ("submission_w62", sub / "submission_w62.csv"),
        ("submission_interim_w62", sub / "submission_interim_w62.csv"),
    ):
        if p.is_file():
            other = pd.read_csv(p)["label"].to_numpy(dtype=float)
            refs[f"spearman_max2_vs_{name}"] = float(spearmanr(max2_t, other).statistic)
            refs[f"spearman_w62_vs_{name}"] = float(spearmanr(w62_t, other).statistic)

    metrics = {
        "source": {"oof": str(oof_path.relative_to(ROOT)), "test": str(te_path.relative_to(ROOT))},
        "arm_oof": {
            "main": float(roc_auc_score(y, main_o)),
            "alt": float(roc_auc_score(y, alt_o)),
            "pearson": float(np.corrcoef(main_o, alt_o)[0, 1]),
        },
        "fusions": {
            "max2": {"oof_auc": max2_auc, "submission": str(path_max2.relative_to(ROOT)), "sha256": sha_max2},
            "w62": {
                "formula": f"{W62_MAIN}*main + {W62_ALT}*alt",
                "oof_auc": w62_auc,
                "delta_vs_max2": w62_auc - max2_auc,
                "submission": str(path_w62.relative_to(ROOT)),
                "sha256": sha_w62,
            },
            "wbest": {
                "w_main": best_w,
                "w_alt": 1.0 - best_w,
                "oof_auc": best_auc,
                "delta_vs_max2": best_auc - max2_auc,
                "submission": str(path_best.relative_to(ROOT)),
                "sha256": sha_best,
            },
        },
        "weight_grid": grid,
        "refs": refs,
        "note": "W62 权重预注册（线上 best_v1 臂已验证 0.71503）；wbest 仅 OOF 选权，提交前需谨慎。",
    }
    out_metrics = art / f"fuse_weights_metrics{suffix}.json"
    out_metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("PASS: Plus 加权融合完成")
    print(f"max2  OOF={max2_auc:.5f}  -> {path_max2.name}")
    print(f"w62   OOF={w62_auc:.5f} (Δ={w62_auc - max2_auc:+.5f}) -> {path_w62.name}")
    print(f"wbest OOF={best_auc:.5f} w_main={best_w:.2f} -> {path_best.name}")
    print(f"metrics: {out_metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

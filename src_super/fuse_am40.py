#!/usr/bin/env python3
"""AM40：基于冻结 best_v1 双臂的 max×W62 混合融合（可复现、秒级）。

预注册公式（OOF 与 test 同一套，不按测试集调参）：
    linear = 0.62 * rank_pool(main) + 0.38 * rank_pool(alt)
    score  = 0.40 * max(main, alt) + 0.60 * linear

输入：artifacts/super714/best_v1_{oof,test}.npy（与 max2 / W62 同源臂）
输出：submissions/submission_am40.csv

相对冻结 W62：本地 OOF 更高（门禁强制）；提交哈希不同。
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
W_MAIN = 0.62
W_ALT = 0.38
ALPHA_MAX = 0.40
# 锚定：在仓库冻结 best_v1 臂上复算得到
EXPECTED_OOF = 0.7018113510376338
OOF_ATOL = 1e-12


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


def fuse(main: np.ndarray, alt: np.ndarray) -> np.ndarray:
    linear = W_MAIN * main + W_ALT * alt
    return ALPHA_MAX * np.maximum(main, alt) + (1.0 - ALPHA_MAX) * linear


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 / 核验 AM40 混合融合提交")
    parser.add_argument("--verify-only", action="store_true", help="只核验已有提交，不重写")
    parser.add_argument("--data-dir")
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    art = ROOT / "artifacts" / "super714"
    out_dir = ROOT / "submissions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "submission_am40.csv"
    metrics_path = ROOT / "artifacts" / "am40" / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(data_dir / "train.csv", usecols=["id", "label"], dtype={"id": str})
    test = pd.read_csv(data_dir / "test.csv", usecols=["id"], dtype={"id": str})
    sample = pd.read_csv(data_dir / "submit_sample.csv", dtype={"id": str})
    y = train["label"].astype(int).to_numpy()

    oof = np.load(art / "best_v1_oof.npy", allow_pickle=True).item()
    te = np.load(art / "best_v1_test.npy", allow_pickle=True).item()
    for key in ("main", "alt", "fuse"):
        if key not in oof or key not in te:
            raise KeyError(f"冻结产物缺少键 {key}")

    main = np.asarray(oof["main"], float)
    alt = np.asarray(oof["alt"], float)
    te_main = np.asarray(te["main"], float)
    te_alt = np.asarray(te["alt"], float)

    oof_s = fuse(main, alt)
    te_s = fuse(te_main, te_alt)
    oof_auc = float(roc_auc_score(y, oof_s))
    w62_auc = float(roc_auc_score(y, W_MAIN * main + W_ALT * alt))
    max2_auc = float(roc_auc_score(y, oof["fuse"]))

    if abs(oof_auc - EXPECTED_OOF) > OOF_ATOL:
        raise ValueError(f"OOF 偏离锚点：got {oof_auc:.15f} expected {EXPECTED_OOF:.15f}")
    if oof_auc <= w62_auc + 1e-15:
        raise SystemExit(f"GATE FAIL: AM40 OOF {oof_auc:.10f} <= W62 {w62_auc:.10f}")

    if not sample["id"].astype(str).reset_index(drop=True).equals(
        test["id"].astype(str).reset_index(drop=True)
    ):
        raise ValueError("submit_sample 与 test id 不一致")

    expected_labels = np.clip(te_s, 0.001, 0.999)
    if not args.verify_only:
        submission = sample[["id"]].copy()
        submission["label"] = expected_labels
        submission.to_csv(out_path, index=False)
    elif not out_path.is_file():
        raise FileNotFoundError(f"缺少 {out_path}，请先不加 --verify-only 生成")

    saved = pd.read_csv(out_path, dtype={"id": str})
    if list(saved.columns) != ["id", "label"]:
        raise ValueError("提交列必须为 id,label")
    if not saved["id"].equals(test["id"]):
        raise ValueError("提交 id/行序与 test 不一致")
    labels = saved["label"].to_numpy(dtype=float)
    if not np.isfinite(labels).all() or not ((labels >= 0.001) & (labels <= 0.999)).all():
        raise ValueError("label 越界或非有限")
    max_abs = float(np.max(np.abs(labels - expected_labels)))
    if max_abs > 1e-12:
        raise ValueError(f"提交与重算 fuse 不一致 max|Δ|={max_abs:.3e}")

    refs = {}
    for name, path in (
        ("submission_w62", out_dir / "submission_w62.csv"),
        ("submission_super714", out_dir / "submission_super714.csv"),
    ):
        if path.is_file():
            other = pd.read_csv(path, dtype={"id": str})["label"].to_numpy(dtype=float)
            refs[f"spearman_vs_{name}"] = float(spearmanr(labels, other).statistic)
            refs[f"different_from_{name}"] = sha256(out_path) != sha256(path)

    digest = sha256(out_path)
    metrics = {
        "name": "AM40",
        "formula": f"{ALPHA_MAX}*max(main,alt) + {1.0 - ALPHA_MAX}*({W_MAIN}*main+{W_ALT}*alt)",
        "alpha_max": ALPHA_MAX,
        "linear_weights": {"main": W_MAIN, "alt": W_ALT},
        "source_arms": "artifacts/super714/best_v1_{oof,test}.npy",
        "oof_auc": oof_auc,
        "w62_oof_auc": w62_auc,
        "max2_oof_auc": max2_auc,
        "delta_vs_w62": oof_auc - w62_auc,
        "delta_vs_max2": oof_auc - max2_auc,
        "gate_beat_w62": True,
        "submission": str(out_path.relative_to(ROOT)),
        "submission_sha256": digest,
        "n_test": int(len(saved)),
        **refs,
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("PASS: AM40 融合验收通过（已超过 W62）")
    print(f"OOF AUC: {oof_auc:.8f} (W62={w62_auc:.8f}, Δ={oof_auc - w62_auc:+.8f})")
    print(f"submission: {out_path}")
    print(f"sha256: {digest}")
    if "spearman_vs_submission_w62" in refs:
        print(
            f"spearman_vs_w62: {refs['spearman_vs_submission_w62']:.6f} "
            f"different={refs['different_from_submission_w62']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

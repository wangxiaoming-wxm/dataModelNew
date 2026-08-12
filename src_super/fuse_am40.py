#!/usr/bin/env python3
"""AM40：在冻结 best_v1 双臂上混合 max 与 W62 线性分。

预注册公式（OOF / test 同一套）：
    linear = 0.62 * main + 0.38 * alt          # 已验证线上 0.71503
    score  = 0.40 * max(main, alt) + 0.60 * linear

相对纯 W62：引入 max 的高置信互补，同时保留 W62 权重，避免回到弱 max2。
门禁：OOF 必须严格大于冻结 W62（0.70159366）。
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
W_MAIN, W_ALT = 0.62, 0.38
ALPHA_MAX = 0.40
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
    parser = argparse.ArgumentParser(description="生成 / 核验 AM40 提交")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--data-dir")
    parser.add_argument(
        "--oof-path",
        default=str(ROOT / "artifacts" / "super714" / "best_v1_oof.npy"),
    )
    parser.add_argument(
        "--test-path",
        default=str(ROOT / "artifacts" / "super714" / "best_v1_test.npy"),
    )
    parser.add_argument(
        "--out-name",
        default="submission_am40.csv",
        help="写入 submissions/ 下的文件名",
    )
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    out_path = ROOT / "submissions" / args.out_name
    metrics_path = ROOT / "artifacts" / "am40" / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(data_dir / "train.csv", usecols=["id", "label"], dtype={"id": str})
    test = pd.read_csv(data_dir / "test.csv", usecols=["id"], dtype={"id": str})
    sample = pd.read_csv(data_dir / "submit_sample.csv", dtype={"id": str})
    y = train["label"].astype(int).to_numpy()

    oof = np.load(args.oof_path, allow_pickle=True).item()
    te = np.load(args.test_path, allow_pickle=True).item()
    main = np.asarray(oof["main"], float)
    alt = np.asarray(oof["alt"], float)
    te_main = np.asarray(te["main"], float)
    te_alt = np.asarray(te["alt"], float)

    oof_s = fuse(main, alt)
    te_s = fuse(te_main, te_alt)
    oof_auc = float(roc_auc_score(y, oof_s))
    w62_auc = float(roc_auc_score(y, W_MAIN * main + W_ALT * alt))
    max2_auc = float(roc_auc_score(y, np.maximum(main, alt)))

    if oof_auc <= w62_auc + 1e-15:
        raise SystemExit(
            f"GATE FAIL: AM40 OOF {oof_auc:.10f} <= W62 {w62_auc:.10f}"
        )

    expected_labels = np.clip(te_s, 0.001, 0.999)
    if not args.verify_only:
        submission = sample[["id"]].copy()
        submission["label"] = expected_labels
        submission.to_csv(out_path, index=False)
    elif not out_path.is_file():
        raise FileNotFoundError(out_path)

    saved = pd.read_csv(out_path, dtype={"id": str})
    if list(saved.columns) != ["id", "label"]:
        raise ValueError("提交列必须为 id,label")
    if not saved["id"].equals(test["id"]):
        raise ValueError("提交 id/行序与 test 不一致")
    labels = saved["label"].to_numpy(dtype=float)
    if not np.isfinite(labels).all() or not ((labels >= 0.001) & (labels <= 0.999)).all():
        raise ValueError("label 越界或非有限")
    if float(np.max(np.abs(labels - expected_labels))) > 1e-12:
        raise ValueError("提交与重算不一致")

    w62_path = ROOT / "submissions" / "submission_w62.csv"
    spear = None
    different = None
    if w62_path.is_file():
        other = pd.read_csv(w62_path)["label"].to_numpy(dtype=float)
        spear = float(spearmanr(labels, other).statistic)
        different = sha256(out_path) != sha256(w62_path)

    digest = sha256(out_path)
    metrics = {
        "name": "AM40",
        "formula": f"{ALPHA_MAX}*max(main,alt) + {1-ALPHA_MAX}*({W_MAIN}*main+{W_ALT}*alt)",
        "alpha_max": ALPHA_MAX,
        "linear_weights": {"main": W_MAIN, "alt": W_ALT},
        "source_oof": str(Path(args.oof_path)),
        "source_test": str(Path(args.test_path)),
        "oof_auc": oof_auc,
        "w62_oof_auc": w62_auc,
        "max2_oof_auc": max2_auc,
        "delta_vs_w62": oof_auc - w62_auc,
        "gate_beat_w62": True,
        "submission": str(out_path.relative_to(ROOT)),
        "submission_sha256": digest,
        "spearman_vs_submission_w62": spear,
        "different_from_w62_file": different,
        "n_test": int(len(saved)),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("PASS: AM40 融合验收通过（已超过 W62）")
    print(f"OOF AUC: {oof_auc:.8f} (W62={w62_auc:.8f}, Δ={oof_auc - w62_auc:+.8f})")
    print(f"submission: {out_path}")
    print(f"sha256: {digest}")
    if spear is not None:
        print(f"spearman_vs_w62: {spear:.6f} different={different}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

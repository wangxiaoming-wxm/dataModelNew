#!/usr/bin/env python3
"""秒级核验 SUPER714 的预计算 best_v1 锚点与主提交。"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "super714"
EXPECTED_SHA256 = {
    "best_v1_oof.npy": "ef23c61013f9ecf469174c55849983677de2b669cce6c052f999808545b7600d",
    "best_v1_test.npy": "aaa43ca48b9d297c35367c873f6001c3607f5cff4a9f96a6ac72e284a57942dd",
}
EXPECTED_AUC = {"main": 0.69992, "alt": 0.69770, "fuse": 0.70128}


def resolve_data_dir(explicit: str | None) -> Path:
    """定位数据目录。"""
    for candidate in (explicit, os.environ.get("DATA_DIR"), ROOT / "data"):
        if not candidate:
            continue
        directory = Path(candidate).expanduser().resolve()
        if (directory / "train.csv").is_file() and (directory / "test.csv").is_file():
            return directory
    raise FileNotFoundError("找不到 train.csv/test.csv；请设置 DATA_DIR 或传入 --data-dir")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="核验 SUPER714 预计算锚点")
    parser.add_argument("--data-dir", help="含 train.csv/test.csv 的目录")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = resolve_data_dir(args.data_dir)
    train = pd.read_csv(data_dir / "train.csv", usecols=["id", "label"], dtype={"id": str})
    test = pd.read_csv(data_dir / "test.csv", usecols=["id"], dtype={"id": str})

    oof_path = ARTIFACT_DIR / "best_v1_oof.npy"
    test_path = ARTIFACT_DIR / "best_v1_test.npy"
    for path in (oof_path, test_path):
        actual_hash = sha256(path)
        expected_hash = EXPECTED_SHA256[path.name]
        if actual_hash != expected_hash:
            raise ValueError(f"{path} SHA-256 不匹配：{actual_hash}")

    oof = np.load(oof_path, allow_pickle=True).item()
    test_pred = np.load(test_path, allow_pickle=True).item()
    expected_keys = {"main", "alt", "fuse"}
    if set(oof) != expected_keys or set(test_pred) != expected_keys:
        raise ValueError("预计算产物必须包含 main/alt/fuse 三个键")

    y = train["label"].astype(int).to_numpy()
    scores = {
        key: float(roc_auc_score(y, np.asarray(oof[key])))
        for key in sorted(expected_keys)
    }
    for key, expected in EXPECTED_AUC.items():
        if abs(scores[key] - expected) > 5e-5:
            raise ValueError(f"{key} OOF={scores[key]:.8f}，偏离锚点 {expected:.5f}")

    if any(len(np.asarray(value)) != len(train) for value in oof.values()):
        raise ValueError("OOF 长度与 train.csv 不一致")
    if any(len(np.asarray(value)) != len(test) for value in test_pred.values()):
        raise ValueError("test 预测长度与 test.csv 不一致")

    submission = pd.read_csv(ROOT / "submissions" / "submission_super714.csv", dtype={"id": str})
    if list(submission.columns) != ["id", "label"]:
        raise ValueError("主提交列必须严格为 id,label")
    if not submission["id"].equals(test["id"]):
        raise ValueError("主提交 id 或行序与 test.csv 不一致")
    labels = submission["label"].to_numpy(dtype=float)
    if not np.isfinite(labels).all() or not ((labels >= 0.001) & (labels <= 0.999)).all():
        raise ValueError("主提交 label 必须为有限数且位于 [0.001, 0.999]")

    print("PASS: SUPER714 预计算锚点与主提交验收通过")
    print("OOF AUC:", {key: f"{value:.5f}" for key, value in scores.items()})
    print("rows:", {"train": len(train), "test": len(test), "submission": len(submission)})
    print("DATA_DIR:", data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

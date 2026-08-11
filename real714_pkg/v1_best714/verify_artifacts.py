#!/usr/bin/env python3
"""秒级核验已保存 best_v1 产物是否与线上锚点一致。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PKG = Path(__file__).resolve().parent


def resolve_data() -> Path:
    env = os.environ.get("DATA_DIR")
    cands = []
    if env:
        cands.append(Path(env))
    cands += [
        Path("/Volumes/pssd/app/ml/正式比赛/data"),
        PKG.parents[2] / "data",
        PKG / "data",
    ]
    for d in cands:
        if (d / "train.csv").is_file():
            return d
    raise FileNotFoundError("找不到 train.csv，请设置 DATA_DIR")


def main() -> int:
    data = resolve_data()
    y = pd.read_csv(data / "train.csv")["label"].astype(int).to_numpy()
    oof = np.load(PKG / "artifacts" / "best_oof.npy", allow_pickle=True).item()
    te = np.load(PKG / "artifacts" / "best_test.npy", allow_pickle=True).item()
    sub = pd.read_csv(PKG / "submissions" / "submission_best.csv")

    scores = {k: float(roc_auc_score(y, np.asarray(v))) for k, v in oof.items()}
    print("DATA:", data)
    print("OOF AUC:", {k: f"{v:.5f}" for k, v in scores.items()})
    print("test lens:", {k: len(np.asarray(v)) for k, v in te.items()})
    print("submission rows:", len(sub), "cols:", list(sub.columns))
    print("label range:", float(sub["label"].min()), float(sub["label"].max()))

    ok = True
    if abs(scores["fuse"] - 0.70128) > 5e-4:
        print("FAIL: fuse OOF 偏离 0.70128 过多:", scores["fuse"])
        ok = False
    if len(sub) != 6398:
        print("FAIL: submission 行数应为 6398")
        ok = False
    if not {"id", "label"}.issubset(sub.columns):
        print("FAIL: submission 列缺失")
        ok = False
    if ok:
        print("PASS: artifacts 与锚点一致（fuse≈0.70128, 线上锚点 0.71453）")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

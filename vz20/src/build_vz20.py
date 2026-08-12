#!/usr/bin/env python3
"""从冻结 best_v1 双臂构造 vz20。

独立审核结论：rebuild V2 诚实 nested≈0.695，弱于已上线 W62（OOF 0.70159 / LB 0.71503）。
vz19 的 max2+byteTE 本地 OOF 抬升未迁移（LB 0.71298 < W62）。
本文件不重训 8×5×3；只从冻结 OOF/test 做预注册融合，保证可复现。

预注册配方（在查看 test 之前锁定）：
  W62  = 0.62*main + 0.38*alt          # 线上锚点 0.71503
  AM40 = 0.40*max(main,alt) + 0.60*W62 # 本地 OOF 最高的同构融合
  vz20 = AM40                          # 实验主交；保守备份 = W62
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
W_MAIN, W_ALT = 0.62, 0.38
ALPHA_MAX = 0.40
EXPECTED_W62_OOF = 0.7015936597140784
EXPECTED_AM40_OOF = 0.7018113510376338
OOF_ATOL = 1e-10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def am40(main: np.ndarray, alt: np.ndarray) -> np.ndarray:
    linear = W_MAIN * main + W_ALT * alt
    return ALPHA_MAX * np.maximum(main, alt) + (1.0 - ALPHA_MAX) * linear


def w62(main: np.ndarray, alt: np.ndarray) -> np.ndarray:
    return W_MAIN * main + W_ALT * alt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--out-dir", default=str(ROOT / "vz20"))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    art = out_dir / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    (out_dir / "submissions").mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(data_dir / "train.csv", dtype={"id": str})
    sample = pd.read_csv(data_dir / "submit_sample.csv", dtype={"id": str})
    y = train["label"].astype(int).to_numpy()

    oof = np.load(ROOT / "artifacts" / "super714" / "best_v1_oof.npy", allow_pickle=True).item()
    test = np.load(ROOT / "artifacts" / "super714" / "best_v1_test.npy", allow_pickle=True).item()
    main_o = np.asarray(oof["main"], float)
    alt_o = np.asarray(oof["alt"], float)
    main_t = np.asarray(test["main"], float)
    alt_t = np.asarray(test["alt"], float)

    w62_o = w62(main_o, alt_o)
    am40_o = am40(main_o, alt_o)
    w62_t = np.clip(w62(main_t, alt_t), 0.001, 0.999)
    am40_t = np.clip(am40(main_t, alt_t), 0.001, 0.999)

    w62_auc = float(roc_auc_score(y, w62_o))
    am40_auc = float(roc_auc_score(y, am40_o))
    if abs(w62_auc - EXPECTED_W62_OOF) > OOF_ATOL:
        raise ValueError(f"W62 OOF mismatch {w62_auc} vs {EXPECTED_W62_OOF}")
    if abs(am40_auc - EXPECTED_AM40_OOF) > OOF_ATOL:
        raise ValueError(f"AM40 OOF mismatch {am40_auc} vs {EXPECTED_AM40_OOF}")

    def write_sub(path: Path, pred: np.ndarray) -> str:
        sub = sample[["id"]].copy()
        sub["label"] = pred
        sub.to_csv(path, index=False)
        return sha256(path)

    vz20_path = out_dir / "submission_vz20.csv"
    w62_path = out_dir / "submission_vz20_w62_anchor.csv"
    repo_vz20 = ROOT / "submissions" / "submission_vz20.csv"
    sha_v = write_sub(vz20_path, am40_t)
    sha_w = write_sub(w62_path, w62_t)
    repo_vz20.parent.mkdir(parents=True, exist_ok=True)
    repo_vz20.write_bytes(vz20_path.read_bytes())
    (ROOT / "submissions" / "submission_vz20_w62_anchor.csv").write_bytes(w62_path.read_bytes())

    np.save(art / "vz20_oof.npy", am40_o)
    np.save(art / "vz20_test.npy", am40_t)
    np.save(art / "w62_oof.npy", w62_o)
    np.save(art / "w62_test.npy", w62_t)

    metrics = {
        "name": "vz20_am40_from_frozen_best_v1",
        "formula": "0.40*max(main,alt) + 0.60*(0.62*main+0.38*alt)",
        "oof_auc": am40_auc,
        "w62_oof_auc": w62_auc,
        "delta_oof_vs_w62": am40_auc - w62_auc,
        "online_w62": 0.71503,
        "online_vz19": 0.71298,
        "online_vz17": 0.71487,
        "expected_online_if_w62_gap": round(am40_auc + (0.71503 - EXPECTED_W62_OOF), 5),
        "championship_floor": 0.72,
        "championship_target": 0.749,
        "path_to_0.72": False,
        "path_to_0.749": False,
        "submission": "vz20/submission_vz20.csv",
        "submission_sha256": sha_v,
        "w62_anchor_sha256": sha_w,
        "tempered": "vz20/submission_vz20_w62_anchor.csv",
        "note": "AM40 仅比 W62 高 0.00022 OOF；期望线上仍在 0.715 附近，不是 0.72/0.749。",
    }
    (art / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    (out_dir / "artifacts" / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    print("PASS vz20 AM40")
    print(f"OOF={am40_auc:.8f} (W62={w62_auc:.8f}, Δ={am40_auc - w62_auc:+.8f})")
    print(f"sha256: {sha_v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

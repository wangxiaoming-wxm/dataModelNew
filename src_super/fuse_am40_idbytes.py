#!/usr/bin/env python3
"""AM40 + id 字节 fold-local TE 混合（可复现）。

发现：id 的 hex byte0/4/5/7 有 fold-local TE 真信号，且与 AM40 近乎正交。
公式（预注册）：
    id_pool = mean_rank( flip_if_needed( TE_foldlocal(id_byte_b) ) for b in [0,4,5,7] )
    score   = 0.75 * AM40(main,alt) + 0.25 * id_pool

AM40 定义同 fuse_am40.py；臂来自冻结 best_v1。
门禁：OOF > 冻结 AM40。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
W_MAIN, W_ALT = 0.62, 0.38
ALPHA_MAX = 0.40
W_AM40 = 0.75
STRONG_BYTES = (0, 4, 5, 7)
TE_SEED = 2026
TE_SPLITS = 5
TE_SMOOTH = 20.0
EXPECTED_OOF = 0.7044127250356068
OOF_ATOL = 1e-10
AM40_OOF = 0.7018113510376338


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
    raise FileNotFoundError("找不到 data/")


def am40(main: np.ndarray, alt: np.ndarray) -> np.ndarray:
    linear = W_MAIN * main + W_ALT * alt
    return ALPHA_MAX * np.maximum(main, alt) + (1.0 - ALPHA_MAX) * linear


def id_byte_series(ids: pd.Series, byte: int) -> pd.Series:
    s = ids.astype(str).str.lower()
    return s.str.slice(2 * byte, 2 * byte + 2)


def te_oof_test(tr_keys: pd.Series, te_keys: pd.Series, y: np.ndarray):
    skf = StratifiedKFold(TE_SPLITS, shuffle=True, random_state=TE_SEED)
    oof = np.zeros(len(y), dtype=float)
    prior = float(y.mean())
    for tri, vali in skf.split(np.zeros(len(y)), y):
        stats = pd.DataFrame({"k": tr_keys.iloc[tri].to_numpy(), "y": y[tri]}).groupby("k")["y"].agg(["sum", "count"])
        mapping = (stats["sum"] + TE_SMOOTH * prior) / (stats["count"] + TE_SMOOTH)
        oof[vali] = tr_keys.iloc[vali].map(mapping).fillna(prior).to_numpy(dtype=float)
    stats = pd.DataFrame({"k": tr_keys.to_numpy(), "y": y}).groupby("k")["y"].agg(["sum", "count"])
    mapping = (stats["sum"] + TE_SMOOTH * prior) / (stats["count"] + TE_SMOOTH)
    test_pred = te_keys.map(mapping).fillna(prior).to_numpy(dtype=float)
    return oof, test_pred


def id_pool(train_ids: pd.Series, test_ids: pd.Series, y: np.ndarray):
    parts_o, parts_t = [], []
    for b in STRONG_BYTES:
        oof, tp = te_oof_test(id_byte_series(train_ids, b), id_byte_series(test_ids, b), y)
        if roc_auc_score(y, oof) < 0.5:
            oof = 1.0 - oof
            tp = 1.0 - tp
        parts_o.append(rankdata(oof) / len(oof))
        parts_t.append(rankdata(tp) / len(tp))
    return np.mean(parts_o, axis=0), np.mean(parts_t, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--data-dir")
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    train = pd.read_csv(data_dir / "train.csv", usecols=["id", "label"], dtype={"id": str})
    test = pd.read_csv(data_dir / "test.csv", usecols=["id"], dtype={"id": str})
    sample = pd.read_csv(data_dir / "submit_sample.csv", dtype={"id": str})
    y = train["label"].astype(int).to_numpy()

    oof = np.load(ROOT / "artifacts" / "super714" / "best_v1_oof.npy", allow_pickle=True).item()
    te = np.load(ROOT / "artifacts" / "super714" / "best_v1_test.npy", allow_pickle=True).item()
    am40_o = am40(np.asarray(oof["main"], float), np.asarray(oof["alt"], float))
    am40_t = am40(np.asarray(te["main"], float), np.asarray(te["alt"], float))
    ip_o, ip_t = id_pool(train["id"], test["id"], y)
    fuse_o = W_AM40 * am40_o + (1.0 - W_AM40) * ip_o
    fuse_t = W_AM40 * am40_t + (1.0 - W_AM40) * ip_t
    oof_auc = float(roc_auc_score(y, fuse_o))
    am40_auc = float(roc_auc_score(y, am40_o))

    if abs(oof_auc - EXPECTED_OOF) > OOF_ATOL:
        raise ValueError(f"OOF 偏离锚点 got {oof_auc:.15f} expected {EXPECTED_OOF:.15f}")
    if oof_auc <= am40_auc + 1e-15:
        raise SystemExit(f"GATE FAIL: {oof_auc} <= AM40 {am40_auc}")

    out_path = ROOT / "submissions" / "submission_am40_idbytes.csv"
    expected = np.clip(fuse_t, 0.001, 0.999)
    if not args.verify_only:
        sub = sample[["id"]].copy()
        sub["label"] = expected
        sub.to_csv(out_path, index=False)
        # 同步 champion
        (ROOT / "submissions" / "submission_champion.csv").write_bytes(out_path.read_bytes())
    saved = pd.read_csv(out_path, dtype={"id": str})
    if float(np.max(np.abs(saved["label"].to_numpy(float) - expected))) > 1e-12:
        raise ValueError("提交与重算不一致")

    art = ROOT / "artifacts" / "id_bytes"
    art.mkdir(parents=True, exist_ok=True)
    metrics = {
        "name": "AM40+id_bytes_TE",
        "formula": f"{W_AM40}*AM40 + {1-W_AM40}*rankmean(TE(id_bytes {list(STRONG_BYTES)}))",
        "w_am40": W_AM40,
        "strong_bytes": list(STRONG_BYTES),
        "oof_auc": oof_auc,
        "am40_oof": am40_auc,
        "delta_vs_am40": oof_auc - am40_auc,
        "id_pool_oof": float(roc_auc_score(y, ip_o)),
        "spearman_id_vs_am40": float(spearmanr(ip_o, am40_o).statistic),
        "submission": str(out_path.relative_to(ROOT)),
        "submission_sha256": sha256(out_path),
        "gate_beat_am40": True,
    }
    w62 = ROOT / "submissions" / "submission_w62.csv"
    if w62.is_file():
        metrics["spearman_vs_w62"] = float(
            spearmanr(saved["label"], pd.read_csv(w62)["label"]).statistic
        )
    (art / "blend_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    print("PASS: AM40+id_bytes 超过 AM40")
    print(f"OOF={oof_auc:.8f} (AM40={am40_auc:.8f}, Δ={oof_auc-am40_auc:+.8f})")
    print(f"submission: {out_path}")
    print(f"sha256: {metrics['submission_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

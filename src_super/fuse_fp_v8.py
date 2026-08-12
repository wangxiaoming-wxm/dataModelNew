#!/usr/bin/env python3
"""第一性原理 v8：在 fp_v7 上加入字节对 XOR/AND TE 池。

证据：bit 级组合已大量使用；整字节对 XOR/AND 是更粗粒度、选择无关的
新键空间（C(8,2)=28），与 v7 Spearman≈0.03。

预注册主交（byte20）：
    bytepair_mean = 0.5*pool(byte_i XOR byte_j) + 0.5*pool(byte_i AND byte_j)
    score = 0.80*v7_cross30 + 0.20*bytepair_mean

门禁：nested fold-mean > fp_v7；锚点 OOF ≈ 0.770161。
备份：tempered=v7；aggressive=0.70*v7+0.30*bytepair_mean。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
TE_SEEDS = (2026, 7, 42, 99, 314, 2718)
TE_SPLITS = 5
TE_SMOOTH = 20.0
W_V7, W_BM = 0.80, 0.20
EXPECTED_OOF = 0.7701614508323125
OOF_ATOL = 1e-8
V7_OOF = 0.7583798471274498
V7_NEST = 0.7585100856647914


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


def te_oof_test(tr_keys: pd.Series, te_keys: pd.Series, y: np.ndarray, seed: int):
    skf = StratifiedKFold(TE_SPLITS, shuffle=True, random_state=seed)
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


def pool_keys(train_keys_list, test_keys_list, y: np.ndarray):
    parts_o, parts_t = [], []
    for tr_keys, te_keys in zip(train_keys_list, test_keys_list):
        seed_o, seed_t = [], []
        for seed in TE_SEEDS:
            oof, tp = te_oof_test(tr_keys, te_keys, y, seed=seed)
            if roc_auc_score(y, oof) < 0.5:
                oof = 1.0 - oof
                tp = 1.0 - tp
            seed_o.append(rankdata(oof) / len(oof))
            seed_t.append(rankdata(tp) / len(tp))
        parts_o.append(np.mean(seed_o, axis=0))
        parts_t.append(np.mean(seed_t, axis=0))
    return np.mean(parts_o, axis=0), np.mean(parts_t, axis=0)


def bytes_arr(ids: pd.Series) -> np.ndarray:
    s = ids.astype(str).str.lower()
    return np.array([[int(s.iloc[i][j : j + 2], 16) for j in range(0, 16, 2)] for i in range(len(s))], np.int16)


def bytepair_pools(train_ids, test_ids, y):
    tr = bytes_arr(train_ids)
    te = bytes_arr(test_ids)
    tr_x, te_x, tr_a, te_a = [], [], [], []
    for i in range(8):
        for j in range(i + 1, 8):
            tr_x.append(pd.Series((tr[:, i] ^ tr[:, j]).astype(str)))
            te_x.append(pd.Series((te[:, i] ^ te[:, j]).astype(str)))
            tr_a.append(pd.Series((tr[:, i] & tr[:, j]).astype(str)))
            te_a.append(pd.Series((te[:, i] & te[:, j]).astype(str)))
    bx_o, bx_t = pool_keys(tr_x, te_x, y)
    ba_o, ba_t = pool_keys(tr_a, te_a, y)
    return 0.5 * (bx_o + ba_o), 0.5 * (bx_t + ba_t), bx_o, ba_o


def ensure_v7(art: Path, reuse: bool) -> None:
    need = ("v7_fuse_oof.npy", "v7_fuse_test.npy")
    if all((art / p).is_file() for p in need):
        return
    cmd = [sys.executable, "-u", str(ROOT / "src_super" / "fuse_fp_v7.py")]
    if reuse:
        cmd.append("--reuse-caches")
    subprocess.check_call(cmd, cwd=str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--data-dir")
    parser.add_argument("--reuse-caches", action="store_true")
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    train = pd.read_csv(data_dir / "train.csv", dtype={"id": str})
    test = pd.read_csv(data_dir / "test.csv", dtype={"id": str})
    sample = pd.read_csv(data_dir / "submit_sample.csv", dtype={"id": str})
    y = train["label"].astype(int).to_numpy()

    art = ROOT / "artifacts" / "first_principles"
    art.mkdir(parents=True, exist_ok=True)
    ensure_v7(art, reuse=args.reuse_caches)

    v7_o = np.load(art / "v7_fuse_oof.npy")
    v7_t = np.load(art / "v7_fuse_test.npy")

    cache_ok = args.reuse_caches and all(
        (art / p).is_file() for p in ("bytepair_mean_oof.npy", "bytepair_mean_test.npy")
    )
    if cache_ok:
        print("reusing bytepair caches...", flush=True)
        bm_o = np.load(art / "bytepair_mean_oof.npy")
        bm_t = np.load(art / "bytepair_mean_test.npy")
        bx_o = np.load(art / "bytepair_xor_oof.npy") if (art / "bytepair_xor_oof.npy").is_file() else None
        ba_o = np.load(art / "bytepair_and_oof.npy") if (art / "bytepair_and_oof.npy").is_file() else None
    else:
        print("computing bytepair pools...", flush=True)
        bm_o, bm_t, bx_o, ba_o = bytepair_pools(train["id"], test["id"], y)
        np.save(art / "bytepair_mean_oof.npy", bm_o)
        np.save(art / "bytepair_mean_test.npy", bm_t)
        np.save(art / "bytepair_xor_oof.npy", bx_o)
        np.save(art / "bytepair_and_oof.npy", ba_o)

    fuse_o = W_V7 * v7_o + W_BM * bm_o
    fuse_t = W_V7 * v7_t + W_BM * bm_t
    oof_auc = float(roc_auc_score(y, fuse_o))
    if abs(oof_auc - EXPECTED_OOF) > OOF_ATOL:
        raise ValueError(f"OOF 偏离锚点 got {oof_auc:.15f} expected {EXPECTED_OOF:.15f}")
    if oof_auc <= V7_OOF + 1e-15:
        raise SystemExit(f"GATE FAIL vs v7: {oof_auc} <= {V7_OOF}")

    skf = StratifiedKFold(5, shuffle=True, random_state=2026)
    nest = float(np.mean([roc_auc_score(y[va], fuse_o[va]) for _, va in skf.split(np.zeros(len(y)), y)]))
    if nest <= V7_NEST + 1e-15:
        raise SystemExit(f"NEST GATE FAIL vs v7: {nest} <= {V7_NEST}")

    agg_o = 0.70 * v7_o + 0.30 * bm_o
    agg_t = 0.70 * v7_t + 0.30 * bm_t

    out_path = ROOT / "submissions" / "submission_fp_v8.csv"
    champ = ROOT / "submissions" / "submission_champion.csv"
    tempered_path = ROOT / "submissions" / "submission_fp_v8_tempered.csv"
    aggressive_path = ROOT / "submissions" / "submission_fp_v8_aggressive.csv"
    expected = np.clip(fuse_t, 0.001, 0.999)

    if not args.verify_only:
        sub = sample[["id"]].copy()
        sub["label"] = expected
        sub.to_csv(out_path, index=False)
        champ.write_bytes(out_path.read_bytes())
        (ROOT / "submissions" / "submission_am40_idbytes.csv").write_bytes(out_path.read_bytes())

        tsub = sample[["id"]].copy()
        tsub["label"] = np.clip(v7_t, 0.001, 0.999)
        tsub.to_csv(tempered_path, index=False)

        asub = sample[["id"]].copy()
        asub["label"] = np.clip(agg_t, 0.001, 0.999)
        asub.to_csv(aggressive_path, index=False)

        np.save(art / "v8_fuse_oof.npy", fuse_o)
        np.save(art / "v8_fuse_test.npy", fuse_t)
        np.save(ROOT / "artifacts" / "id_bytes" / "fuse_oof.npy", fuse_o)
        np.save(ROOT / "artifacts" / "id_bytes" / "fuse_test.npy", fuse_t)

    saved = pd.read_csv(champ if not args.verify_only or champ.is_file() else out_path, dtype={"id": str})
    check = champ if (args.verify_only and champ.is_file()) else out_path
    saved = pd.read_csv(check, dtype={"id": str})
    if float(np.max(np.abs(saved["label"].to_numpy(float) - expected))) > 1e-12:
        raise ValueError("提交与重算不一致")

    metrics = {
        "name": "fp_v8_byte20",
        "formula": f"{W_V7}*v7_cross30 + {W_BM}*bytepair_mean",
        "weights": {"v7": W_V7, "bytepair_mean": W_BM},
        "oof_auc": oof_auc,
        "nested_fold_mean": nest,
        "delta_nest_vs_v7": nest - V7_NEST,
        "delta_full_vs_v7": oof_auc - V7_OOF,
        "arm_aucs": {
            "byte_mean": float(roc_auc_score(y, bm_o)),
            **({"byte_xor": float(roc_auc_score(y, bx_o))} if bx_o is not None else {}),
            **({"byte_and": float(roc_auc_score(y, ba_o))} if ba_o is not None else {}),
        },
        "spearman_bm_vs_v7": float(spearmanr(bm_o, v7_o).statistic),
        "tempered": {
            "name": "fp_v7_cross30",
            "oof": float(roc_auc_score(y, v7_o)),
            "sha256": sha256(tempered_path) if tempered_path.is_file() else None,
        },
        "aggressive": {
            "name": "byte30",
            "oof": float(roc_auc_score(y, agg_o)),
            "sha256": sha256(aggressive_path) if aggressive_path.is_file() else None,
        },
        "fp_v7_oof": V7_OOF,
        "submission": "submissions/submission_champion.csv",
        "submission_sha256": sha256(check),
        "EXPECTED_OOF": EXPECTED_OOF,
        "gate_beat_v7": True,
    }
    (art / "v8_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    print("PASS: fp_v8 byte20 超过 fp_v7")
    print(f"OOF={oof_auc:.8f} nest={nest:.8f} (v7={V7_OOF:.8f}, Δ={oof_auc - V7_OOF:+.8f})")
    print(f"sha256: {metrics['submission_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

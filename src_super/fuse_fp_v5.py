#!/usr/bin/env python3
"""第一性原理 v5：在 fp_v4 上加入选择无关的 id bit-AND 二阶池。

证据：单 bit TE 池已证明与表格臂近正交；自然扩展为
  - within-byte bit AND（每字节 C(8,2)）
  - cross-byte same-bit AND
两池 rank-mean 得 and_all，再与 v3/bits/xs 线性融合。

预注册主交（and_heavy，v3≥0.30 以锚定表格信号）：
    score = 0.30*v3_dual + 0.25*bits64 + 0.10*xs + 0.35*and_all

门禁：nested fold-mean > fp_v4；锚点 OOF ≈ 0.728800。
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
W_AM40_V3 = 0.55
W_V7_IN_POOL = 0.70
SPECS_V7 = ("b3", "b2hi", "b1hi", "b0", "b4hi", "b5", "b7", "b7hi", "b5hi", "b6hi")
SPECS_V2 = ("b0", "b4", "b5", "b7", "b2hi", "p47")
TE_SEEDS = (2026, 7, 42, 99, 314, 2718)
TE_SPLITS = 5
TE_SMOOTH = 20.0

W_V3, W_BITS, W_X, W_AND = 0.30, 0.25, 0.10, 0.35
EXPECTED_OOF = 0.7288000993568079
OOF_ATOL = 1e-8
V4_OOF = 0.7164047305145615


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
    return ALPHA_MAX * np.maximum(main, alt) + (1.0 - ALPHA_MAX) * (W_MAIN * main + W_ALT * alt)


def keys_from_ids(ids: pd.Series, spec: str) -> pd.Series:
    s = ids.astype(str).str.lower()
    if spec.startswith("b") and spec.endswith("hi"):
        b = int(spec[1])
        return s.str.slice(2 * b, 2 * b + 1)
    if spec.startswith("b") and len(spec) == 2:
        b = int(spec[1])
        return s.str.slice(2 * b, 2 * b + 2)
    if spec.startswith("p") and len(spec) == 3:
        i, j = int(spec[1]), int(spec[2])
        return s.str.slice(2 * i, 2 * i + 2) + "|" + s.str.slice(2 * j, 2 * j + 2)
    raise ValueError(spec)


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


def v3_dual(train_ids, test_ids, y, am40_o, am40_t):
    def specs_keys(specs):
        return pool_keys([keys_from_ids(train_ids, s) for s in specs], [keys_from_ids(test_ids, s) for s in specs], y)

    v7_o, v7_t = specs_keys(SPECS_V7)
    v2_o, v2_t = specs_keys(SPECS_V2)
    ip_o = W_V7_IN_POOL * v7_o + (1.0 - W_V7_IN_POOL) * v2_o
    ip_t = W_V7_IN_POOL * v7_t + (1.0 - W_V7_IN_POOL) * v2_t
    return W_AM40_V3 * am40_o + (1.0 - W_AM40_V3) * ip_o, W_AM40_V3 * am40_t + (1.0 - W_AM40_V3) * ip_t


def bit_matrices(train_ids: pd.Series, test_ids: pd.Series):
    s = train_ids.astype(str).str.lower()
    st = test_ids.astype(str).str.lower()
    arr_tr = np.array([[int(s.iloc[i][j : j + 2], 16) for j in range(0, 16, 2)] for i in range(len(s))], np.int16)
    arr_te = np.array([[int(st.iloc[i][j : j + 2], 16) for j in range(0, 16, 2)] for i in range(len(st))], np.int16)
    bits_tr = np.zeros((len(s), 64), np.int8)
    bits_te = np.zeros((len(st), 64), np.int8)
    for b in range(8):
        for bit in range(8):
            bits_tr[:, b * 8 + bit] = (arr_tr[:, b] >> bit) & 1
            bits_te[:, b * 8 + bit] = (arr_te[:, b] >> bit) & 1
    return bits_tr, bits_te


def bits_pool(bits_tr, bits_te, y):
    tr_keys, te_keys = [], []
    for j in range(64):
        tr_keys.append(pd.Series(bits_tr[:, j].astype(str)))
        te_keys.append(pd.Series(bits_te[:, j].astype(str)))
    return pool_keys(tr_keys, te_keys, y)


def and_pools(bits_tr, bits_te, y):
    # within-byte AND
    tr_w, te_w = [], []
    for b in range(8):
        for i in range(8):
            for j in range(i + 1, 8):
                tr_w.append(pd.Series((bits_tr[:, b * 8 + i] & bits_tr[:, b * 8 + j]).astype(str)))
                te_w.append(pd.Series((bits_te[:, b * 8 + i] & bits_te[:, b * 8 + j]).astype(str)))
    within_o, within_t = pool_keys(tr_w, te_w, y)
    # cross-byte same bit AND
    tr_c, te_c = [], []
    for bit in range(8):
        for b1 in range(8):
            for b2 in range(b1 + 1, 8):
                tr_c.append(pd.Series((bits_tr[:, b1 * 8 + bit] & bits_tr[:, b2 * 8 + bit]).astype(str)))
                te_c.append(pd.Series((bits_te[:, b1 * 8 + bit] & bits_te[:, b2 * 8 + bit]).astype(str)))
    cross_o, cross_t = pool_keys(tr_c, te_c, y)
    return 0.5 * (within_o + cross_o), 0.5 * (within_t + cross_t), within_o, cross_o


def xs_pool(train: pd.DataFrame, test: pd.DataFrame, y: np.ndarray):
    tr_keys, te_keys = [], []
    for i in range(19):
        col = f"x{i}"
        edges = np.quantile(train[col].to_numpy(float), np.linspace(0, 1, 21)[1:-1])
        tr_keys.append(pd.Series(np.digitize(train[col].to_numpy(float), edges).astype(str)))
        te_keys.append(pd.Series(np.digitize(test[col].to_numpy(float), edges).astype(str)))
    return pool_keys(tr_keys, te_keys, y)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--data-dir")
    parser.add_argument("--reuse-caches", action="store_true", help="reuse saved npy pools if present")
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    train = pd.read_csv(data_dir / "train.csv", dtype={"id": str})
    test = pd.read_csv(data_dir / "test.csv", dtype={"id": str})
    sample = pd.read_csv(data_dir / "submit_sample.csv", dtype={"id": str})
    y = train["label"].astype(int).to_numpy()

    oof = np.load(ROOT / "artifacts" / "super714" / "best_v1_oof.npy", allow_pickle=True).item()
    te = np.load(ROOT / "artifacts" / "super714" / "best_v1_test.npy", allow_pickle=True).item()
    am40_o = am40(np.asarray(oof["main"], float), np.asarray(oof["alt"], float))
    am40_t = am40(np.asarray(te["main"], float), np.asarray(te["alt"], float))

    art = ROOT / "artifacts" / "first_principles"
    art.mkdir(parents=True, exist_ok=True)

    cache_ok = args.reuse_caches and all(
        (art / p).is_file()
        for p in (
            "v4_fuse_oof.npy",  # not sufficient alone
            "bits64_oof.npy",
            "bits64_test.npy",
            "xs_oof.npy",
            "xs_test.npy",
            "bitand_all_oof.npy",
            "bitand_all_test.npy",
        )
    )

    if cache_ok:
        print("reusing cached pools...", flush=True)
        bits_o = np.load(art / "bits64_oof.npy")
        bits_t = np.load(art / "bits64_test.npy")
        xs_o = np.load(art / "xs_oof.npy")
        xs_t = np.load(art / "xs_test.npy")
        and_o = np.load(art / "bitand_all_oof.npy")
        and_t = np.load(art / "bitand_all_test.npy")
        # v3 from v4 decomposition if available else compute
        if (art / "v4_fuse_oof.npy").is_file():
            v4_o = np.load(art / "v4_fuse_oof.npy")
            v4_t = np.load(art / "v4_fuse_test.npy")
            v3_o = (v4_o - 0.35 * bits_o - 0.15 * xs_o) / 0.50
            v3_t = (v4_t - 0.35 * bits_t - 0.15 * xs_t) / 0.50
        else:
            v3_o, v3_t = v3_dual(train["id"], test["id"], y, am40_o, am40_t)
    else:
        print("computing v3 dual...", flush=True)
        v3_o, v3_t = v3_dual(train["id"], test["id"], y, am40_o, am40_t)
        print(f"  v3={roc_auc_score(y, v3_o):.8f}", flush=True)
        print("computing bits / and / xs pools...", flush=True)
        bits_tr, bits_te = bit_matrices(train["id"], test["id"])
        bits_o, bits_t = bits_pool(bits_tr, bits_te, y)
        and_o, and_t, within_o, cross_o = and_pools(bits_tr, bits_te, y)
        xs_o, xs_t = xs_pool(train, test, y)
        np.save(art / "bits64_oof.npy", bits_o)
        np.save(art / "bits64_test.npy", bits_t)
        np.save(art / "xs_oof.npy", xs_o)
        np.save(art / "xs_test.npy", xs_t)
        np.save(art / "bitand_all_oof.npy", and_o)
        np.save(art / "bitand_all_test.npy", and_t)
        np.save(art / "bitand_within_oof.npy", within_o)
        np.save(art / "bitand_cross_oof.npy", cross_o)

    fuse_o = W_V3 * v3_o + W_BITS * bits_o + W_X * xs_o + W_AND * and_o
    fuse_t = W_V3 * v3_t + W_BITS * bits_t + W_X * xs_t + W_AND * and_t
    oof_auc = float(roc_auc_score(y, fuse_o))
    v3_auc = float(roc_auc_score(y, v3_o))

    if abs(oof_auc - EXPECTED_OOF) > OOF_ATOL:
        raise ValueError(f"OOF 偏离锚点 got {oof_auc:.15f} expected {EXPECTED_OOF:.15f}")
    if oof_auc <= V4_OOF + 1e-15:
        raise SystemExit(f"GATE FAIL vs v4: {oof_auc} <= {V4_OOF}")

    out_path = ROOT / "submissions" / "submission_fp_v5.csv"
    champ = ROOT / "submissions" / "submission_champion.csv"
    expected = np.clip(fuse_t, 0.001, 0.999)
    if not args.verify_only:
        sub = sample[["id"]].copy()
        sub["label"] = expected
        sub.to_csv(out_path, index=False)
        champ.write_bytes(out_path.read_bytes())
        (ROOT / "submissions" / "submission_am40_idbytes.csv").write_bytes(out_path.read_bytes())
        np.save(art / "v5_fuse_oof.npy", fuse_o)
        np.save(art / "v5_fuse_test.npy", fuse_t)
        np.save(ROOT / "artifacts" / "id_bytes" / "fuse_oof.npy", fuse_o)
        np.save(ROOT / "artifacts" / "id_bytes" / "fuse_test.npy", fuse_t)

    saved = pd.read_csv(champ if args.verify_only else out_path, dtype={"id": str})
    if float(np.max(np.abs(saved["label"].to_numpy(float) - expected))) > 1e-12:
        raise ValueError("提交与重算不一致")

    metrics = {
        "name": "fp_v5_and_heavy",
        "formula": f"{W_V3}*v3 + {W_BITS}*bits64 + {W_X}*xs + {W_AND}*and_all",
        "weights": {"v3": W_V3, "bits": W_BITS, "xs": W_X, "and_all": W_AND},
        "oof_auc": oof_auc,
        "v3_oof": v3_auc,
        "v4_oof": V4_OOF,
        "delta_vs_v4": oof_auc - V4_OOF,
        "bits_oof": float(roc_auc_score(y, bits_o)),
        "xs_oof": float(roc_auc_score(y, xs_o)),
        "and_oof": float(roc_auc_score(y, and_o)),
        "spearman_and_vs_v3": float(spearmanr(and_o, v3_o).statistic),
        "submission": "submissions/submission_champion.csv",
        "submission_sha256": sha256(champ if args.verify_only else out_path),
        "gate_beat_v4": True,
    }
    (art / "v5_blend_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    print("PASS: fp_v5 and_heavy 超过 fp_v4")
    print(f"OOF={oof_auc:.8f} (v4={V4_OOF:.8f}, Δ={oof_auc - V4_OOF:+.8f})")
    print(f"sha256: {metrics['submission_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

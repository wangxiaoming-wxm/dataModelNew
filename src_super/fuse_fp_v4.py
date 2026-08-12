#!/usr/bin/env python3
"""第一性原理 v4：v3 dual champion + 全量 id bit 池 + 被忽略 x0..x18 TE 池。

证据来源（见 artifacts/first_principles/）：
1. build_main/alt 从未使用原始列 x0..x18（仅 x19/x20 作类别）
2. id 的 bit 平面 TE 与 v3 champion Spearman≈0.03，近乎正交
3. 选择无关协议：64 个 bit 全进池；x0..x18 全进池（不做特征挑选）

预注册公式：
    bits = mean_rank(TE_6seed(bit_b{0..7}_{0..7}))          # 64 keys
    xs   = mean_rank(TE_6seed(qbin20(x{0..18})))             # 19 keys
    score = 0.50 * v3_dual + 0.35 * bits + 0.15 * xs

其中 v3_dual 由 fuse_am40_idbytes 同款 AM40+id 字节/半字节池产生
（内联计算，不依赖事先写好的 npy）。

门禁：OOF > v3 dual；锚点 OOF ≈ 0.716405。
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

# --- AM40 / v3 dual constants (match fuse_am40_idbytes.py) ---
W_MAIN, W_ALT = 0.62, 0.38
ALPHA_MAX = 0.40
W_AM40_V3 = 0.55
W_V7_IN_POOL = 0.70
SPECS_V7 = ("b3", "b2hi", "b1hi", "b0", "b4hi", "b5", "b7", "b7hi", "b5hi", "b6hi")
SPECS_V2 = ("b0", "b4", "b5", "b7", "b2hi", "p47")
TE_SEEDS = (2026, 7, 42, 99, 314, 2718)
TE_SPLITS = 5
TE_SMOOTH = 20.0

# --- v4 blend ---
W_V3, W_BITS, W_X = 0.50, 0.35, 0.15
EXPECTED_OOF = 0.7164047305145615
OOF_ATOL = 1e-8
V3_OOF = 0.707165960500892


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


def keys_from_ids(ids: pd.Series, spec: str) -> pd.Series:
    s = ids.astype(str).str.lower()
    if spec.startswith("b") and spec.endswith("hi"):
        b = int(spec[1])
        return s.str.slice(2 * b, 2 * b + 1)
    if spec.startswith("b") and spec.endswith("lo"):
        b = int(spec[1])
        return s.str.slice(2 * b + 1, 2 * b + 2)
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


def v3_dual(train_ids: pd.Series, test_ids: pd.Series, y: np.ndarray, am40_o, am40_t):
    def specs_keys(specs):
        tr = [keys_from_ids(train_ids, s) for s in specs]
        te = [keys_from_ids(test_ids, s) for s in specs]
        return pool_keys(tr, te, y)

    v7_o, v7_t = specs_keys(SPECS_V7)
    v2_o, v2_t = specs_keys(SPECS_V2)
    ip_o = W_V7_IN_POOL * v7_o + (1.0 - W_V7_IN_POOL) * v2_o
    ip_t = W_V7_IN_POOL * v7_t + (1.0 - W_V7_IN_POOL) * v2_t
    return W_AM40_V3 * am40_o + (1.0 - W_AM40_V3) * ip_o, W_AM40_V3 * am40_t + (1.0 - W_AM40_V3) * ip_t


def bits_pool(train_ids: pd.Series, test_ids: pd.Series, y: np.ndarray):
    s = train_ids.astype(str).str.lower()
    st = test_ids.astype(str).str.lower()
    arr_tr = np.array([[int(s.iloc[i][j : j + 2], 16) for j in range(0, 16, 2)] for i in range(len(s))], np.int16)
    arr_te = np.array([[int(st.iloc[i][j : j + 2], 16) for j in range(0, 16, 2)] for i in range(len(st))], np.int16)
    tr_keys, te_keys = [], []
    for b in range(8):
        for bit in range(8):
            tr_keys.append(pd.Series(((arr_tr[:, b] >> bit) & 1).astype(str)))
            te_keys.append(pd.Series(((arr_te[:, b] >> bit) & 1).astype(str)))
    return pool_keys(tr_keys, te_keys, y)


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

    print("computing v3 dual...", flush=True)
    v3_o, v3_t = v3_dual(train["id"], test["id"], y, am40_o, am40_t)
    v3_auc = float(roc_auc_score(y, v3_o))
    print(f"  v3_dual OOF={v3_auc:.8f}", flush=True)

    print("computing bits64 pool...", flush=True)
    bits_o, bits_t = bits_pool(train["id"], test["id"], y)
    print(f"  bits OOF={roc_auc_score(y, bits_o):.8f}", flush=True)

    print("computing x0..x18 pool...", flush=True)
    xs_o, xs_t = xs_pool(train, test, y)
    print(f"  xs OOF={roc_auc_score(y, xs_o):.8f}", flush=True)

    fuse_o = W_V3 * v3_o + W_BITS * bits_o + W_X * xs_o
    fuse_t = W_V3 * v3_t + W_BITS * bits_t + W_X * xs_t
    oof_auc = float(roc_auc_score(y, fuse_o))

    if abs(oof_auc - EXPECTED_OOF) > OOF_ATOL:
        raise ValueError(f"OOF 偏离锚点 got {oof_auc:.15f} expected {EXPECTED_OOF:.15f}")
    if oof_auc <= v3_auc + 1e-15:
        raise SystemExit(f"GATE FAIL: {oof_auc} <= v3 {v3_auc}")

    out_path = ROOT / "submissions" / "submission_fp_v4.csv"
    champ = ROOT / "submissions" / "submission_champion.csv"
    alias = ROOT / "submissions" / "submission_am40_idbytes.csv"
    expected = np.clip(fuse_t, 0.001, 0.999)
    if not args.verify_only:
        sub = sample[["id"]].copy()
        sub["label"] = expected
        sub.to_csv(out_path, index=False)
        champ.write_bytes(out_path.read_bytes())
        alias.write_bytes(out_path.read_bytes())
        (ROOT / "submissions" / "submission_fp_richid.csv").write_bytes(out_path.read_bytes())

    saved = pd.read_csv(champ if args.verify_only else out_path, dtype={"id": str})
    if float(np.max(np.abs(saved["label"].to_numpy(float) - expected))) > 1e-12:
        raise ValueError("提交与重算不一致")

    art = ROOT / "artifacts" / "first_principles"
    art.mkdir(parents=True, exist_ok=True)
    np.save(art / "bits64_oof.npy", bits_o)
    np.save(art / "bits64_test.npy", bits_t)
    np.save(art / "xs_oof.npy", xs_o)
    np.save(art / "xs_test.npy", xs_t)
    np.save(art / "v4_fuse_oof.npy", fuse_o)
    np.save(art / "v4_fuse_test.npy", fuse_t)
    # also refresh id_bytes fuse pointers used elsewhere
    np.save(ROOT / "artifacts" / "id_bytes" / "fuse_oof.npy", fuse_o)
    np.save(ROOT / "artifacts" / "id_bytes" / "fuse_test.npy", fuse_t)

    metrics = {
        "name": "fp_v4_v3_bits_x",
        "formula": f"{W_V3}*v3_dual + {W_BITS}*bits64 + {W_X}*x0_18_q20",
        "weights": {"v3": W_V3, "bits": W_BITS, "xs": W_X},
        "te_seeds": list(TE_SEEDS),
        "oof_auc": oof_auc,
        "v3_oof": v3_auc,
        "delta_vs_v3": oof_auc - v3_auc,
        "bits_oof": float(roc_auc_score(y, bits_o)),
        "xs_oof": float(roc_auc_score(y, xs_o)),
        "spearman_bits_vs_v3": float(spearmanr(bits_o, v3_o).statistic),
        "spearman_xs_vs_v3": float(spearmanr(xs_o, v3_o).statistic),
        "submission": "submissions/submission_champion.csv",
        "submission_sha256": sha256(champ if args.verify_only else out_path),
        "gate_beat_v3": True,
        "data": {"train_rows": int(len(y)), "test_rows": int(len(test))},
        "ignored_raw_recovered": [f"x{i}" for i in range(19)],
        "richer_id": "all 64 bit planes of 8 hex bytes",
    }
    w62 = ROOT / "submissions" / "submission_w62.csv"
    if w62.is_file():
        metrics["spearman_vs_w62"] = float(
            spearmanr(saved["label"], pd.read_csv(w62)["label"]).statistic
        )
    (art / "v4_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    print("PASS: fp_v4 超过 v3 dual")
    print(f"OOF={oof_auc:.8f} (v3={v3_auc:.8f}, Δ={oof_auc - v3_auc:+.8f})")
    print(f"sha256: {metrics['submission_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

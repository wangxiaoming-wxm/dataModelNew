#!/usr/bin/env python3
"""第一性原理 v7：在 fp_v6 heavy_xor 上加入 cross-byte OR/XOR 池。

证据：within-byte OR/XOR 已抬升；cross-byte same-bit OR/XOR 与 heavy_xor
Spearman≈0.05–0.10，近正交。选择无关全量池，6-seed TE。

预注册主交（cross30）：
    cmean = 0.5 * pool(cross-byte OR) + 0.5 * pool(cross-byte XOR)
    score = 0.15*v3 + 0.10*bits + 0.05*xs + 0.22*and_all
          + 0.06*tri + 0.06*or + 0.06*xor + 0.30*cmean

门禁：nested fold-mean > fp_v6 heavy_xor；锚点 OOF ≈ 0.758380。
备份：tempered=heavy_xor；aggressive=0.5*heavy+0.5*cmean。
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

W = dict(v3=0.15, bits=0.10, xs=0.05, and_all=0.22, tri=0.06, or_=0.06, xor=0.06, cmean=0.30)
EXPECTED_OOF = 0.7583798471274498
OOF_ATOL = 1e-8
V6_OOF = 0.7434166163115536
V6_NEST = 0.743499816019302


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
    tr_w, te_w = [], []
    for b in range(8):
        for i in range(8):
            for j in range(i + 1, 8):
                tr_w.append(pd.Series((bits_tr[:, b * 8 + i] & bits_tr[:, b * 8 + j]).astype(str)))
                te_w.append(pd.Series((bits_te[:, b * 8 + i] & bits_te[:, b * 8 + j]).astype(str)))
    within_o, within_t = pool_keys(tr_w, te_w, y)
    tr_c, te_c = [], []
    for bit in range(8):
        for b1 in range(8):
            for b2 in range(b1 + 1, 8):
                tr_c.append(pd.Series((bits_tr[:, b1 * 8 + bit] & bits_tr[:, b2 * 8 + bit]).astype(str)))
                te_c.append(pd.Series((bits_te[:, b1 * 8 + bit] & bits_te[:, b2 * 8 + bit]).astype(str)))
    cross_o, cross_t = pool_keys(tr_c, te_c, y)
    return 0.5 * (within_o + cross_o), 0.5 * (within_t + cross_t)


def within_pair_op_pool(bits_tr, bits_te, y, op):
    tr_keys, te_keys = [], []
    for b in range(8):
        for i in range(8):
            for j in range(i + 1, 8):
                tr_keys.append(pd.Series(op(bits_tr[:, b * 8 + i], bits_tr[:, b * 8 + j]).astype(str)))
                te_keys.append(pd.Series(op(bits_te[:, b * 8 + i], bits_te[:, b * 8 + j]).astype(str)))
    return pool_keys(tr_keys, te_keys, y)


def within_tri_pool(bits_tr, bits_te, y):
    tr_keys, te_keys = [], []
    for b in range(8):
        for i in range(8):
            for j in range(i + 1, 8):
                for k in range(j + 1, 8):
                    tr_keys.append(
                        pd.Series((bits_tr[:, b * 8 + i] & bits_tr[:, b * 8 + j] & bits_tr[:, b * 8 + k]).astype(str))
                    )
                    te_keys.append(
                        pd.Series((bits_te[:, b * 8 + i] & bits_te[:, b * 8 + j] & bits_te[:, b * 8 + k]).astype(str))
                    )
    return pool_keys(tr_keys, te_keys, y)


def cross_pair_op_pool(bits_tr, bits_te, y, op):
    tr_keys, te_keys = [], []
    for bit in range(8):
        for b1 in range(8):
            for b2 in range(b1 + 1, 8):
                tr_keys.append(pd.Series(op(bits_tr[:, b1 * 8 + bit], bits_tr[:, b2 * 8 + bit]).astype(str)))
                te_keys.append(pd.Series(op(bits_te[:, b1 * 8 + bit], bits_te[:, b2 * 8 + bit]).astype(str)))
    return pool_keys(tr_keys, te_keys, y)


def xs_pool(train: pd.DataFrame, test: pd.DataFrame, y: np.ndarray):
    tr_keys, te_keys = [], []
    for i in range(19):
        col = f"x{i}"
        edges = np.quantile(train[col].to_numpy(float), np.linspace(0, 1, 21)[1:-1])
        tr_keys.append(pd.Series(np.digitize(train[col].to_numpy(float), edges).astype(str)))
        te_keys.append(pd.Series(np.digitize(test[col].to_numpy(float), edges).astype(str)))
    return pool_keys(tr_keys, te_keys, y)


def fuse(parts, weights):
    out = np.zeros_like(next(iter(parts.values())))
    for key, weight in weights.items():
        if weight:
            out = out + weight * parts[key]
    return out


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

    oof = np.load(ROOT / "artifacts" / "super714" / "best_v1_oof.npy", allow_pickle=True).item()
    te = np.load(ROOT / "artifacts" / "super714" / "best_v1_test.npy", allow_pickle=True).item()
    am40_o = am40(np.asarray(oof["main"], float), np.asarray(oof["alt"], float))
    am40_t = am40(np.asarray(te["main"], float), np.asarray(te["alt"], float))

    art = ROOT / "artifacts" / "first_principles"
    art.mkdir(parents=True, exist_ok=True)

    needed = (
        "bits64_oof.npy",
        "bits64_test.npy",
        "xs_oof.npy",
        "xs_test.npy",
        "bitand_all_oof.npy",
        "bitand_all_test.npy",
        "bitor_oof.npy",
        "bitor_test.npy",
        "bittri_oof.npy",
        "bittri_test.npy",
        "bitxor_oof.npy",
        "bitxor_test.npy",
        "bitor_cross_oof.npy",
        "bitor_cross_test.npy",
        "bitxor_cross_oof.npy",
        "bitxor_cross_test.npy",
        "v5_fuse_oof.npy",
        "v5_fuse_test.npy",
    )
    cache_ok = args.reuse_caches and all((art / p).is_file() for p in needed)

    if cache_ok:
        print("reusing cached pools...", flush=True)
        bits_o = np.load(art / "bits64_oof.npy")
        bits_t = np.load(art / "bits64_test.npy")
        xs_o = np.load(art / "xs_oof.npy")
        xs_t = np.load(art / "xs_test.npy")
        and_o = np.load(art / "bitand_all_oof.npy")
        and_t = np.load(art / "bitand_all_test.npy")
        or_o = np.load(art / "bitor_oof.npy")
        or_t = np.load(art / "bitor_test.npy")
        tri_o = np.load(art / "bittri_oof.npy")
        tri_t = np.load(art / "bittri_test.npy")
        xor_o = np.load(art / "bitxor_oof.npy")
        xor_t = np.load(art / "bitxor_test.npy")
        cor_o = np.load(art / "bitor_cross_oof.npy")
        cor_t = np.load(art / "bitor_cross_test.npy")
        cx_o = np.load(art / "bitxor_cross_oof.npy")
        cx_t = np.load(art / "bitxor_cross_test.npy")
        v5_o = np.load(art / "v5_fuse_oof.npy")
        v5_t = np.load(art / "v5_fuse_test.npy")
        v3_o = (v5_o - 0.25 * bits_o - 0.10 * xs_o - 0.35 * and_o) / 0.30
        v3_t = (v5_t - 0.25 * bits_t - 0.10 * xs_t - 0.35 * and_t) / 0.30
    else:
        print("computing pools (full)...", flush=True)
        v3_o, v3_t = v3_dual(train["id"], test["id"], y, am40_o, am40_t)
        bits_tr, bits_te = bit_matrices(train["id"], test["id"])
        bits_o, bits_t = bits_pool(bits_tr, bits_te, y)
        and_o, and_t = and_pools(bits_tr, bits_te, y)
        or_o, or_t = within_pair_op_pool(bits_tr, bits_te, y, np.bitwise_or)
        xor_o, xor_t = within_pair_op_pool(bits_tr, bits_te, y, np.bitwise_xor)
        tri_o, tri_t = within_tri_pool(bits_tr, bits_te, y)
        cor_o, cor_t = cross_pair_op_pool(bits_tr, bits_te, y, np.bitwise_or)
        cx_o, cx_t = cross_pair_op_pool(bits_tr, bits_te, y, np.bitwise_xor)
        xs_o, xs_t = xs_pool(train, test, y)
        for name, arr in (
            ("bits64_oof", bits_o),
            ("bits64_test", bits_t),
            ("xs_oof", xs_o),
            ("xs_test", xs_t),
            ("bitand_all_oof", and_o),
            ("bitand_all_test", and_t),
            ("bitor_oof", or_o),
            ("bitor_test", or_t),
            ("bittri_oof", tri_o),
            ("bittri_test", tri_t),
            ("bitxor_oof", xor_o),
            ("bitxor_test", xor_t),
            ("bitor_cross_oof", cor_o),
            ("bitor_cross_test", cor_t),
            ("bitxor_cross_oof", cx_o),
            ("bitxor_cross_test", cx_t),
        ):
            np.save(art / f"{name}.npy", arr)

    cmean_o = 0.5 * (cor_o + cx_o)
    cmean_t = 0.5 * (cor_t + cx_t)
    np.save(art / "bitcross_mean_oof.npy", cmean_o)
    np.save(art / "bitcross_mean_test.npy", cmean_t)

    parts_o = {
        "v3": v3_o,
        "bits": bits_o,
        "xs": xs_o,
        "and_all": and_o,
        "tri": tri_o,
        "or_": or_o,
        "xor": xor_o,
        "cmean": cmean_o,
    }
    parts_t = {
        "v3": v3_t,
        "bits": bits_t,
        "xs": xs_t,
        "and_all": and_t,
        "tri": tri_t,
        "or_": or_t,
        "xor": xor_t,
        "cmean": cmean_t,
    }

    fuse_o = fuse(parts_o, W)
    fuse_t = fuse(parts_t, W)
    oof_auc = float(roc_auc_score(y, fuse_o))
    if abs(oof_auc - EXPECTED_OOF) > OOF_ATOL:
        raise ValueError(f"OOF 偏离锚点 got {oof_auc:.15f} expected {EXPECTED_OOF:.15f}")
    if oof_auc <= V6_OOF + 1e-15:
        raise SystemExit(f"GATE FAIL vs v6: {oof_auc} <= {V6_OOF}")

    heavy_w = dict(v3=0.20, bits=0.15, xs=0.05, and_all=0.30, tri=0.10, or_=0.10, xor=0.10, cmean=0.0)
    heavy_o = fuse(parts_o, heavy_w)
    heavy_t = fuse(parts_t, heavy_w)
    agg_o = 0.5 * heavy_o + 0.5 * cmean_o
    agg_t = 0.5 * heavy_t + 0.5 * cmean_t

    skf = StratifiedKFold(5, shuffle=True, random_state=2026)
    nest = float(np.mean([roc_auc_score(y[va], fuse_o[va]) for _, va in skf.split(np.zeros(len(y)), y)]))
    if nest <= V6_NEST + 1e-15:
        raise SystemExit(f"NEST GATE FAIL vs v6: {nest} <= {V6_NEST}")

    out_path = ROOT / "submissions" / "submission_fp_v7.csv"
    champ = ROOT / "submissions" / "submission_champion.csv"
    tempered_path = ROOT / "submissions" / "submission_fp_v7_tempered.csv"
    aggressive_path = ROOT / "submissions" / "submission_fp_v7_aggressive.csv"
    expected = np.clip(fuse_t, 0.001, 0.999)

    if not args.verify_only:
        sub = sample[["id"]].copy()
        sub["label"] = expected
        sub.to_csv(out_path, index=False)
        # 不覆盖全局 champion（由更高版本持有）

        tsub = sample[["id"]].copy()
        tsub["label"] = np.clip(heavy_t, 0.001, 0.999)
        tsub.to_csv(tempered_path, index=False)
        (ROOT / "submissions" / "submission_fp_v6.csv").write_bytes(tempered_path.read_bytes())

        asub = sample[["id"]].copy()
        asub["label"] = np.clip(agg_t, 0.001, 0.999)
        asub.to_csv(aggressive_path, index=False)

        np.save(art / "v7_fuse_oof.npy", fuse_o)
        np.save(art / "v7_fuse_test.npy", fuse_t)

    # 全局 champion 由更高版本持有；校验以 submission_fp_v7.csv 为准
    check_path = out_path if out_path.is_file() else champ
    saved = pd.read_csv(check_path, dtype={"id": str})
    if float(np.max(np.abs(saved["label"].to_numpy(float) - expected))) > 1e-12:
        raise ValueError("提交与重算不一致")

    metrics = {
        "name": "fp_v7_cross30",
        "formula": (
            "0.15*v3 + 0.10*bits + 0.05*xs + 0.22*and_all + 0.06*tri + 0.06*or + 0.06*xor + 0.30*cmean"
        ),
        "weights": {("or" if k == "or_" else k): v for k, v in W.items()},
        "oof_auc": oof_auc,
        "nested_fold_mean": nest,
        "delta_nest_vs_v6": nest - V6_NEST,
        "delta_full_vs_v6": oof_auc - V6_OOF,
        "arm_aucs": {
            "v3": float(roc_auc_score(y, v3_o)),
            "bits": float(roc_auc_score(y, bits_o)),
            "xs": float(roc_auc_score(y, xs_o)),
            "and_all": float(roc_auc_score(y, and_o)),
            "or": float(roc_auc_score(y, or_o)),
            "tri": float(roc_auc_score(y, tri_o)),
            "xor": float(roc_auc_score(y, xor_o)),
            "cross_or": float(roc_auc_score(y, cor_o)),
            "cross_xor": float(roc_auc_score(y, cx_o)),
            "cmean": float(roc_auc_score(y, cmean_o)),
        },
        "spearman_cmean_vs_heavy": float(spearmanr(cmean_o, heavy_o).statistic),
        "tempered": {
            "name": "fp_v6_heavy_xor",
            "oof": float(roc_auc_score(y, heavy_o)),
            "sha256": sha256(tempered_path) if tempered_path.is_file() else None,
        },
        "aggressive": {
            "name": "heavy_cmean50",
            "oof": float(roc_auc_score(y, agg_o)),
            "sha256": sha256(aggressive_path) if aggressive_path.is_file() else None,
        },
        "fp_v6_oof": V6_OOF,
        "submission": "submissions/submission_fp_v7.csv",
        "submission_sha256": sha256(check_path),
        "EXPECTED_OOF": EXPECTED_OOF,
        "gate_beat_v6": True,
        "note": "global champion may be newer; v7 retained as tempered backup",
    }
    (art / "v7_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    print("PASS: fp_v7 cross30 超过 fp_v6")
    print(f"OOF={oof_auc:.8f} nest={nest:.8f} (v6={V6_OOF:.8f}, Δ={oof_auc - V6_OOF:+.8f})")
    print(f"sha256: {metrics['submission_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

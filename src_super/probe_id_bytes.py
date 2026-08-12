#!/usr/bin/env python3
"""验证 id 十六进制字节是否含 fold-local 真信号，以及加入 CatBoost 后 OOF 是否上涨。

协议：
- id 为 16 位 hex → 8 个 byte（0..7）
- fold-local TE：外层 5-fold；编码仅用训练折；验证折只读映射（无泄漏）
- CatBoost 对照：同一 CV，build_main 基线 vs + id_byte 类别特征
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src_super"))
from train_super714 import build_main, fit_edges_main, resolve_data_dir  # noqa: E402

N_SPLITS = 5
SEED = 2026
ITER = 800
LR = 0.03


def id_bytes(ids: pd.Series) -> pd.DataFrame:
    s = ids.astype(str).str.lower()
    out = {}
    for b in range(8):
        out[f"id_b{b}"] = s.str.slice(2 * b, 2 * b + 2)
    return pd.DataFrame(out, index=ids.index)


def fold_local_te_auc(keys: pd.Series, y: np.ndarray, n_splits: int = 5, seed: int = 2026, smooth: float = 20.0):
    """返回 fold-local TE 的 OOF AUC 与 |sig|=|auc-0.5|。"""
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    prior = float(y.mean())
    for tri, vali in skf.split(np.zeros(len(y)), y):
        tr_key = keys.iloc[tri]
        tr_y = y[tri]
        stats = pd.DataFrame({"k": tr_key.to_numpy(), "y": tr_y}).groupby("k")["y"].agg(["sum", "count"])
        te = (stats["sum"] + smooth * prior) / (stats["count"] + smooth)
        mapped = keys.iloc[vali].map(te).fillna(prior).to_numpy(dtype=float)
        oof[vali] = mapped
    auc = float(roc_auc_score(y, oof))
    return auc, abs(auc - 0.5), oof


def run_cb(xtr, xte, y, cats, ordered=True, depth=5, l2=10, thread_count=3):
    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    te = np.zeros(len(xte))
    for tri, vali in skf.split(xtr, y):
        kw = dict(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=ITER,
            learning_rate=LR,
            depth=depth,
            l2_leaf_reg=l2,
            random_strength=0.7,
            verbose=0,
            allow_writing_files=False,
            thread_count=thread_count,
            random_seed=SEED,
        )
        if ordered:
            kw["boosting_type"] = "Ordered"
        model = CatBoostRegressor(**kw)
        model.fit(Pool(xtr.iloc[tri], y[tri], cat_features=cats), verbose=False)
        oof[vali] = model.predict(xtr.iloc[vali])
        te += model.predict(xte)
    te /= N_SPLITS
    return float(roc_auc_score(y, oof)), oof, te


def main() -> int:
    data_dir = resolve_data_dir(None)
    train = pd.read_csv(data_dir / "train.csv", dtype={"id": str})
    test = pd.read_csv(data_dir / "test.csv", dtype={"id": str})
    y = train["label"].astype(int).to_numpy()
    raw_all = pd.concat([train.drop(columns=["label"]), test], ignore_index=True)

    print("=== id bytes fold-local TE ===", flush=True)
    btr = id_bytes(train["id"])
    te_rows = []
    for b in range(8):
        auc, sig, _ = fold_local_te_auc(btr[f"id_b{b}"], y)
        flag = "★" if sig >= 0.01 else ("·" if sig >= 0.005 else "")
        te_rows.append({"byte": b, "te_auc": auc, "abs_sig": sig, "flag": flag})
        print(f"  byte{b}: TE_AUC={auc:.4f} |sig|={sig:.4f} {flag}", flush=True)
    # combo key of strong bytes
    strong = [r["byte"] for r in te_rows if r["abs_sig"] >= 0.01]
    if strong:
        combo = btr[[f"id_b{b}" for b in strong]].astype(str).agg("|".join, axis=1)
        auc, sig, _ = fold_local_te_auc(combo, y)
        print(f"  combo{strong}: TE_AUC={auc:.4f} |sig|={sig:.4f}", flush=True)

    print("\n=== CatBoost: baseline build_main vs +id_bytes ===", flush=True)
    t0 = time.time()
    edges = fit_edges_main(raw_all)
    x_all, cats = build_main(raw_all, edges)
    for c in cats:
        x_all[c] = x_all[c].astype(str)
    xtr = x_all.iloc[: len(train)].reset_index(drop=True)
    xte = x_all.iloc[len(train) :].reset_index(drop=True)

    # limit threads so jitter training can continue
    base_auc, base_oof, _ = run_cb(xtr, xte, y, cats, thread_count=3)
    print(f"baseline main OOF={base_auc:.5f}", flush=True)

    b_all = id_bytes(raw_all["id"])
    xtr_id = xtr.copy()
    xte_id = xte.copy()
    cats_id = list(cats)
    for b in range(8):
        col = f"id_b{b}"
        xtr_id[col] = b_all.iloc[: len(train)][col].astype(str).to_numpy()
        xte_id[col] = b_all.iloc[len(train) :][col].astype(str).to_numpy()
        cats_id.append(col)
    # also add strong-byte crosses with source/region
    for b in strong or [0, 4, 5, 7]:
        for other in ("source", "region"):
            col = f"id_b{b}_{other}"
            xtr_id[col] = xtr_id[f"id_b{b}"] + "|" + xtr_id[other].astype(str)
            xte_id[col] = xte_id[f"id_b{b}"] + "|" + xte_id[other].astype(str)
            cats_id.append(col)

    id_auc, id_oof, id_te = run_cb(xtr_id, xte_id, y, cats_id, thread_count=3)
    print(f"main+id_bytes OOF={id_auc:.5f}  Δ={id_auc-base_auc:+.5f}", flush=True)

    # strong bytes only
    xtr_s = xtr.copy()
    xte_s = xte.copy()
    cats_s = list(cats)
    use_bytes = strong or [0, 4, 5, 7]
    for b in use_bytes:
        col = f"id_b{b}"
        xtr_s[col] = b_all.iloc[: len(train)][col].astype(str).to_numpy()
        xte_s[col] = b_all.iloc[len(train) :][col].astype(str).to_numpy()
        cats_s.append(col)
    s_auc, s_oof, s_te = run_cb(xtr_s, xte_s, y, cats_s, thread_count=3)
    print(f"main+strong_bytes{use_bytes} OOF={s_auc:.5f}  Δ={s_auc-base_auc:+.5f}", flush=True)

    art = ROOT / "artifacts" / "id_bytes"
    art.mkdir(parents=True, exist_ok=True)
    metrics = {
        "te_by_byte": te_rows,
        "strong_bytes": use_bytes,
        "catboost": {
            "baseline_main_oof": base_auc,
            "main_all_id_bytes_oof": id_auc,
            "delta_all": id_auc - base_auc,
            "main_strong_id_bytes_oof": s_auc,
            "delta_strong": s_auc - base_auc,
        },
        "elapsed_minutes": (time.time() - t0) / 60,
        "note": "Single-seed 5fold Ordered d5; compare Δ vs baseline only.",
    }
    (art / "probe_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    np.save(art / "oof_baseline.npy", base_oof)
    np.save(art / "oof_id_all.npy", id_oof)
    np.save(art / "oof_id_strong.npy", s_oof)
    print(json.dumps(metrics["catboost"], indent=2), flush=True)
    print(f"saved {art}/probe_metrics.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

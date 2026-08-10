"""alt2-repair: restore cond_r/ratio and shrink region×source z (gpt56 S3).

Honest fixed trees, 5-fold, Plain boosting (alt world family). Only enters
max fusion if bag ≥ ~0.694 and improves nested max3.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src2"))
from features import (  # noqa: E402
    BIN_COLS,
    GRADE_MAP,
    _derive,
    _qbins,
    add_noise_view,
)
from jitter import add_jitter_views  # noqa: E402

ART = ROOT / "artifacts" / "v4max3pro"
DATA = ROOT / "data"
QUANTS = (4, 9, 16)
K_SHRINK = 50.0


def _shrunk_z(df: pd.DataFrame) -> pd.Series:
    """Median/IQR z inside region×source, shrunk toward source-level stats."""
    cond = pd.to_numeric(df["condition"])
    # source-level
    s_med = df.groupby("source")["condition"].transform("median")
    s_iqr = (
        df.groupby("source")["condition"].transform(lambda s: s.quantile(0.75) - s.quantile(0.25))
    ).replace(0, np.nan)
    # region×source
    gcols = ["source", "region"]
    g_med = df.groupby(gcols)["condition"].transform("median")
    g_iqr = (
        df.groupby(gcols)["condition"].transform(lambda s: s.quantile(0.75) - s.quantile(0.25))
    ).replace(0, np.nan)
    n = df.groupby(gcols)["condition"].transform("count")
    w = n / (n + K_SHRINK)
    center = w * g_med + (1 - w) * s_med
    scale = (w * g_iqr + (1 - w) * s_iqr).clip(lower=1e-6)
    return ((cond - center) / scale).fillna(0.0).clip(-8, 8)


def fit_edges_alt2fix(df: pd.DataFrame) -> dict:
    scale = df.groupby("source")["condition"].median()
    der = _derive(df, scale)
    cz = _shrunk_z(df)
    dpc = df.groupby("region")["days"].rank(pct=True)
    edges: dict = {"__scale__": scale}
    for n in QUANTS:
        qs = np.linspace(0, 1, n + 1)[1:-1]
        edges[f"cz_{n}"] = np.quantile(cz, qs)
        edges[f"dpc_{n}"] = np.quantile(dpc, qs)
        edges[f"condr_{n}"] = np.quantile(der["cond_r"], qs)
        edges[f"ratio_{n}"] = np.quantile(der["ratio"], qs)
        edges[f"days_{n}"] = np.quantile(df["days"].dropna(), qs)
    return edges


def build_alt2fix(df: pd.DataFrame, edges: dict) -> tuple[pd.DataFrame, list[str]]:
    out = pd.DataFrame(index=df.index)
    days = pd.to_numeric(df["days"])
    cond = pd.to_numeric(df["condition"])
    der = _derive(df, edges["__scale__"])
    cz = _shrunk_z(df)
    dpc = df.groupby("region")["days"].rank(pct=True)

    out["days"] = days
    out["log_days"] = np.log1p(days.clip(lower=0))
    out["condition"] = cond
    out["condition_missing"] = cond.isna().astype(int)
    out["cond_r"] = der["cond_r"]
    out["log_cond_r"] = np.log(der["cond_r"].clip(lower=1e-9))
    out["ratio"] = der["ratio"]
    out["log_ratio"] = np.log(der["ratio"].clip(lower=1e-9))
    out["ratio_p75"] = days / der["cond_r"].clip(lower=1e-9) ** 0.75
    out["cz"] = cz
    out["dpc"] = dpc
    out["age_range"] = df["age_range"].astype(float)
    out["grade_ord"] = df["grades"].map(GRADE_MAP).astype(float)
    for c in BIN_COLS:
        out[c] = df[c].astype(int)
    out["bin_sum"] = df[BIN_COLS].sum(axis=1)

    cats: list[str] = []
    out["region"] = df["region"].astype(str)
    out["source"] = df["source"].astype(str)
    out["month"] = df["month"].astype(str)
    out["version"] = df["version"].astype(str)
    out["grades_c"] = df["grades"].astype(str)
    out["age_cat"] = df["age_range"].astype(str)
    out["bin_pat"] = df[BIN_COLS].astype(str).agg("".join, axis=1)
    cats += ["region", "source", "month", "version", "grades_c", "age_cat", "bin_pat"]

    for n in QUANTS:
        out[f"z{n}"] = _qbins(cz, edges[f"cz_{n}"]).astype(str)
        out[f"p{n}"] = _qbins(dpc, edges[f"dpc_{n}"]).astype(str)
        out[f"cr{n}"] = _qbins(der["cond_r"], edges[f"condr_{n}"]).astype(str)
        out[f"r{n}"] = _qbins(der["ratio"], edges[f"ratio_{n}"]).astype(str)
        out[f"d{n}"] = _qbins(days, edges[f"days_{n}"]).astype(str)
        cats += [f"z{n}", f"p{n}", f"cr{n}", f"r{n}", f"d{n}"]

    def cross(name, *parts):
        s = out[parts[0]].astype(str)
        for p in parts[1:]:
            s = s + "|" + out[p].astype(str)
        out[name] = s
        cats.append(name)

    cross("R_z9_src", "z9", "source")
    cross("R_z16_src", "z16", "source")
    cross("R_z9_reg", "z9", "region")
    cross("R_cr9_src", "cr9", "source")
    cross("R_cr9_reg", "cr9", "region")
    cross("R_r9_src", "r9", "source")
    cross("R_r9_reg", "r9", "region")
    cross("R_d9_reg", "d9", "region")
    cross("R_d9_src", "d9", "source")
    cross("R_reg_src", "region", "source")
    cross("R_reg_age", "region", "age_cat")
    cross("R_src_age", "source", "age_cat")
    cross("R_cr9_age", "cr9", "age_cat")
    cross("R_d9_cr9", "d9", "cr9")
    cross("R_z9_age", "z4", "age_cat")
    return out, cats


def alt2fix_frame(raw, edges, stream_offset: int):
    X, cats = build_alt2fix(raw, edges)
    add_noise_view(X, cats, raw)
    der = _derive(raw, edges["__scale__"])
    add_jitter_views(
        X,
        cats,
        raw,
        der["cond_r"],
        pd.to_numeric(raw["days"]),
        n_views=3,
        n_bins=9,
        stream_offset=200 + stream_offset,
    )
    for c in cats:
        X[c] = X[c].astype(str)
    num = [c for c in X.columns if c not in cats]
    X[num] = X[num].astype(float).fillna(-999.0)
    return X, cats


def run_seed(seed: int, stream_offset: int, threads: int):
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["label"].astype(int).values
    raw = pd.concat([train.drop(columns=["label"]), test], ignore_index=True)
    edges = fit_edges_alt2fix(raw)
    X, cats = alt2fix_frame(raw, edges, stream_offset)
    Xtr = X.iloc[: len(train)].reset_index(drop=True)
    Xte = X.iloc[len(train) :].reset_index(drop=True)
    params = dict(
        loss_function="Logloss",
        learning_rate=0.03,
        l2_leaf_reg=14,
        random_strength=1.0,
        depth=6,
        iterations=900,
        verbose=False,
        thread_count=threads,
        allow_writing_files=False,
    )
    oof = np.zeros(len(y))
    te = np.zeros(len(test))
    t0 = time.time()
    for fold, (ti, vi) in enumerate(
        StratifiedKFold(5, shuffle=True, random_state=seed).split(Xtr, y)
    ):
        m = CatBoostClassifier(**dict(params, random_seed=seed + fold))
        m.fit(Xtr.iloc[ti], y[ti], cat_features=cats, verbose=False)
        oof[vi] = m.predict_proba(Xtr.iloc[vi])[:, 1]
        te += m.predict_proba(Xte)[:, 1] / 5
    auc = float(roc_auc_score(y, oof))
    print(f"[alt2fix] seed={seed} OOF={auc:.6f} ({time.time()-t0:.0f}s)", flush=True)
    ART.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ART / f"part_alt2fix_s{seed}.npz", oof=oof, test=te, y=y)
    (ART / f"part_alt2fix_s{seed}.json").write_text(
        json.dumps({"seed": seed, "oof_auc": auc, "elapsed_sec": round(time.time() - t0, 1)}, indent=2)
    )
    return auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--stream-offset", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    run_seed(args.seed, args.stream_offset, args.threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

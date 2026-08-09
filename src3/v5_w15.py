"""V5 world w15: alt-style world driven by ``x20`` (noisy condition) instead of condition.

``x20 ≈ 1.2·condition + U(-1.5,1.5)`` (docs/DATA_STRUCTURE.md).  Building a full
encoding world on x20 gives a *jittered* copy of the same interactions — the
same mechanism that makes the organisers' noise columns useful inside one
model, but as a separate arm for ``max`` fusion.

Recipe mirrors ``build_alt`` / ``altboost_frame``:
  * rk = rank(x20 | source)
  * rate = days * (1 - rk)
  * bins (7, 13, 25), same cross list
  * also keep a thin cond_r/ratio numeric pair so the arm does not go soft

Label-free.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features import BIN_COLS, GRADE_MAP, DAYS_FIXED_EDGES, add_noise_view, _derive, _qbins
from jitter import add_jitter_views

W15_Q = (7, 13, 25)


def fit_edges_w15(df: pd.DataFrame) -> dict:
    x20 = pd.to_numeric(df["x20"])
    rk = x20.groupby(df["source"]).rank(pct=True).fillna(0.5)
    days = pd.to_numeric(df["days"])
    rate = days * (1.0 - rk)
    scale = df.groupby("source")["condition"].median()
    der = _derive(df, scale)
    edges: dict = {"__scale__": scale}
    for n in W15_Q:
        qs = np.linspace(0, 1, n + 1)[1:-1]
        edges[f"days_{n}"] = np.quantile(days.dropna(), qs)
        edges[f"crk_{n}"] = np.quantile(rk, qs)
        edges[f"rate_{n}"] = np.quantile(rate, qs)
        edges[f"condr_{n}"] = np.quantile(der["cond_r"], qs)
        edges[f"ratio_{n}"] = np.quantile(der["ratio"], qs)
    return edges


def build_w15(df: pd.DataFrame, edges: dict) -> tuple[pd.DataFrame, list[str]]:
    out = pd.DataFrame(index=df.index)
    days = pd.to_numeric(df["days"])
    x20 = pd.to_numeric(df["x20"])
    cond = pd.to_numeric(df["condition"])
    rk = x20.groupby(df["source"]).rank(pct=True).fillna(0.5)
    rate = days * (1.0 - rk)
    der = _derive(df, edges["__scale__"])

    out["days"] = days
    out["sqrt_days"] = np.sqrt(days.clip(lower=0))
    out["x20"] = x20
    out["condition"] = cond
    out["x20_rk"] = rk
    out["rate"] = rate
    out["log_rate"] = np.log1p(rate.clip(lower=0))
    out["rate_over_age"] = rate / df["age_range"].astype(float)
    out["cond_r"] = der["cond_r"]
    out["log_cond_r"] = np.log(der["cond_r"].clip(lower=1e-9))
    out["ratio"] = der["ratio"]
    out["log_ratio"] = np.log(der["ratio"].clip(lower=1e-9))
    out["condition_missing"] = cond.isna().astype(int)
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
    out["days_fx"] = np.digitize(days.to_numpy(dtype=float), DAYS_FIXED_EDGES).astype(str)
    cats += ["region", "source", "month", "version", "grades_c", "age_cat", "bin_pat", "days_fx"]

    for n in W15_Q:
        out[f"d{n}"] = _qbins(days, edges[f"days_{n}"]).astype(str)
        out[f"k{n}"] = _qbins(rk, edges[f"crk_{n}"]).astype(str)
        out[f"e{n}"] = _qbins(rate, edges[f"rate_{n}"]).astype(str)
        out[f"cr{n}"] = _qbins(der["cond_r"], edges[f"condr_{n}"]).astype(str)
        out[f"rr{n}"] = _qbins(der["ratio"], edges[f"ratio_{n}"]).astype(str)
        cats += [f"d{n}", f"k{n}", f"e{n}", f"cr{n}", f"rr{n}"]

    def cross(name, *parts):
        s = out[parts[0]].astype(str)
        for p in parts[1:]:
            s = s + "|" + out[p].astype(str)
        out[name] = s
        cats.append(name)

    # alt-style crosses on x20-rank / rate
    cross("A_k7_src", "k7", "source")
    cross("A_k13_src", "k13", "source")
    cross("A_k25_src", "k25", "source")
    cross("A_k13_reg", "k13", "region")
    cross("A_k7_age", "k7", "age_cat")
    cross("A_d13_reg", "d13", "region")
    cross("A_d13_src", "d13", "source")
    cross("A_d7_age", "d7", "age_cat")
    cross("A_d25_reg", "d25", "region")
    cross("A_e13_reg", "e13", "region")
    cross("A_e13_src", "e13", "source")
    cross("A_e7_age", "e7", "age_cat")
    cross("A_e7_pat", "e7", "bin_pat")
    cross("A_d7_k7", "d7", "k7")
    cross("A_d13_k13", "d13", "k13")
    cross("A_reg_src", "region", "source")
    cross("A_reg_age", "region", "age_cat")
    cross("A_src_age", "source", "age_cat")
    cross("A_d7_reg_src", "d7", "region", "source")
    cross("A_k7_reg_age", "k7", "region", "age_cat")
    cross("A_e7_reg_src", "e7", "region", "source")
    cross("A_d7_pat", "d7", "bin_pat")
    cross("A_reg_pat", "region", "bin_pat")
    cross("A_dfx_src", "days_fx", "source")
    cross("A_dfx_k13", "days_fx", "k13")
    cross("A_dfx_e13", "days_fx", "e13")
    # thin main-signal crosses so the arm stays strong
    cross("G_cr13_src", "cr13", "source")
    cross("G_rr13_src", "rr13", "source")
    cross("G_dfx_cr13", "days_fx", "cr13")
    cross("G_dfx_rr13", "days_fx", "rr13")

    for c in ("region", "source", "bin_pat", "A_reg_src", "A_k13_src", "A_d13_reg"):
        out[f"freq_{c}"] = out[c].map(out[c].value_counts()).astype(float)
    return out, cats


def w15_frame(raw: pd.DataFrame, edges: dict, stream_offset: int, n_views: int = 3):
    X, cats = build_w15(raw, edges)
    add_noise_view(X, cats, raw)
    x20 = pd.to_numeric(raw["x20"])
    rk = x20.groupby(raw["source"]).rank(pct=True).fillna(0.5)
    add_jitter_views(X, cats, raw, rk, pd.to_numeric(raw["days"]),
                     n_views=n_views, n_bins=13, stream_offset=130 + stream_offset)
    for c in cats:
        X[c] = X[c].astype(str)
    num = [c for c in X.columns if c not in cats]
    X[num] = X[num].astype(float).fillna(-999.0)
    return X, cats

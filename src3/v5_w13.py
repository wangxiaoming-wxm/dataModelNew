"""V5 world w13: strengthen the near-miss w5 world (bag was 0.69366).

w5 already keeps cond_r/ratio and adds within-source percentile cells.
It missed the gate by ~0.0003 and soft-blend showed a tiny positive
(+0.0002 nested) — so the recipe is right directionally.

w13 pushes the same idea harder, still label-free:
  * bin grid (8, 15, 25) instead of (8, 15);
  * add alt's rate = days*(1-cp) as a parallel numeric + e*_src crosses;
  * add the main-style cr/ratio × source × age triple crosses;
  * jitter on diag with a fresh stream family.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features import BIN_COLS, GRADE_MAP, DAYS_FIXED_EDGES, add_noise_view
from jitter import add_jitter_views
from worlds import _qbins, _cross

W13_BINS = (8, 15, 25)


def _w13_core(df: pd.DataFrame) -> pd.DataFrame:
    cond = pd.to_numeric(df["condition"])
    days = pd.to_numeric(df["days"])
    cp = df.groupby("source")["condition"].rank(pct=True).fillna(0.5)
    dp = df.groupby("source")["days"].rank(pct=True).fillna(0.5)
    diag = dp - cp
    anti = (dp + cp) / 2.0
    med = df.groupby("source")["condition"].transform("median")
    cond_r = (cond / med.replace(0, np.nan)).fillna(1.0)
    ratio = days / cond_r.clip(lower=1e-9)
    rate = days * (1.0 - cp)
    return pd.DataFrame({
        "cp": cp, "dp": dp, "diag": diag, "anti": anti,
        "days": days, "cond": cond, "cond_r": cond_r, "ratio": ratio, "rate": rate,
    }, index=df.index)


def fit_edges_w13(df: pd.DataFrame) -> dict:
    c = _w13_core(df)
    edges: dict = {}
    for n in W13_BINS:
        qs = np.linspace(0, 1, n + 1)[1:-1]
        edges[f"cp_{n}"] = np.quantile(c["cp"], qs)
        edges[f"dp_{n}"] = np.quantile(c["dp"], qs)
        edges[f"dg_{n}"] = np.quantile(c["diag"], qs)
        edges[f"an_{n}"] = np.quantile(c["anti"], qs)
        edges[f"rt_{n}"] = np.quantile(c["ratio"], qs)
        edges[f"cr_{n}"] = np.quantile(c["cond_r"], qs)
        edges[f"er_{n}"] = np.quantile(c["rate"], qs)
    edges["dq_5x"] = np.quantile(c["days"], np.linspace(0, 1, 6)[1:-1])
    return edges


def build_w13(df: pd.DataFrame, edges: dict) -> tuple[pd.DataFrame, list[str]]:
    c = _w13_core(df)
    out = pd.DataFrame(index=df.index)
    out["cp"] = c["cp"]
    out["dp"] = c["dp"]
    out["diag"] = c["diag"]
    out["anti"] = c["anti"]
    out["ratio"] = c["ratio"]
    out["log_ratio"] = np.log(c["ratio"].clip(lower=1e-9))
    out["cond_r"] = c["cond_r"]
    out["log_cond_r"] = np.log(c["cond_r"].clip(lower=1e-9))
    out["rate"] = c["rate"]
    out["log_rate"] = np.log1p(c["rate"].clip(lower=0))
    out["days"] = c["days"]
    out["condition"] = c["cond"]
    out["condition_missing"] = c["cond"].isna().astype(int)
    out["diag_x_age"] = c["diag"] * df["age_range"].astype(float)
    out["ratio_p75"] = c["days"] / c["cond_r"].clip(lower=1e-9) ** 0.75
    out["age_range"] = df["age_range"].astype(float)
    out["grade_ord"] = df["grades"].map(GRADE_MAP).astype(float)
    for b in BIN_COLS:
        out[b] = df[b].astype(int)
    out["bin_sum"] = df[BIN_COLS].sum(axis=1)

    cats: list[str] = []
    out["region"] = df["region"].astype(str)
    out["source"] = df["source"].astype(str)
    out["month"] = df["month"].astype(str)
    out["version"] = df["version"].astype(str)
    out["grades_c"] = df["grades"].astype(str)
    out["age_cat"] = df["age_range"].astype(str)
    out["bin_pat"] = df[BIN_COLS].astype(str).agg("".join, axis=1)
    out["days_fx"] = np.digitize(c["days"].to_numpy(dtype=float), DAYS_FIXED_EDGES).astype(str)
    cats += ["region", "source", "month", "version", "grades_c", "age_cat",
             "bin_pat", "days_fx"]

    for n in W13_BINS:
        out[f"C{n}"] = _qbins(c["cp"], edges[f"cp_{n}"]).astype(str)
        out[f"T{n}"] = _qbins(c["dp"], edges[f"dp_{n}"]).astype(str)
        out[f"G{n}"] = _qbins(c["diag"], edges[f"dg_{n}"]).astype(str)
        out[f"A{n}"] = _qbins(c["anti"], edges[f"an_{n}"]).astype(str)
        out[f"U{n}"] = _qbins(c["ratio"], edges[f"rt_{n}"]).astype(str)
        out[f"S{n}"] = _qbins(c["cond_r"], edges[f"cr_{n}"]).astype(str)
        out[f"E{n}"] = _qbins(c["rate"], edges[f"er_{n}"]).astype(str)
        cats += [f"C{n}", f"T{n}", f"G{n}", f"A{n}", f"U{n}", f"S{n}", f"E{n}"]
    out["days_q5"] = _qbins(c["days"], edges["dq_5x"]).astype(str)
    cats.append("days_q5")

    x = _cross
    x(out, cats, "V_cell8", "C8", "T8")
    x(out, cats, "V_cell15", "C15", "T15")
    x(out, cats, "V_cell25", "C25", "T25")
    x(out, cats, "V_cell8_src", "C8", "T8", "source")
    x(out, cats, "V_cell8_reg", "C8", "T8", "region")
    x(out, cats, "V_C8_src", "C8", "source")
    x(out, cats, "V_C15_src", "C15", "source")
    x(out, cats, "V_C25_src", "C25", "source")
    x(out, cats, "V_T8_src", "T8", "source")
    x(out, cats, "V_T15_reg", "T15", "region")
    x(out, cats, "V_G8_src", "G8", "source")
    x(out, cats, "V_G15_src", "G15", "source")
    x(out, cats, "V_G8_reg", "G8", "region")
    x(out, cats, "V_G8_age", "G8", "age_cat")
    x(out, cats, "V_G8_pat", "G8", "bin_pat")
    x(out, cats, "V_A8_reg", "A8", "region")
    x(out, cats, "V_A8_src", "A8", "source")
    x(out, cats, "V_dfx_src", "days_fx", "source")
    x(out, cats, "V_dfx_C8", "days_fx", "C8")
    x(out, cats, "V_dfx_S8", "days_fx", "S8")
    x(out, cats, "V_dfx_U8", "days_fx", "U8")
    x(out, cats, "V_dfx_E8", "days_fx", "E8")
    x(out, cats, "V_U8_src", "U8", "source")
    x(out, cats, "V_U15_src", "U15", "source")
    x(out, cats, "V_U8_reg", "U8", "region")
    x(out, cats, "V_U8_age", "U8", "age_cat")
    x(out, cats, "V_S8_src", "S8", "source")
    x(out, cats, "V_S15_src", "S15", "source")
    x(out, cats, "V_S25_src", "S25", "source")
    x(out, cats, "V_E8_src", "E8", "source")
    x(out, cats, "V_E15_src", "E15", "source")
    x(out, cats, "V_E8_reg", "E8", "region")
    x(out, cats, "V_src_S8_age", "source", "S8", "age_cat")
    x(out, cats, "V_reg_src", "region", "source")
    x(out, cats, "V_reg_age", "region", "age_cat")
    x(out, cats, "V_src_age", "source", "age_cat")
    x(out, cats, "V_reg_src_age", "region", "source", "age_cat")
    x(out, cats, "V_G8_reg_src", "G8", "region", "source")
    x(out, cats, "V_U8_reg_src", "U8", "region", "source")
    x(out, cats, "V_reg_pat", "region", "bin_pat")
    for col in ("region", "source", "bin_pat", "V_reg_src", "V_cell8", "V_G8_src", "V_U8_src", "V_S8_src"):
        out[f"freq_{col}"] = out[col].map(out[col].value_counts()).astype(float)
    return out, cats


def w13_frame(raw: pd.DataFrame, edges: dict, stream_offset: int, n_views: int = 4):
    X, cats = build_w13(raw, edges)
    add_noise_view(X, cats, raw)
    c = _w13_core(raw)
    add_jitter_views(X, cats, raw, c["diag"], pd.to_numeric(raw["days"]),
                     n_views=n_views, n_bins=11, stream_offset=180 + stream_offset)
    for col in cats:
        X[col] = X[col].astype(str)
    num = [col for col in X.columns if col not in cats]
    X[num] = X[num].astype(float).fillna(-999.0)
    return X, cats

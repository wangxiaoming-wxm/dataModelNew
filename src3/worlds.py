"""Two further encoding worlds (w4, w5).

What actually moves the score on this dataset is averaging over *encodings* of
the same interactions, not over model families -- see docs/EXPERIMENTS.md.
Three worlds already exist in src2/features.py:

  main  condition / median(condition | source),  quantile bins (5,10,20,40)
  alt   rank(condition) within source,           quantile bins (7,13,25)
  alt2  condition z-scored within source x region, bins (4,9,16)

The handoff's requirement for a new world is that it decorrelates *without*
losing strength: `alt2` decorrelates well (rank corr 0.950) but sits 0.004
below the others, and `max` fusion is sensitive to a weak arm.  So both worlds
below keep the two signals that carry the data -- condition normalised inside
the vehicle model, and the days/condition rate -- and only change how they are
expressed.

  w4  Gaussianised condition within source (rank -> probit) and a robust
      median/MAD z-score; the rate is taken in log space and binned on equal
      width there rather than on quantiles.  Bin counts (6,11,22).
  w5  Joint (days percentile, condition percentile) cells computed *inside*
      each vehicle model, so the strongest interaction is handed to the model
      as a single coordinate instead of being recovered from a cross.
      Bin counts (8,15) plus fixed physical day edges.

Everything here is label-free, so fitting on train+test is transductive but
leakage-free -- the same guarantee the rest of the branch relies on, and
src3/audit.py re-tests it by permutation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtri

from features import BIN_COLS, GRADE_MAP, DAYS_FIXED_EDGES, add_noise_view
from jitter import add_jitter_views


def _qbins(values, edges) -> np.ndarray:
    return np.digitize(np.asarray(values, dtype=float), edges)


def _cross(out: pd.DataFrame, cats: list[str], name: str, *parts: str) -> None:
    s = out[parts[0]].astype(str)
    for p in parts[1:]:
        s = s + "|" + out[p].astype(str)
    out[name] = s
    cats.append(name)


# ---------------------------------------------------------------------------
# world 4: Gaussianised / robust normalisation, log-space rate
# ---------------------------------------------------------------------------
W4_BINS = (6, 11, 22)


def _w4_core(df: pd.DataFrame) -> pd.DataFrame:
    cond = pd.to_numeric(df["condition"])
    days = pd.to_numeric(df["days"])
    g = df.groupby("source")["condition"]

    # rank inside the vehicle model, pushed through a probit so the tails get
    # the resolution that a flat percentile scale throws away
    n = g.transform("count")
    rk = g.rank(method="average")
    probit = pd.Series(ndtri(((rk - 0.5) / n).clip(1e-6, 1 - 1e-6).fillna(0.5)),
                       index=df.index).fillna(0.0)

    med = g.transform("median")
    mad = (cond - med).abs().groupby(df["source"]).transform("median") * 1.4826
    rz = ((cond - med) / mad.replace(0, np.nan)).fillna(0.0).clip(-8, 8)

    log_days = np.log1p(days.clip(lower=0))
    # rate in log space: log(days) - log(condition ratio); additive, so equal
    # width bins here are geometric bins on the original ratio
    cond_r = (cond / med.replace(0, np.nan)).fillna(1.0)
    log_rate = log_days - np.log(cond_r.clip(lower=1e-6))
    return pd.DataFrame({"probit": probit, "rz": rz, "log_days": log_days,
                         "log_rate": log_rate, "cond_r": cond_r, "days": days,
                         "cond": cond}, index=df.index)


def fit_edges_w4(df: pd.DataFrame) -> dict:
    c = _w4_core(df)
    edges: dict = {}
    for n in W4_BINS:
        edges[f"pb_{n}"] = np.quantile(c["probit"], np.linspace(0, 1, n + 1)[1:-1])
        edges[f"rz_{n}"] = np.quantile(c["rz"], np.linspace(0, 1, n + 1)[1:-1])
        edges[f"ld_{n}"] = np.quantile(c["log_days"], np.linspace(0, 1, n + 1)[1:-1])
        edges[f"q_{n}"] = np.quantile(c["cond"].dropna(), np.linspace(0, 1, n + 1)[1:-1])
        # equal-width in log space, not quantile: a genuinely different cut set
        lo, hi = np.percentile(c["log_rate"], [0.5, 99.5])
        edges[f"lr_{n}"] = np.linspace(lo, hi, n + 1)[1:-1]
    edges["ld_5x"] = np.quantile(c["log_days"], np.linspace(0, 1, 6)[1:-1])
    return edges


def build_w4(df: pd.DataFrame, edges: dict) -> tuple[pd.DataFrame, list[str]]:
    c = _w4_core(df)
    out = pd.DataFrame(index=df.index)
    out["probit"] = c["probit"]
    out["rz"] = c["rz"]
    out["log_days"] = c["log_days"]
    out["log_rate"] = c["log_rate"]
    out["days"] = c["days"]
    out["condition"] = c["cond"]
    out["condition_missing"] = c["cond"].isna().astype(int)
    out["rate_x_age"] = c["log_rate"] * df["age_range"].astype(float)
    out["probit_x_days"] = c["probit"] * c["log_days"]
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
    cats += ["region", "source", "month", "version", "grades_c", "age_cat", "bin_pat"]

    for n in W4_BINS:
        out[f"P{n}"] = _qbins(c["probit"], edges[f"pb_{n}"]).astype(str)
        out[f"Z{n}"] = _qbins(c["rz"], edges[f"rz_{n}"]).astype(str)
        out[f"D{n}"] = _qbins(c["log_days"], edges[f"ld_{n}"]).astype(str)
        out[f"R{n}"] = _qbins(c["log_rate"], edges[f"lr_{n}"]).astype(str)
        out[f"Q{n}"] = _qbins(c["cond"].fillna(-1), edges[f"q_{n}"]).astype(str)
        cats += [f"P{n}", f"Z{n}", f"D{n}", f"R{n}", f"Q{n}"]
    # add_noise_view crosses t3 against a 5-bin day column; keep that hook alive
    out["days_q5"] = _qbins(c["log_days"], edges["ld_5x"]).astype(str)
    # Fixed physical day edges rather than quantiles.  `dfx_c10` in the main
    # world is the single strongest column measured anywhere in this dataset
    # (honest OOF-TE AUC 0.628) and w4 had no member of that family at all.
    out["days_fx"] = np.digitize(c["days"].to_numpy(dtype=float), DAYS_FIXED_EDGES).astype(str)
    cats += ["days_q5", "days_fx"]

    x = _cross
    x(out, cats, "W_P6_src", "P6", "source")
    x(out, cats, "W_P11_src", "P11", "source")
    x(out, cats, "W_P22_src", "P22", "source")
    x(out, cats, "W_Z11_src", "Z11", "source")
    x(out, cats, "W_P11_reg", "P11", "region")
    x(out, cats, "W_P6_age", "P6", "age_cat")
    x(out, cats, "W_D11_reg", "D11", "region")
    x(out, cats, "W_D11_src", "D11", "source")
    x(out, cats, "W_D22_reg", "D22", "region")
    x(out, cats, "W_D6_age", "D6", "age_cat")
    x(out, cats, "W_R11_reg", "R11", "region")
    x(out, cats, "W_R11_src", "R11", "source")
    x(out, cats, "W_R6_age", "R6", "age_cat")
    x(out, cats, "W_R6_pat", "R6", "bin_pat")
    x(out, cats, "W_D6_P6", "D6", "P6")
    x(out, cats, "W_D11_P11", "D11", "P11")
    x(out, cats, "W_reg_src", "region", "source")
    x(out, cats, "W_reg_age", "region", "age_cat")
    x(out, cats, "W_src_age", "source", "age_cat")
    x(out, cats, "W_D6_reg_src", "D6", "region", "source")
    x(out, cats, "W_R6_reg_src", "R6", "region", "source")
    x(out, cats, "W_P6_reg_age", "P6", "region", "age_cat")
    x(out, cats, "W_D6_pat", "D6", "bin_pat")
    x(out, cats, "W_reg_pat", "region", "bin_pat")
    x(out, cats, "W_dfx_Q11", "days_fx", "Q11")
    x(out, cats, "W_dfx_P11", "days_fx", "P11")
    x(out, cats, "W_dfx_R11", "days_fx", "R11")
    x(out, cats, "W_dfx_src", "days_fx", "source")
    x(out, cats, "W_dfx_reg", "days_fx", "region")
    x(out, cats, "W_Q11_src", "Q11", "source")
    x(out, cats, "W_Q6_reg", "Q6", "region")
    for col in ("region", "source", "bin_pat", "W_reg_src", "W_P11_src", "W_D11_reg"):
        out[f"freq_{col}"] = out[col].map(out[col].value_counts()).astype(float)
    return out, cats


# ---------------------------------------------------------------------------
# world 5: joint within-model percentile cells
# ---------------------------------------------------------------------------
W5_BINS = (8, 15)


def _w5_core(df: pd.DataFrame) -> pd.DataFrame:
    cond = pd.to_numeric(df["condition"])
    days = pd.to_numeric(df["days"])
    # both drivers as percentiles *inside* the vehicle model, so the strongest
    # interaction becomes a plain 2-D coordinate rather than a recovered cross
    cp = df.groupby("source")["condition"].rank(pct=True).fillna(0.5)
    dp = df.groupby("source")["days"].rank(pct=True).fillna(0.5)
    # diagonal / anti-diagonal of that square carry the rate and the exposure
    diag = dp - cp
    anti = (dp + cp) / 2.0
    return pd.DataFrame({"cp": cp, "dp": dp, "diag": diag, "anti": anti,
                         "days": days, "cond": cond}, index=df.index)


def fit_edges_w5(df: pd.DataFrame) -> dict:
    c = _w5_core(df)
    edges: dict = {}
    for n in W5_BINS:
        qs = np.linspace(0, 1, n + 1)[1:-1]
        edges[f"cp_{n}"] = np.quantile(c["cp"], qs)
        edges[f"dp_{n}"] = np.quantile(c["dp"], qs)
        edges[f"dg_{n}"] = np.quantile(c["diag"], qs)
        edges[f"an_{n}"] = np.quantile(c["anti"], qs)
    edges["dq_5x"] = np.quantile(c["days"], np.linspace(0, 1, 6)[1:-1])
    return edges


def build_w5(df: pd.DataFrame, edges: dict) -> tuple[pd.DataFrame, list[str]]:
    c = _w5_core(df)
    out = pd.DataFrame(index=df.index)
    out["cp"] = c["cp"]
    out["dp"] = c["dp"]
    out["diag"] = c["diag"]
    out["anti"] = c["anti"]
    out["days"] = c["days"]
    out["condition"] = c["cond"]
    out["condition_missing"] = c["cond"].isna().astype(int)
    out["diag_x_age"] = c["diag"] * df["age_range"].astype(float)
    out["diag_x_bin"] = c["diag"] * df[BIN_COLS].sum(axis=1)
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

    for n in W5_BINS:
        out[f"C{n}"] = _qbins(c["cp"], edges[f"cp_{n}"]).astype(str)
        out[f"T{n}"] = _qbins(c["dp"], edges[f"dp_{n}"]).astype(str)
        out[f"G{n}"] = _qbins(c["diag"], edges[f"dg_{n}"]).astype(str)
        out[f"A{n}"] = _qbins(c["anti"], edges[f"an_{n}"]).astype(str)
        cats += [f"C{n}", f"T{n}", f"G{n}", f"A{n}"]
    # add_noise_view crosses t3 against a 5-bin day column; keep that hook alive
    out["days_q5"] = _qbins(c["days"], edges["dq_5x"]).astype(str)
    cats.append("days_q5")

    x = _cross
    # the joint cell itself, at two resolutions, and inside each segment
    x(out, cats, "V_cell8", "C8", "T8")
    x(out, cats, "V_cell15", "C15", "T15")
    x(out, cats, "V_cell8_src", "C8", "T8", "source")
    x(out, cats, "V_cell8_reg", "C8", "T8", "region")
    x(out, cats, "V_C8_src", "C8", "source")
    x(out, cats, "V_C15_src", "C15", "source")
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
    x(out, cats, "V_reg_src", "region", "source")
    x(out, cats, "V_reg_age", "region", "age_cat")
    x(out, cats, "V_src_age", "source", "age_cat")
    x(out, cats, "V_reg_src_age", "region", "source", "age_cat")
    x(out, cats, "V_G8_reg_src", "G8", "region", "source")
    x(out, cats, "V_reg_pat", "region", "bin_pat")
    for col in ("region", "source", "bin_pat", "V_reg_src", "V_cell8", "V_G8_src"):
        out[f"freq_{col}"] = out[col].map(out[col].value_counts()).astype(float)
    return out, cats


# ---------------------------------------------------------------------------
# frame builders (mirror src2/arms.py so the runner can treat them alike)
# ---------------------------------------------------------------------------
def _finish(X: pd.DataFrame, cats: list[str]) -> tuple[pd.DataFrame, list[str]]:
    for c in cats:
        X[c] = X[c].astype(str)
    num = [c for c in X.columns if c not in cats]
    X[num] = X[num].astype(float).fillna(-999.0)
    return X, cats


def w4_frame(raw: pd.DataFrame, edges: dict, stream_offset: int, n_views: int = 3):
    X, cats = build_w4(raw, edges)
    add_noise_view(X, cats, raw)
    c = _w4_core(raw)
    add_jitter_views(X, cats, raw, c["probit"], pd.to_numeric(raw["days"]),
                     n_views=n_views, n_bins=11, stream_offset=150 + stream_offset)
    return _finish(X, cats)


def w5_frame(raw: pd.DataFrame, edges: dict, stream_offset: int, n_views: int = 3):
    X, cats = build_w5(raw, edges)
    add_noise_view(X, cats, raw)
    c = _w5_core(raw)
    add_jitter_views(X, cats, raw, c["diag"], pd.to_numeric(raw["days"]),
                     n_views=n_views, n_bins=9, stream_offset=200 + stream_offset)
    return _finish(X, cats)

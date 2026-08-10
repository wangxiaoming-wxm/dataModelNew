"""V4 encoding worlds w6 / w7.

V3 mid-score residual probe: after conditioning on views_max, days/ratio still
rank labels at ~0.54 AUC inside the central score band.  That is unused signal
relative to the current encoding, not a licence to peek at labels.

  w6  "days-primary": keep source-normalised condition, but bin and cross
      predominantly on days / log-days / ratio with a *different* cut family
      (equal-width on rank-gaussianised days; fixed physical day edges mixed
      with fine quantiles).  Condition enters mainly through ratio and a small
      set of cond crosses so the world cannot collapse into main/alt.

  w7  "cell-grid": within each (region, source) build a 2-D percentile cell of
      (days, cond_r), plus a global percentile twin.  Bin counts deliberately
      avoid {5,7,8,10,13,15,20,40} used by main/alt/w4/w5.

All transforms are label-free; edges fitted on train+test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtri

from features import BIN_COLS, GRADE_MAP, DAYS_FIXED_EDGES, add_noise_view
from jitter import add_jitter_views

W6_BINS = (9, 14, 28)
W7_BINS = (12, 18)


def _qbins(values, edges) -> np.ndarray:
    return np.digitize(np.asarray(values, dtype=float), edges)


def _cross(out: pd.DataFrame, cats: list[str], name: str, *parts: str) -> None:
    s = out[parts[0]].astype(str)
    for p in parts[1:]:
        s = s + "|" + out[p].astype(str)
    out[name] = s
    cats.append(name)


def _core(df: pd.DataFrame) -> pd.DataFrame:
    cond = pd.to_numeric(df["condition"])
    days = pd.to_numeric(df["days"])
    med = df.groupby("source")["condition"].transform("median")
    cond_r = (cond / med.replace(0, np.nan)).fillna(1.0)
    ratio = days / cond_r.clip(lower=1e-9)
    # rank-gaussianised days (global) — different resolution from quantile bins
    n = len(df)
    rk = days.rank(method="average")
    gdays = pd.Series(ndtri(((rk - 0.5) / n).clip(1e-6, 1 - 1e-6)), index=df.index)
    return pd.DataFrame(
        {
            "days": days,
            "log_days": np.log1p(days.clip(lower=0)),
            "cond": cond,
            "cond_r": cond_r,
            "ratio": ratio,
            "log_ratio": np.log(ratio.clip(lower=1e-9)),
            "gdays": gdays,
        },
        index=df.index,
    )


# ---------------------------------------------------------------------------
# w6
# ---------------------------------------------------------------------------
def fit_edges_w6(df: pd.DataFrame) -> dict:
    c = _core(df)
    edges: dict = {"__scale__": df.groupby("source")["condition"].median()}
    for n in W6_BINS:
        qs = np.linspace(0, 1, n + 1)[1:-1]
        edges[f"d_{n}"] = np.quantile(c["days"], qs)
        edges[f"ld_{n}"] = np.quantile(c["log_days"], qs)
        edges[f"r_{n}"] = np.quantile(c["ratio"], qs)
        edges[f"gd_{n}"] = np.quantile(c["gdays"], qs)
        # equal-width on gdays (different from quantile)
        lo, hi = np.percentile(c["gdays"], [0.5, 99.5])
        edges[f"gdew_{n}"] = np.linspace(lo, hi, n + 1)[1:-1]
    for n in (10, 20):
        edges[f"cr_{n}"] = np.quantile(c["cond_r"], np.linspace(0, 1, n + 1)[1:-1])
    return edges


def build_w6(df: pd.DataFrame, edges: dict) -> tuple[pd.DataFrame, list[str]]:
    c = _core(df)
    out = pd.DataFrame(index=df.index)
    out["days"] = c["days"]
    out["log_days"] = c["log_days"]
    out["gdays"] = c["gdays"]
    out["cond_r"] = c["cond_r"]
    out["ratio"] = c["ratio"]
    out["log_ratio"] = c["log_ratio"]
    out["ratio_p75"] = c["days"] / c["cond_r"].clip(lower=1e-9) ** 0.75
    out["days_x_age"] = c["days"] * df["age_range"].astype(float)
    out["ratio_x_age"] = c["ratio"] * df["age_range"].astype(float)
    out["age_range"] = df["age_range"].astype(float)
    out["grade_ord"] = df["grades"].map(GRADE_MAP).astype(float)
    out["condition"] = c["cond"]
    out["condition_missing"] = c["cond"].isna().astype(int)
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
    cats += ["region", "source", "month", "version", "grades_c", "age_cat", "bin_pat", "days_fx"]

    for n in W6_BINS:
        out[f"d_q{n}"] = _qbins(c["days"], edges[f"d_{n}"]).astype(str)
        out[f"ld_q{n}"] = _qbins(c["log_days"], edges[f"ld_{n}"]).astype(str)
        out[f"r_q{n}"] = _qbins(c["ratio"], edges[f"r_{n}"]).astype(str)
        out[f"gd_q{n}"] = _qbins(c["gdays"], edges[f"gd_{n}"]).astype(str)
        out[f"gd_ew{n}"] = _qbins(c["gdays"], edges[f"gdew_{n}"]).astype(str)
        cats += [f"d_q{n}", f"ld_q{n}", f"r_q{n}", f"gd_q{n}", f"gd_ew{n}"]
    for n in (10, 20):
        out[f"cr_q{n}"] = _qbins(c["cond_r"], edges[f"cr_{n}"]).astype(str)
        cats.append(f"cr_q{n}")

    _cross(out, cats, "d9_reg", "d_q9", "region")
    _cross(out, cats, "d9_src", "d_q9", "source")
    _cross(out, cats, "d14_reg", "d_q14", "region")
    _cross(out, cats, "d14_src", "d_q14", "source")
    _cross(out, cats, "r9_reg", "r_q9", "region")
    _cross(out, cats, "r9_src", "r_q9", "source")
    _cross(out, cats, "r14_reg", "r_q14", "region")
    _cross(out, cats, "r14_age", "r_q14", "age_cat")
    _cross(out, cats, "gd14_reg", "gd_q14", "region")
    _cross(out, cats, "gd14_src", "gd_q14", "source")
    _cross(out, cats, "d9_pat", "d_q9", "bin_pat")
    _cross(out, cats, "r9_pat", "r_q9", "bin_pat")
    _cross(out, cats, "d9_cr10", "d_q9", "cr_q10")
    _cross(out, cats, "reg_src", "region", "source")
    _cross(out, cats, "d9_reg_src", "d_q9", "region", "source")
    return out, cats


def w6_frame(raw: pd.DataFrame, edges: dict, stream_offset: int, n_views: int = 4):
    X, cats = build_w6(raw, edges)
    add_noise_view(X, cats, raw)
    der_cr = _core(raw)["cond_r"]
    add_jitter_views(
        X, cats, raw, der_cr, pd.to_numeric(raw["days"]),
        n_views=n_views, n_bins=11, stream_offset=200 + stream_offset,
    )
    for c in cats:
        X[c] = X[c].astype(str)
    num = [c for c in X.columns if c not in cats]
    X[num] = X[num].astype(float).fillna(-999.0)
    return X, cats


# ---------------------------------------------------------------------------
# w7
# ---------------------------------------------------------------------------
def fit_edges_w7(df: pd.DataFrame) -> dict:
    c = _core(df)
    edges: dict = {"__scale__": df.groupby("source")["condition"].median()}
    for n in W7_BINS:
        qs = np.linspace(0, 1, n + 1)[1:-1]
        edges[f"d_{n}"] = np.quantile(c["days"], qs)
        edges[f"cr_{n}"] = np.quantile(c["cond_r"], qs)
        edges[f"r_{n}"] = np.quantile(c["ratio"], qs)
    # within-source percentile edges are implicit (rank); keep global twins only
    return edges


def build_w7(df: pd.DataFrame, edges: dict) -> tuple[pd.DataFrame, list[str]]:
    c = _core(df)
    out = pd.DataFrame(index=df.index)
    out["days"] = c["days"]
    out["log_days"] = c["log_days"]
    out["cond_r"] = c["cond_r"]
    out["ratio"] = c["ratio"]
    out["log_ratio"] = c["log_ratio"]
    out["age_range"] = df["age_range"].astype(float)
    out["grade_ord"] = df["grades"].map(GRADE_MAP).astype(float)
    out["condition"] = c["cond"]
    out["condition_missing"] = c["cond"].isna().astype(int)
    for b in BIN_COLS:
        out[b] = df[b].astype(int)
    out["bin_sum"] = df[BIN_COLS].sum(axis=1)

    # within-source percentile coordinates (label-free)
    out["days_psrc"] = df.groupby("source")["days"].rank(pct=True).fillna(0.5)
    out["cond_psrc"] = df.groupby("source")["condition"].rank(pct=True).fillna(0.5)
    # within region
    out["days_preg"] = df.groupby("region")["days"].rank(pct=True).fillna(0.5)
    out["cond_preg"] = df.groupby("region")["condition"].rank(pct=True).fillna(0.5)

    cats: list[str] = []
    out["region"] = df["region"].astype(str)
    out["source"] = df["source"].astype(str)
    out["month"] = df["month"].astype(str)
    out["version"] = df["version"].astype(str)
    out["grades_c"] = df["grades"].astype(str)
    out["age_cat"] = df["age_range"].astype(str)
    out["bin_pat"] = df[BIN_COLS].astype(str).agg("".join, axis=1)
    cats += ["region", "source", "month", "version", "grades_c", "age_cat", "bin_pat"]

    for n in W7_BINS:
        out[f"d_q{n}"] = _qbins(c["days"], edges[f"d_{n}"]).astype(str)
        out[f"cr_q{n}"] = _qbins(c["cond_r"], edges[f"cr_{n}"]).astype(str)
        out[f"r_q{n}"] = _qbins(c["ratio"], edges[f"r_{n}"]).astype(str)
        # discretise within-group percentiles into n cells
        out[f"dps_q{n}"] = (np.floor(out["days_psrc"] * n).clip(0, n - 1)).astype(int).astype(str)
        out[f"cps_q{n}"] = (np.floor(out["cond_psrc"] * n).clip(0, n - 1)).astype(int).astype(str)
        out[f"dpr_q{n}"] = (np.floor(out["days_preg"] * n).clip(0, n - 1)).astype(int).astype(str)
        cats += [f"d_q{n}", f"cr_q{n}", f"r_q{n}", f"dps_q{n}", f"cps_q{n}", f"dpr_q{n}"]
        # 2-D cells
        _cross(out, cats, f"cell_src_{n}", f"dps_q{n}", f"cps_q{n}")
        _cross(out, cats, f"cell_src_reg_{n}", f"dps_q{n}", f"cps_q{n}", "region")

    _cross(out, cats, "d12_reg", "d_q12", "region")
    _cross(out, cats, "d12_src", "d_q12", "source")
    _cross(out, cats, "r12_reg", "r_q12", "region")
    _cross(out, cats, "r12_src", "r_q12", "source")
    _cross(out, cats, "reg_src", "region", "source")
    _cross(out, cats, "d12_pat", "d_q12", "bin_pat")
    _cross(out, cats, "cell12_src", "cell_src_12", "source")
    return out, cats


def w7_frame(raw: pd.DataFrame, edges: dict, stream_offset: int, n_views: int = 4):
    X, cats = build_w7(raw, edges)
    add_noise_view(X, cats, raw)
    der_cr = _core(raw)["cond_r"]
    add_jitter_views(
        X, cats, raw, der_cr, pd.to_numeric(raw["days"]),
        n_views=n_views, n_bins=9, stream_offset=300 + stream_offset,
    )
    for c in cats:
        X[c] = X[c].astype(str)
    num = [c for c in X.columns if c not in cats]
    X[num] = X[num].astype(float).fillna(-999.0)
    return X, cats


# ---------------------------------------------------------------------------
# w8 — age-coupled source scale (HANDOFF 5.1)
# ---------------------------------------------------------------------------
# Keep cond_r / ratio as the carriers, but normalise condition inside
# (source, age_range) cells (fallback to source median when the cell is tiny).
# Bin family (8, 16, 32) deliberately avoids main/alt/w4/w5/w6/w7 counts.
# Cross list mirrors the high-value main interactions so strength stays close
# to the three strong arms while the cut geometry decorrelates.

W8_BINS = (8, 16, 32)
W8_MIN_CELL = 40


def _w8_core(df: pd.DataFrame) -> pd.DataFrame:
    cond = pd.to_numeric(df["condition"])
    days = pd.to_numeric(df["days"])
    src = df["source"]
    age = df["age_range"]
    cell = src.astype(str) + "|" + age.astype(str)
    cell_med = cond.groupby(cell).transform("median")
    cell_n = cond.groupby(cell).transform("count")
    src_med = cond.groupby(src).transform("median")
    # fallback when the age cell is too small to be stable
    scale = cell_med.where(cell_n >= W8_MIN_CELL, src_med)
    cond_r = (cond / scale.replace(0, np.nan)).fillna(1.0)
    # classic source ratio kept as a twin view (strength + diversity)
    cond_r_src = (cond / src_med.replace(0, np.nan)).fillna(1.0)
    mad = (cond - src_med).abs().groupby(src).transform("median") * 1.4826
    rz = ((cond - src_med) / mad.replace(0, np.nan)).fillna(0.0).clip(-8, 8)
    ratio = days / cond_r.clip(lower=1e-9)
    ratio_src = days / cond_r_src.clip(lower=1e-9)
    log_ratio = np.log(ratio.clip(lower=1e-9))
    return pd.DataFrame(
        {
            "days": days,
            "log_days": np.log1p(days.clip(lower=0)),
            "cond": cond,
            "cond_r": cond_r,
            "cond_r_src": cond_r_src,
            "rz": rz,
            "ratio": ratio,
            "ratio_src": ratio_src,
            "log_ratio": log_ratio,
        },
        index=df.index,
    )


def fit_edges_w8(df: pd.DataFrame) -> dict:
    c = _w8_core(df)
    edges: dict = {}
    for n in W8_BINS:
        qs = np.linspace(0, 1, n + 1)[1:-1]
        edges[f"d_{n}"] = np.quantile(c["days"], qs)
        edges[f"r_{n}"] = np.quantile(c["ratio"], qs)
        edges[f"rs_{n}"] = np.quantile(c["ratio_src"], qs)
        edges[f"cr_{n}"] = np.quantile(c["cond_r"], qs)
        edges[f"crs_{n}"] = np.quantile(c["cond_r_src"], qs)
        edges[f"rz_{n}"] = np.quantile(c["rz"], qs)
        # equal-width on log-ratio (geometric cuts on the rate)
        lo, hi = np.percentile(c["log_ratio"], [0.5, 99.5])
        edges[f"lr_{n}"] = np.linspace(lo, hi, n + 1)[1:-1]
    return edges


def build_w8(df: pd.DataFrame, edges: dict) -> tuple[pd.DataFrame, list[str]]:
    c = _w8_core(df)
    out = pd.DataFrame(index=df.index)
    out["days"] = c["days"]
    out["log_days"] = c["log_days"]
    out["condition"] = c["cond"]
    out["condition_missing"] = c["cond"].isna().astype(int)
    out["cond_r"] = c["cond_r"]
    out["cond_r_src"] = c["cond_r_src"]
    out["rz"] = c["rz"]
    out["ratio"] = c["ratio"]
    out["ratio_src"] = c["ratio_src"]
    out["log_ratio"] = c["log_ratio"]
    out["ratio_p75"] = c["days"] / c["cond_r"].clip(lower=1e-9) ** 0.75
    out["age_range"] = df["age_range"].astype(float)
    out["days_over_age"] = c["days"] / df["age_range"].astype(float).clip(lower=1e-9)
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
    cats += ["region", "source", "month", "version", "grades_c", "age_cat", "bin_pat", "days_fx"]

    for n in W8_BINS:
        out[f"d_q{n}"] = _qbins(c["days"], edges[f"d_{n}"]).astype(str)
        out[f"r_q{n}"] = _qbins(c["ratio"], edges[f"r_{n}"]).astype(str)
        out[f"rs_q{n}"] = _qbins(c["ratio_src"], edges[f"rs_{n}"]).astype(str)
        out[f"cr_q{n}"] = _qbins(c["cond_r"], edges[f"cr_{n}"]).astype(str)
        out[f"crs_q{n}"] = _qbins(c["cond_r_src"], edges[f"crs_{n}"]).astype(str)
        out[f"rz_q{n}"] = _qbins(c["rz"], edges[f"rz_{n}"]).astype(str)
        out[f"lr_ew{n}"] = _qbins(c["log_ratio"], edges[f"lr_{n}"]).astype(str)
        cats += [f"d_q{n}", f"r_q{n}", f"rs_q{n}", f"cr_q{n}", f"crs_q{n}", f"rz_q{n}", f"lr_ew{n}"]

    # high-value crosses (main-like, but on the new bins)
    _cross(out, cats, "reg_src", "region", "source")
    _cross(out, cats, "src_age", "source", "age_cat")
    _cross(out, cats, "reg_age", "region", "age_cat")
    _cross(out, cats, "d8_reg", "d_q8", "region")
    _cross(out, cats, "d8_src", "d_q8", "source")
    _cross(out, cats, "d16_reg", "d_q16", "region")
    _cross(out, cats, "d16_src", "d_q16", "source")
    _cross(out, cats, "d8_age", "d_q8", "age_cat")
    _cross(out, cats, "r8_reg", "r_q8", "region")
    _cross(out, cats, "r8_src", "r_q8", "source")
    _cross(out, cats, "r8_age", "r_q8", "age_cat")
    _cross(out, cats, "r16_reg", "r_q16", "region")
    _cross(out, cats, "r16_src", "r_q16", "source")
    _cross(out, cats, "cr8_src", "cr_q8", "source")
    _cross(out, cats, "cr16_src", "cr_q16", "source")
    _cross(out, cats, "crs8_src", "crs_q8", "source")
    _cross(out, cats, "crs16_src", "crs_q16", "source")
    _cross(out, cats, "cr8_age", "cr_q8", "age_cat")
    _cross(out, cats, "rz8_src", "rz_q8", "source")
    _cross(out, cats, "d8_cr8", "d_q8", "cr_q8")
    _cross(out, cats, "d16_cr16", "d_q16", "cr_q16")
    _cross(out, cats, "d8_crs8", "d_q8", "crs_q8")
    _cross(out, cats, "lr8_reg", "lr_ew8", "region")
    _cross(out, cats, "lr8_src", "lr_ew8", "source")
    _cross(out, cats, "d8_pat", "d_q8", "bin_pat")
    _cross(out, cats, "r8_pat", "r_q8", "bin_pat")
    _cross(out, cats, "d8_reg_src", "d_q8", "region", "source")
    _cross(out, cats, "r8_reg_src", "r_q8", "region", "source")
    _cross(out, cats, "src_cr8_age", "source", "cr_q8", "age_cat")
    _cross(out, cats, "dfx_src", "days_fx", "source")
    _cross(out, cats, "dfx_cr8", "days_fx", "cr_q8")
    for col in ("region", "source", "bin_pat", "reg_src", "d8_reg", "cr8_src"):
        out[f"freq_{col}"] = out[col].map(out[col].value_counts()).astype(float)
    return out, cats


def w8_frame(raw: pd.DataFrame, edges: dict, stream_offset: int, n_views: int = 4):
    X, cats = build_w8(raw, edges)
    add_noise_view(X, cats, raw)
    der_cr = _w8_core(raw)["cond_r"]
    add_jitter_views(
        X, cats, raw, der_cr, pd.to_numeric(raw["days"]),
        n_views=n_views, n_bins=16, stream_offset=400 + stream_offset,
    )
    for c in cats:
        X[c] = X[c].astype(str)
    num = [c for c in X.columns if c not in cats]
    X[num] = X[num].astype(float).fillna(-999.0)
    return X, cats


# ---------------------------------------------------------------------------
# w9 — dual-rate world (median ratio + rank-rate in one frame)
# ---------------------------------------------------------------------------
# alt's rank-rate and main's median-ratio are the two encodings that already
# work.  Packing both into one world with a fresh bin family (9, 18, 27) and a
# mixed cross list should stay near-strong while sitting between main and alt
# in rank space — useful for max fusion if bagged AUC clears the threshold.

W9_BINS = (9, 18, 27)


def _w9_core(df: pd.DataFrame) -> pd.DataFrame:
    cond = pd.to_numeric(df["condition"])
    days = pd.to_numeric(df["days"])
    src_med = df.groupby("source")["condition"].transform("median")
    cond_r = (cond / src_med.replace(0, np.nan)).fillna(1.0)
    ratio = days / cond_r.clip(lower=1e-9)
    rk = df.groupby("source")["condition"].rank(pct=True).fillna(0.5)
    rate = days * (1.0 - rk)
    # source-internal days percentile (unused by main/alt as a primary axis)
    days_ps = df.groupby("source")["days"].rank(pct=True).fillna(0.5)
    # soft blend of the two rates in log space
    blend = 0.5 * np.log(ratio.clip(lower=1e-9)) + 0.5 * np.log1p(rate.clip(lower=0))
    return pd.DataFrame(
        {
            "days": days,
            "log_days": np.log1p(days.clip(lower=0)),
            "cond": cond,
            "cond_r": cond_r,
            "ratio": ratio,
            "log_ratio": np.log(ratio.clip(lower=1e-9)),
            "rk": rk,
            "rate": rate,
            "log_rate": np.log1p(rate.clip(lower=0)),
            "days_ps": days_ps,
            "blend": blend,
        },
        index=df.index,
    )


def fit_edges_w9(df: pd.DataFrame) -> dict:
    c = _w9_core(df)
    edges: dict = {}
    for n in W9_BINS:
        qs = np.linspace(0, 1, n + 1)[1:-1]
        edges[f"d_{n}"] = np.quantile(c["days"], qs)
        edges[f"r_{n}"] = np.quantile(c["ratio"], qs)
        edges[f"e_{n}"] = np.quantile(c["rate"], qs)
        edges[f"cr_{n}"] = np.quantile(c["cond_r"], qs)
        edges[f"k_{n}"] = np.quantile(c["rk"], qs)
        edges[f"b_{n}"] = np.quantile(c["blend"], qs)
        lo, hi = np.percentile(c["days_ps"], [0.5, 99.5])
        edges[f"dps_{n}"] = np.linspace(lo, hi, n + 1)[1:-1]
    return edges


def build_w9(df: pd.DataFrame, edges: dict) -> tuple[pd.DataFrame, list[str]]:
    c = _w9_core(df)
    out = pd.DataFrame(index=df.index)
    out["days"] = c["days"]
    out["log_days"] = c["log_days"]
    out["condition"] = c["cond"]
    out["condition_missing"] = c["cond"].isna().astype(int)
    out["cond_r"] = c["cond_r"]
    out["ratio"] = c["ratio"]
    out["log_ratio"] = c["log_ratio"]
    out["cond_rk"] = c["rk"]
    out["rate"] = c["rate"]
    out["log_rate"] = c["log_rate"]
    out["days_ps"] = c["days_ps"]
    out["blend"] = c["blend"]
    out["ratio_p75"] = c["days"] / c["cond_r"].clip(lower=1e-9) ** 0.75
    out["rate_over_age"] = c["rate"] / df["age_range"].astype(float).clip(lower=1e-9)
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
    cats += ["region", "source", "month", "version", "grades_c", "age_cat", "bin_pat", "days_fx"]

    for n in W9_BINS:
        out[f"d_q{n}"] = _qbins(c["days"], edges[f"d_{n}"]).astype(str)
        out[f"r_q{n}"] = _qbins(c["ratio"], edges[f"r_{n}"]).astype(str)
        out[f"e_q{n}"] = _qbins(c["rate"], edges[f"e_{n}"]).astype(str)
        out[f"cr_q{n}"] = _qbins(c["cond_r"], edges[f"cr_{n}"]).astype(str)
        out[f"k_q{n}"] = _qbins(c["rk"], edges[f"k_{n}"]).astype(str)
        out[f"b_q{n}"] = _qbins(c["blend"], edges[f"b_{n}"]).astype(str)
        out[f"dps_ew{n}"] = _qbins(c["days_ps"], edges[f"dps_{n}"]).astype(str)
        cats += [f"d_q{n}", f"r_q{n}", f"e_q{n}", f"cr_q{n}", f"k_q{n}", f"b_q{n}", f"dps_ew{n}"]

    _cross(out, cats, "reg_src", "region", "source")
    _cross(out, cats, "src_age", "source", "age_cat")
    _cross(out, cats, "reg_age", "region", "age_cat")
    _cross(out, cats, "d9_reg", "d_q9", "region")
    _cross(out, cats, "d9_src", "d_q9", "source")
    _cross(out, cats, "d18_reg", "d_q18", "region")
    _cross(out, cats, "d18_src", "d_q18", "source")
    _cross(out, cats, "r9_reg", "r_q9", "region")
    _cross(out, cats, "r9_src", "r_q9", "source")
    _cross(out, cats, "r9_age", "r_q9", "age_cat")
    _cross(out, cats, "e9_reg", "e_q9", "region")
    _cross(out, cats, "e9_src", "e_q9", "source")
    _cross(out, cats, "e9_age", "e_q9", "age_cat")
    _cross(out, cats, "k9_src", "k_q9", "source")
    _cross(out, cats, "k18_src", "k_q18", "source")
    _cross(out, cats, "k9_reg", "k_q9", "region")
    _cross(out, cats, "cr9_src", "cr_q9", "source")
    _cross(out, cats, "cr18_src", "cr_q18", "source")
    _cross(out, cats, "b9_reg", "b_q9", "region")
    _cross(out, cats, "b9_src", "b_q9", "source")
    _cross(out, cats, "d9_k9", "d_q9", "k_q9")
    _cross(out, cats, "d9_cr9", "d_q9", "cr_q9")
    _cross(out, cats, "d18_k18", "d_q18", "k_q18")
    _cross(out, cats, "e9_k9", "e_q9", "k_q9")
    _cross(out, cats, "r9_e9", "r_q9", "e_q9")
    _cross(out, cats, "dps9_k9", "dps_ew9", "k_q9")
    _cross(out, cats, "dps9_src", "dps_ew9", "source")
    _cross(out, cats, "d9_pat", "d_q9", "bin_pat")
    _cross(out, cats, "e9_pat", "e_q9", "bin_pat")
    _cross(out, cats, "d9_reg_src", "d_q9", "region", "source")
    _cross(out, cats, "e9_reg_src", "e_q9", "region", "source")
    _cross(out, cats, "k9_reg_age", "k_q9", "region", "age_cat")
    _cross(out, cats, "dfx_src", "days_fx", "source")
    _cross(out, cats, "dfx_k9", "days_fx", "k_q9")
    for col in ("region", "source", "bin_pat", "reg_src", "d9_reg", "k9_src", "e9_src"):
        out[f"freq_{col}"] = out[col].map(out[col].value_counts()).astype(float)
    return out, cats


def w9_frame(raw: pd.DataFrame, edges: dict, stream_offset: int, n_views: int = 4):
    X, cats = build_w9(raw, edges)
    add_noise_view(X, cats, raw)
    der = _w9_core(raw)
    # jitter around the blend axis so the stream differs from main/alt
    add_jitter_views(
        X, cats, raw, der["blend"], pd.to_numeric(raw["days"]),
        n_views=n_views, n_bins=18, stream_offset=500 + stream_offset,
    )
    for c in cats:
        X[c] = X[c].astype(str)
    num = [c for c in X.columns if c not in cats]
    X[num] = X[num].astype(float).fillna(-999.0)
    return X, cats

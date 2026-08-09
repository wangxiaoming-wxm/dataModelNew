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

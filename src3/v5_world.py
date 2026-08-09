"""V5 encoding world (w6): strong fourth view under the V2 protocol.

Design constraints forced by the V3 failure and by docs/HANDOFF.md §5.1:

* Evaluation stays on V2's ruler: stratified 5-fold, fixed tree counts, no
  early stopping, nested rule selection.  Changing the fold count is how V3
  manufactured a local lift that the leaderboard rejected.
* A new world only earns a seat in ``max`` if its bagged OOF reaches ~0.694;
  weaker arms drag ``max`` down (cat_alt2's lesson).
* Keep the two signals that carry this data (source-normalised condition and
  the days/condition rate); change only how they are expressed, so the arm
  decorrelates without going soft.

How w6 differs from main / alt / alt2
------------------------------------
* condition: robust z = (x - median) / (1.4826·MAD) inside ``source``
  (main uses median-ratio; alt uses rank; alt2 uses mean/std inside
  region×source).
* rate: ``log1p(days) - log(cond_r)`` binned on equal width in log space
  (geometric bins on the original ratio), not quantiles.
* day cuts: the fixed physical edges that produced the single strongest
  column in the whole project (``dfx_c10``, honest OOF-TE 0.628) are first-
  class, with several condition crossings — main has them, alt barely does.
* bin counts (6, 12, 24): disjoint from main (5,10,20,40), alt (7,13,25),
  alt2 (4,9,16).

Everything here is label-free.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features import BIN_COLS, GRADE_MAP, DAYS_FIXED_EDGES, add_noise_view
from jitter import add_jitter_views

W6_BINS = (6, 12, 24)


def _qbins(values, edges) -> np.ndarray:
    return np.digitize(np.asarray(values, dtype=float), edges)


def _cross(out, cats, name, *parts):
    s = out[parts[0]].astype(str)
    for p in parts[1:]:
        s = s + "|" + out[p].astype(str)
    out[name] = s
    cats.append(name)


def _w6_core(df: pd.DataFrame) -> pd.DataFrame:
    cond = pd.to_numeric(df["condition"])
    days = pd.to_numeric(df["days"])
    g = df.groupby("source")["condition"]
    med = g.transform("median")
    mad = (cond - med).abs().groupby(df["source"]).transform("median") * 1.4826
    rz = ((cond - med) / mad.replace(0, np.nan)).fillna(0.0).clip(-8, 8)
    cond_r = (cond / med.replace(0, np.nan)).fillna(1.0)
    log_days = np.log1p(days.clip(lower=0))
    log_rate = log_days - np.log(cond_r.clip(lower=1e-6))
    # a second rate that alt uses a rank for; here use the robust z so the
    # interaction is expressed on a signed scale rather than a percentile
    signed_rate = log_days - 0.5 * rz
    return pd.DataFrame(
        {
            "rz": rz,
            "cond_r": cond_r,
            "log_days": log_days,
            "log_rate": log_rate,
            "signed_rate": signed_rate,
            "days": days,
            "cond": cond,
        },
        index=df.index,
    )


def fit_edges_w6(df: pd.DataFrame) -> dict:
    c = _w6_core(df)
    edges: dict = {}
    for n in W6_BINS:
        qs = np.linspace(0, 1, n + 1)[1:-1]
        edges[f"rz_{n}"] = np.quantile(c["rz"], qs)
        edges[f"cr_{n}"] = np.quantile(c["cond_r"], qs)
        edges[f"ld_{n}"] = np.quantile(c["log_days"], qs)
        edges[f"sr_{n}"] = np.quantile(c["signed_rate"], qs)
        lo, hi = np.percentile(c["log_rate"], [0.5, 99.5])
        edges[f"lr_{n}"] = np.linspace(lo, hi, n + 1)[1:-1]
        edges[f"q_{n}"] = np.quantile(c["cond"].dropna(), qs)
    return edges


def build_w6(df: pd.DataFrame, edges: dict) -> tuple[pd.DataFrame, list[str]]:
    c = _w6_core(df)
    out = pd.DataFrame(index=df.index)
    out["rz"] = c["rz"]
    out["cond_r"] = c["cond_r"]
    out["log_days"] = c["log_days"]
    out["log_rate"] = c["log_rate"]
    out["signed_rate"] = c["signed_rate"]
    out["days"] = c["days"]
    out["condition"] = c["cond"]
    out["condition_missing"] = c["cond"].isna().astype(int)
    out["ratio"] = c["days"] / c["cond_r"].clip(lower=1e-9)
    out["ratio_p75"] = c["days"] / c["cond_r"].clip(lower=1e-9) ** 0.75
    out["rz_x_days"] = c["rz"] * c["log_days"]
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

    for n in W6_BINS:
        out[f"Z{n}"] = _qbins(c["rz"], edges[f"rz_{n}"]).astype(str)
        out[f"C{n}"] = _qbins(c["cond_r"], edges[f"cr_{n}"]).astype(str)
        out[f"D{n}"] = _qbins(c["log_days"], edges[f"ld_{n}"]).astype(str)
        out[f"R{n}"] = _qbins(c["log_rate"], edges[f"lr_{n}"]).astype(str)
        out[f"S{n}"] = _qbins(c["signed_rate"], edges[f"sr_{n}"]).astype(str)
        out[f"Q{n}"] = _qbins(c["cond"].fillna(-1), edges[f"q_{n}"]).astype(str)
        cats += [f"Z{n}", f"C{n}", f"D{n}", f"R{n}", f"S{n}", f"Q{n}"]

    # hook for add_noise_view's t3 × days_q5 cross
    qs5 = np.linspace(0, 1, 6)[1:-1]
    out["days_q5"] = _qbins(c["log_days"], np.quantile(c["log_days"], qs5)).astype(str)
    cats.append("days_q5")

    x = _cross
    # condition × source at several resolutions (the interaction that moves
    # this dataset); both robust-z and raw-condition versions
    x(out, cats, "W6_Z6_src", "Z6", "source")
    x(out, cats, "W6_Z12_src", "Z12", "source")
    x(out, cats, "W6_Z24_src", "Z24", "source")
    x(out, cats, "W6_C6_src", "C6", "source")
    x(out, cats, "W6_C12_src", "C12", "source")
    x(out, cats, "W6_C24_src", "C24", "source")
    x(out, cats, "W6_Q6_src", "Q6", "source")
    x(out, cats, "W6_Q12_src", "Q12", "source")
    x(out, cats, "W6_Q12_reg", "Q12", "region")
    # fixed-day × condition family — strongest single columns in the project
    x(out, cats, "W6_dfx_Q12", "days_fx", "Q12")
    x(out, cats, "W6_dfx_C12", "days_fx", "C12")
    x(out, cats, "W6_dfx_Z12", "days_fx", "Z12")
    x(out, cats, "W6_dfx_R12", "days_fx", "R12")
    x(out, cats, "W6_dfx_src", "days_fx", "source")
    x(out, cats, "W6_dfx_reg", "days_fx", "region")
    # rate / signed-rate crossings
    x(out, cats, "W6_R12_src", "R12", "source")
    x(out, cats, "W6_R12_reg", "R12", "region")
    x(out, cats, "W6_R6_age", "R6", "age_cat")
    x(out, cats, "W6_R6_pat", "R6", "bin_pat")
    x(out, cats, "W6_S12_src", "S12", "source")
    x(out, cats, "W6_S12_reg", "S12", "region")
    x(out, cats, "W6_S6_age", "S6", "age_cat")
    # joint day × condition cells
    x(out, cats, "W6_D6_C6", "D6", "C6")
    x(out, cats, "W6_D12_C12", "D12", "C12")
    x(out, cats, "W6_D6_Z6", "D6", "Z6")
    x(out, cats, "W6_D12_Z12", "D12", "Z12")
    x(out, cats, "W6_D6_reg_src", "D6", "region", "source")
    x(out, cats, "W6_R6_reg_src", "R6", "region", "source")
    x(out, cats, "W6_C6_reg_age", "C6", "region", "age_cat")
    x(out, cats, "W6_reg_src", "region", "source")
    x(out, cats, "W6_reg_age", "region", "age_cat")
    x(out, cats, "W6_src_age", "source", "age_cat")
    x(out, cats, "W6_reg_pat", "region", "bin_pat")
    x(out, cats, "W6_D6_pat", "D6", "bin_pat")
    for col in ("region", "source", "bin_pat", "W6_reg_src", "W6_Z12_src",
                "W6_dfx_Q12", "W6_R12_reg"):
        out[f"freq_{col}"] = out[col].map(out[col].value_counts()).astype(float)
    return out, cats


def w6_frame(raw: pd.DataFrame, edges: dict, stream_offset: int, n_views: int = 4):
    X, cats = build_w6(raw, edges)
    add_noise_view(X, cats, raw)
    c = _w6_core(raw)
    add_jitter_views(
        X, cats, raw, c["rz"], pd.to_numeric(raw["days"]),
        n_views=n_views, n_bins=12, stream_offset=250 + stream_offset,
    )
    for cname in cats:
        X[cname] = X[cname].astype(str)
    num = [c for c in X.columns if c not in cats]
    X[num] = X[num].astype(float).fillna(-999.0)
    return X, cats

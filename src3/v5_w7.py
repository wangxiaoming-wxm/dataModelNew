"""V5 world w7: main-world clone with surgical re-expression.

The w6 attempt bagged at 0.689 and was correctly rejected by the ≥0.694 gate.
cat_alt works because it keeps strength while changing the encoding; w6 lost
too much strength.  w7 therefore starts from the proven ``build(..., cross2)``
pipeline and only swaps three things:

1. condition scale: robust MAD z inside source, then ``cond_r = softplus(rz)``
   mapped back to a positive scale so ``ratio = days/cond_r`` still exists;
2. quantile grid: (6, 12, 24) instead of (5, 10, 20, 40);
3. jitter stream family: offset base 300, n_bins=12.

Noise view and the dfx / condition×source crosses that carry this dataset are
kept intact.  Label-free throughout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features import (
    BIN_COLS, GRADE_MAP, DAYS_FIXED_EDGES, QUANTS, add_noise_view, _qbins, _derive,
)
from jitter import add_jitter_views

# Override the quantile grid for this world only.
W7_QUANTS = (6, 12, 24)


def _robust_scale(df: pd.DataFrame) -> pd.Series:
    cond = pd.to_numeric(df["condition"])
    g = df.groupby("source")["condition"]
    med = g.transform("median")
    mad = (cond - med).abs().groupby(df["source"]).transform("median") * 1.4826
    rz = ((cond - med) / mad.replace(0, np.nan)).fillna(0.0).clip(-8, 8)
    # map signed robust z to a positive "ratio-like" scale centred at 1
    return pd.Series(np.exp(0.5 * rz.to_numpy(dtype=float)), index=df.index)


def fit_edges_w7(df: pd.DataFrame) -> dict:
    scale = _robust_scale(df)  # stored per-row? no — we need per-source med/mad
    # store the per-source median and mad so transform is stable on train+test
    cond = pd.to_numeric(df["condition"])
    med = df.groupby("source")["condition"].median()
    # MAD per source
    tmp = df[["source"]].copy()
    tmp["dev"] = (cond - df["source"].map(med)).abs()
    mad = tmp.groupby("source")["dev"].median() * 1.4826
    # derive ratio on the robust scale for quantile edges
    rz = ((cond - df["source"].map(med)) / df["source"].map(mad).replace(0, np.nan)).fillna(0.0).clip(-8, 8)
    cond_r = pd.Series(np.exp(0.5 * rz.to_numpy(dtype=float)), index=df.index)
    days = pd.to_numeric(df["days"])
    ratio = days / cond_r.clip(lower=1e-9)
    edges: dict = {"__med__": med, "__mad__": mad}
    for n in W7_QUANTS:
        qs = np.linspace(0, 1, n + 1)[1:-1]
        edges[f"days_{n}"] = np.quantile(days.dropna(), qs)
        edges[f"cond_{n}"] = np.quantile(cond.dropna(), qs)
        edges[f"condr_{n}"] = np.quantile(cond_r, qs)
        edges[f"ratio_{n}"] = np.quantile(ratio, qs)
    return edges


def build_w7(df: pd.DataFrame, edges: dict) -> tuple[pd.DataFrame, list[str]]:
    """Mirror features.build(cross2) but on the robust scale / W7_QUANTS."""
    out = pd.DataFrame(index=df.index)
    days = pd.to_numeric(df["days"])
    cond = pd.to_numeric(df["condition"])
    med, mad = edges["__med__"], edges["__mad__"]
    rz = ((cond - df["source"].map(med)) / df["source"].map(mad).replace(0, np.nan)).fillna(0.0).clip(-8, 8)
    cond_r = pd.Series(np.exp(0.5 * rz.to_numpy(dtype=float)), index=df.index)
    ratio = days / cond_r.clip(lower=1e-9)

    out["days"] = days
    out["log_days"] = np.log1p(days.clip(lower=0))
    out["condition"] = cond
    out["log_condition"] = np.log1p(cond.clip(lower=0))
    out["condition_missing"] = cond.isna().astype(int)
    out["cond_r"] = cond_r
    out["log_cond_r"] = np.log(cond_r.clip(lower=1e-9))
    out["rz"] = rz
    out["ratio"] = ratio
    out["log_ratio"] = np.log(ratio.clip(lower=1e-9))
    out["ratio_p75"] = days / cond_r.clip(lower=1e-9) ** 0.75
    out["cond_x_days"] = cond * days
    out["cond_over_days"] = cond / (days.abs() + 1.0)
    out["age_range"] = df["age_range"].astype(float)
    out["days_over_age"] = days / df["age_range"].astype(float)
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

    for n in W7_QUANTS:
        out[f"days_q{n}"] = _qbins(days, edges[f"days_{n}"]).astype(str)
        out[f"ratio_q{n}"] = _qbins(ratio, edges[f"ratio_{n}"]).astype(str)
        cats += [f"days_q{n}", f"ratio_q{n}"]
    for n in (6, 12):
        out[f"cond_q{n}"] = _qbins(cond.fillna(-1), edges[f"cond_{n}"]).astype(str)
        out[f"condr_q{n}"] = _qbins(cond_r, edges[f"condr_{n}"]).astype(str)
        cats += [f"cond_q{n}", f"condr_q{n}"]

    # aliases so the cross list below can reuse main's naming patterns
    out["days_q5"] = out["days_q6"]
    out["days_q10"] = out["days_q12"]
    out["days_q20"] = out["days_q24"]
    out["cond_q5"] = out["cond_q6"]
    out["cond_q10"] = out["cond_q12"]
    out["cond_q20"] = out["cond_q12"]  # no 24-bin cond; reuse 12
    out["condr_q5"] = out["condr_q6"]
    out["condr_q10"] = out["condr_q12"]
    out["condr_q20"] = out["condr_q12"]
    out["ratio_q5"] = out["ratio_q6"]
    out["ratio_q10"] = out["ratio_q12"]
    out["ratio_q20"] = out["ratio_q24"]
    cats += ["days_q5", "days_q10", "days_q20", "cond_q5", "cond_q10", "cond_q20",
             "condr_q5", "condr_q10", "condr_q20", "ratio_q5", "ratio_q10", "ratio_q20"]

    def cross(name, *parts):
        s = out[parts[0]].astype(str)
        for p in parts[1:]:
            s = s + "|" + out[p].astype(str)
        out[name] = s
        cats.append(name)

    # identical cross recipe to features.build(cross2) — this is what keeps strength
    cross("reg_src", "region", "source")
    cross("d10_reg", "days_q10", "region")
    cross("d10_src", "days_q10", "source")
    cross("d20_reg", "days_q20", "region")
    cross("d20_src", "days_q20", "source")
    cross("d10_age", "days_q10", "age_cat")
    cross("d10_c10", "days_q10", "cond_q10")
    cross("c10_reg", "cond_q10", "region")
    cross("c10_src", "cond_q10", "source")
    cross("reg_age", "region", "age_cat")
    cross("src_age", "source", "age_cat")
    cross("d10_pat", "days_q10", "bin_pat")
    cross("reg_pat", "region", "bin_pat")
    cross("d5_reg_src", "days_q5", "region", "source")
    cross("r10_reg", "ratio_q10", "region")
    cross("r10_src", "ratio_q10", "source")
    cross("r10_age", "ratio_q10", "age_cat")
    cross("r20_reg", "ratio_q20", "region")
    cross("r10_pat", "ratio_q10", "bin_pat")
    cross("cr10_reg", "condr_q10", "region")
    cross("cr10_age", "condr_q10", "age_cat")
    cross("c5_src", "cond_q5", "source")
    cross("c20_src", "cond_q20", "source")
    cross("cr5_src", "condr_q5", "source")
    cross("cr10_src", "condr_q10", "source")
    cross("cr20_src", "condr_q20", "source")
    cross("cr5_reg", "condr_q5", "region")
    cross("cr20_reg", "condr_q20", "region")
    cross("c5_reg", "cond_q5", "region")
    cross("d5_c5", "days_q5", "cond_q5")
    cross("d20_c20", "days_q20", "cond_q20")
    cross("d5_cr5", "days_q5", "condr_q5")
    cross("d10_cr10", "days_q10", "condr_q10")
    cross("d10c10_reg", "days_q10", "cond_q10", "region")
    cross("d10c10_src", "days_q10", "cond_q10", "source")
    cross("d10c10_age", "days_q10", "cond_q10", "age_cat")
    cross("src_c10_age", "source", "cond_q10", "age_cat")
    cross("reg_c10_age", "region", "cond_q10", "age_cat")
    cross("reg_src_age", "region", "source", "age_cat")
    cross("dfx_src", "days_fx", "source")
    cross("dfx_c10", "days_fx", "cond_q10")
    cross("dfx_cr10", "days_fx", "condr_q10")
    cross("dfx_reg", "days_fx", "region")
    cross("r5_reg_src", "ratio_q5", "region", "source")
    for c in ("region", "source", "bin_pat", "reg_src", "d10_reg", "c10_src", "month", "version"):
        out[f"freq_{c}"] = out[c].map(out[c].value_counts()).astype(float)
    return out, cats


def w7_frame(raw: pd.DataFrame, edges: dict, stream_offset: int, n_views: int = 4):
    X, cats = build_w7(raw, edges)
    add_noise_view(X, cats, raw)
    cond_r = X["cond_r"]
    add_jitter_views(
        X, cats, raw, cond_r, pd.to_numeric(raw["days"]),
        n_views=n_views, n_bins=12, stream_offset=300 + stream_offset,
    )
    for c in cats:
        X[c] = X[c].astype(str)
    num = [c for c in X.columns if c not in cats]
    X[num] = X[num].astype(float).fillna(-999.0)
    return X, cats

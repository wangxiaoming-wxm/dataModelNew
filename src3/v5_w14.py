"""V5 world w14: main world with condition scaled inside source × age_range.

w11 proved that grafting alt onto main raises single-arm strength (bag 0.696)
but stays too collinear with cat_d5 (corr 0.989) for ``max`` to move.

w14 keeps the full main ``cross2`` recipe and only changes the *scale* used
for cond_r / ratio: median(condition | source, age_range) instead of
median(condition | source).  That is the HANDOFF §5.1 suggestion
「按 source × age 分组」, and it should decorrelate from frozen cat_d5
without discarding the carrying signals.

Label-free.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features import (BIN_COLS, GRADE_MAP, DAYS_FIXED_EDGES, QUANTS,
                      add_noise_view, _qbins, _derive)
from jitter import add_jitter_views


def _scale_src_age(df: pd.DataFrame) -> pd.Series:
    return df.groupby(["source", "age_range"])["condition"].transform("median")


def fit_edges_w14(df: pd.DataFrame) -> dict:
    scale = _scale_src_age(df)
    # _derive expects a Series indexed by source; build a compatible frame
    days = pd.to_numeric(df["days"])
    cond = pd.to_numeric(df["condition"])
    cond_r = (cond / scale.replace(0, np.nan)).fillna(1.0)
    ratio = days / cond_r.clip(lower=1e-9)
    edges: dict = {"__scale_sa__": True}
    for n in QUANTS:
        qs = np.linspace(0, 1, n + 1)[1:-1]
        edges[f"days_{n}"] = np.quantile(days.dropna(), qs)
        edges[f"cond_{n}"] = np.quantile(cond.dropna(), qs)
        edges[f"condr_{n}"] = np.quantile(cond_r, qs)
        edges[f"ratio_{n}"] = np.quantile(ratio, qs)
    return edges


def build_w14(df: pd.DataFrame, edges: dict) -> tuple[pd.DataFrame, list[str]]:
    """Mirror features.build(..., 'cross2') but with source×age cond_r."""
    out = pd.DataFrame(index=df.index)
    days = pd.to_numeric(df["days"])
    cond = pd.to_numeric(df["condition"])
    scale = _scale_src_age(df)
    cond_r = (cond / scale.replace(0, np.nan)).fillna(1.0)
    ratio = days / cond_r.clip(lower=1e-9)

    out["days"] = days
    out["log_days"] = np.log1p(days.clip(lower=0))
    out["condition"] = cond
    out["log_condition"] = np.log1p(cond.clip(lower=0))
    out["condition_missing"] = cond.isna().astype(int)
    out["cond_r"] = cond_r
    out["log_cond_r"] = np.log(cond_r.clip(lower=1e-9))
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

    for n in QUANTS:
        out[f"days_q{n}"] = _qbins(days, edges[f"days_{n}"]).astype(str)
        out[f"ratio_q{n}"] = _qbins(ratio, edges[f"ratio_{n}"]).astype(str)
        cats += [f"days_q{n}", f"ratio_q{n}"]
    for n in (5, 10, 20):
        out[f"cond_q{n}"] = _qbins(cond.fillna(-1), edges[f"cond_{n}"]).astype(str)
        out[f"condr_q{n}"] = _qbins(cond_r, edges[f"condr_{n}"]).astype(str)
        cats += [f"cond_q{n}", f"condr_q{n}"]

    def cross(name: str, *parts: str) -> None:
        s = out[parts[0]].astype(str)
        for p in parts[1:]:
            s = s + "|" + out[p].astype(str)
        out[name] = s
        cats.append(name)

    # same cross list as features.build cross2
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


def w14_frame(raw: pd.DataFrame, edges: dict, stream_offset: int, n_views: int = 4):
    X, cats = build_w14(raw, edges)
    add_noise_view(X, cats, raw)
    scale = _scale_src_age(raw)
    cond_r = (pd.to_numeric(raw["condition"]) / scale.replace(0, np.nan)).fillna(1.0)
    add_jitter_views(X, cats, raw, cond_r, pd.to_numeric(raw["days"]),
                     n_views=n_views, stream_offset=90 + stream_offset)
    for c in cats:
        X[c] = X[c].astype(str)
    num = [c for c in X.columns if c not in cats]
    X[num] = X[num].astype(float).fillna(-999.0)
    return X, cats

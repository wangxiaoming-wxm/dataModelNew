"""V5 world w10: strengthen alt2 by grafting main's cond_r / ratio signals.

``cat_alt2`` decorrelates well (~0.95) but sits at bag 0.691 and drags ``max``.
HANDOFF §5.1 says the fix is to keep the two carrying signals — condition
normalised inside the vehicle model, and the days/condition rate — while
preserving alt2's distinct region×source z-score expression for diversity.

w10 therefore:
  * keeps cz / dpc / load and alt2's bin grid (4, 9, 16);
  * adds cond_r = condition / median(condition|source) and ratio = days / cond_r
    (exact main definitions);
  * adds the strongest main-style crosses on those signals (cr*_src, r*_src,
    dfx family) without deleting alt2's own crosses.

Label-free.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features import BIN_COLS, GRADE_MAP, DAYS_FIXED_EDGES, add_noise_view, _derive
from jitter import add_jitter_views

W10_QUANTS = (4, 9, 16)


def _qbins(values, edges):
    return np.digitize(np.asarray(values, dtype=float), edges)


def fit_edges_w10(df: pd.DataFrame) -> dict:
    g = df.groupby(["source", "region"])["condition"]
    cz = ((df["condition"] - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0.0)
    dpc = df.groupby("region")["days"].rank(pct=True)
    load = cz - 2.0 * dpc
    scale = df.groupby("source")["condition"].median()
    der = _derive(df, scale)
    edges: dict = {"__scale__": scale}
    for n in W10_QUANTS:
        qs = np.linspace(0, 1, n + 1)[1:-1]
        edges[f"cz_{n}"] = np.quantile(cz, qs)
        edges[f"dpc_{n}"] = np.quantile(dpc, qs)
        edges[f"load_{n}"] = np.quantile(load, qs)
        edges[f"condr_{n}"] = np.quantile(der["cond_r"], qs)
        edges[f"ratio_{n}"] = np.quantile(der["ratio"], qs)
        edges[f"days_{n}"] = np.quantile(pd.to_numeric(df["days"]).dropna(), qs)
    return edges


def build_w10(df: pd.DataFrame, edges: dict) -> tuple[pd.DataFrame, list[str]]:
    out = pd.DataFrame(index=df.index)
    days = pd.to_numeric(df["days"])
    cond = pd.to_numeric(df["condition"])
    g = df.groupby(["source", "region"])["condition"]
    cz = ((cond - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0.0)
    dpc = df.groupby("region")["days"].rank(pct=True)
    load = cz - 2.0 * dpc
    der = _derive(df, edges["__scale__"])
    cond_r, ratio = der["cond_r"], der["ratio"]

    out["cz"] = cz
    out["dpc"] = dpc
    out["load"] = load
    out["days"] = days
    out["condition"] = cond
    out["condition_missing"] = cond.isna().astype(int)
    out["cz_x_age"] = cz * df["age_range"].astype(float)
    out["load_x_bin"] = load * df[BIN_COLS].sum(axis=1)
    # grafted main signals
    out["cond_r"] = cond_r
    out["log_cond_r"] = np.log(cond_r.clip(lower=1e-9))
    out["ratio"] = ratio
    out["log_ratio"] = np.log(ratio.clip(lower=1e-9))
    out["ratio_p75"] = days / cond_r.clip(lower=1e-9) ** 0.75
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

    for n in W10_QUANTS:
        out[f"z{n}"] = _qbins(cz, edges[f"cz_{n}"]).astype(str)
        out[f"p{n}"] = _qbins(dpc, edges[f"dpc_{n}"]).astype(str)
        out[f"l{n}"] = _qbins(load, edges[f"load_{n}"]).astype(str)
        out[f"cr{n}"] = _qbins(cond_r, edges[f"condr_{n}"]).astype(str)
        out[f"r{n}"] = _qbins(ratio, edges[f"ratio_{n}"]).astype(str)
        out[f"d{n}"] = _qbins(days, edges[f"days_{n}"]).astype(str)
        cats += [f"z{n}", f"p{n}", f"l{n}", f"cr{n}", f"r{n}", f"d{n}"]

    def cross(name, *parts):
        s = out[parts[0]].astype(str)
        for p in parts[1:]:
            s = s + "|" + out[p].astype(str)
        out[name] = s
        cats.append(name)

    # alt2 crosses (kept)
    cross("B_z9_src", "z9", "source")
    cross("B_z16_src", "z16", "source")
    cross("B_z9_reg", "z9", "region")
    cross("B_z4_age", "z4", "age_cat")
    cross("B_p9_reg", "p9", "region")
    cross("B_p9_src", "p9", "source")
    cross("B_p16_reg", "p16", "region")
    cross("B_p4_age", "p4", "age_cat")
    cross("B_l9_reg", "l9", "region")
    cross("B_l9_src", "l9", "source")
    cross("B_l16_src", "l16", "source")
    cross("B_l4_pat", "l4", "bin_pat")
    cross("B_p9_z9", "p9", "z9")
    cross("B_p4_z4", "p4", "z4")
    cross("B_reg_src", "region", "source")
    cross("B_reg_age", "region", "age_cat")
    cross("B_src_age", "source", "age_cat")
    cross("B_reg_src_age", "region", "source", "age_cat")
    cross("B_p4_reg_src", "p4", "region", "source")
    cross("B_l4_reg_src", "l4", "region", "source")
    cross("B_z4_reg_age", "z4", "region", "age_cat")
    cross("B_p9_pat", "p9", "bin_pat")
    cross("B_reg_pat", "region", "bin_pat")

    # grafted main-strength crosses on cond_r / ratio
    cross("G_cr4_src", "cr4", "source")
    cross("G_cr9_src", "cr9", "source")
    cross("G_cr16_src", "cr16", "source")
    cross("G_cr9_reg", "cr9", "region")
    cross("G_r4_src", "r4", "source")
    cross("G_r9_src", "r9", "source")
    cross("G_r16_src", "r16", "source")
    cross("G_r9_reg", "r9", "region")
    cross("G_r9_age", "r9", "age_cat")
    cross("G_d9_cr9", "d9", "cr9")
    cross("G_d9_r9", "d9", "r9")
    cross("G_dfx_src", "days_fx", "source")
    cross("G_dfx_cr9", "days_fx", "cr9")
    cross("G_dfx_r9", "days_fx", "r9")
    cross("G_cr9_age", "cr9", "age_cat")
    cross("G_src_cr9_age", "source", "cr9", "age_cat")

    for c in ("region", "source", "bin_pat", "B_reg_src", "B_z9_src", "G_cr9_src"):
        out[f"freq_{c}"] = out[c].map(out[c].value_counts()).astype(float)
    return out, cats


def w10_frame(raw: pd.DataFrame, edges: dict, stream_offset: int, n_views: int = 3):
    X, cats = build_w10(raw, edges)
    add_noise_view(X, cats, raw)
    # jitter on cond_r (main's scale) for a third stream family
    der = _derive(raw, edges["__scale__"])
    add_jitter_views(X, cats, raw, der["cond_r"], pd.to_numeric(raw["days"]),
                     n_views=n_views, n_bins=9, stream_offset=110 + stream_offset)
    for c in cats:
        X[c] = X[c].astype(str)
    num = [c for c in X.columns if c not in cats]
    X[num] = X[num].astype(float).fillna(-999.0)
    return X, cats

"""V5 world w8: near-clone of the proven ``build_alt`` world.

w6/w7 lost ~0.006 of strength by inventing new feature recipes.  w8 refuses
to invent: it copies ``features.build_alt`` line-for-line, then applies only
three controlled diffs that are known (HANDOFF §5.1) to produce a new encoding
world without going soft:

1. quantile grid (9, 17, 33) instead of (7, 13, 25);
2. rate = days / (rank + 0.15) instead of days * (1 - rank);
3. add the fixed-day ``days_fx`` × condition-rank crossings that the main
   world proved are the single strongest columns.

Noise view + jitter follow the alt arm.  Label-free.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features import BIN_COLS, GRADE_MAP, DAYS_FIXED_EDGES, add_noise_view
from jitter import add_jitter_views

W8_QUANTS = (9, 17, 33)


def _qbins(values, edges):
    return np.digitize(np.asarray(values, dtype=float), edges)


def fit_edges_w8(df: pd.DataFrame) -> dict:
    rk = df.groupby("source")["condition"].rank(pct=True).fillna(0.5)
    days = pd.to_numeric(df["days"])
    rate = days / (rk + 0.15)
    edges: dict = {}
    for n in W8_QUANTS:
        qs = np.linspace(0, 1, n + 1)[1:-1]
        edges[f"days_{n}"] = np.quantile(days.dropna(), qs)
        edges[f"crk_{n}"] = np.quantile(rk, qs)
        edges[f"rate_{n}"] = np.quantile(rate, qs)
    return edges


def build_w8(df: pd.DataFrame, edges: dict) -> tuple[pd.DataFrame, list[str]]:
    out = pd.DataFrame(index=df.index)
    days = pd.to_numeric(df["days"])
    cond = pd.to_numeric(df["condition"])
    rk = df.groupby("source")["condition"].rank(pct=True).fillna(0.5)
    rate = days / (rk + 0.15)

    out["days"] = days
    out["sqrt_days"] = np.sqrt(days.clip(lower=0))
    out["condition"] = cond
    out["cond_rk"] = rk
    out["rate"] = rate
    out["log_rate"] = np.log1p(rate.clip(lower=0))
    out["rate_over_age"] = rate / df["age_range"].astype(float)
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

    for n in W8_QUANTS:
        out[f"d{n}"] = _qbins(days, edges[f"days_{n}"]).astype(str)
        out[f"k{n}"] = _qbins(rk, edges[f"crk_{n}"]).astype(str)
        out[f"e{n}"] = _qbins(rate, edges[f"rate_{n}"]).astype(str)
        cats += [f"d{n}", f"k{n}", f"e{n}"]

    # aliases so alt-style cross names and noise-view hooks keep working
    out["d7"], out["d13"], out["d25"] = out["d9"], out["d17"], out["d33"]
    out["k7"], out["k13"], out["k25"] = out["k9"], out["k17"], out["k33"]
    out["e7"], out["e13"], out["e25"] = out["e9"], out["e17"], out["e33"]
    cats += ["d7", "d13", "d25", "k7", "k13", "k25", "e7", "e13", "e25"]
    out["days_q5"] = out["d9"]
    cats.append("days_q5")

    def cross(name, *parts):
        s = out[parts[0]].astype(str)
        for p in parts[1:]:
            s = s + "|" + out[p].astype(str)
        out[name] = s
        cats.append(name)

    # alt's cross list (verbatim) + the dfx family main proved is strongest
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
    cross("A_dfx_reg", "days_fx", "region")
    for c in ("region", "source", "bin_pat", "A_reg_src", "A_k13_src", "A_d13_reg", "A_dfx_k13"):
        out[f"freq_{c}"] = out[c].map(out[c].value_counts()).astype(float)
    return out, cats


def w8_frame(raw: pd.DataFrame, edges: dict, stream_offset: int, n_views: int = 3):
    X, cats = build_w8(raw, edges)
    add_noise_view(X, cats, raw)
    rk = raw.groupby("source")["condition"].rank(pct=True).fillna(0.5)
    add_jitter_views(
        X, cats, raw, rk, pd.to_numeric(raw["days"]),
        n_views=n_views, n_bins=17, stream_offset=400 + stream_offset,
    )
    for c in cats:
        X[c] = X[c].astype(str)
    num = [c for c in X.columns if c not in cats]
    X[num] = X[num].astype(float).fillna(-999.0)
    return X, cats

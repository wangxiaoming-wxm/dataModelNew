"""V5 world w9: clean near-clone of proven ``build_alt``.

w8 failed (bag 0.6907) after two self-inflicted wounds:
  - it changed the rate formula away from the proven ``days * (1 - rank)``;
  - it registered duplicate alias columns (d7/d9 pointing at the same series),
    which dilutes CatBoost's ordered statistics.

w9 keeps the alt world *verbatim* on the signal side and only applies the
HANDOFF-safe diffs:

1. quantile grid (8, 15, 30) instead of (7, 13, 25);
2. add the fixed-day ``days_fx`` × condition-rank crossings from main;
3. jitter uses a different stream family and bin count.

Label-free.  Rate formula is exactly ``days * (1 - rank)``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features import BIN_COLS, GRADE_MAP, DAYS_FIXED_EDGES, add_noise_view
from jitter import add_jitter_views

W9_QUANTS = (8, 15, 30)


def _qbins(values, edges):
    return np.digitize(np.asarray(values, dtype=float), edges)


def fit_edges_w9(df: pd.DataFrame) -> dict:
    rk = df.groupby("source")["condition"].rank(pct=True).fillna(0.5)
    days = pd.to_numeric(df["days"])
    rate = days * (1.0 - rk)
    edges: dict = {}
    for n in W9_QUANTS:
        qs = np.linspace(0, 1, n + 1)[1:-1]
        edges[f"days_{n}"] = np.quantile(days.dropna(), qs)
        edges[f"crk_{n}"] = np.quantile(rk, qs)
        edges[f"rate_{n}"] = np.quantile(rate, qs)
    return edges


def build_w9(df: pd.DataFrame, edges: dict) -> tuple[pd.DataFrame, list[str]]:
    out = pd.DataFrame(index=df.index)
    days = pd.to_numeric(df["days"])
    cond = pd.to_numeric(df["condition"])
    rk = df.groupby("source")["condition"].rank(pct=True).fillna(0.5)
    rate = days * (1.0 - rk)

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

    # primary bin names at the new grid; short aliases only for cross names
    # (aliases are NOT added to ``cats`` — that was w8's bug)
    for n in W9_QUANTS:
        out[f"d{n}"] = _qbins(days, edges[f"days_{n}"]).astype(str)
        out[f"k{n}"] = _qbins(rk, edges[f"crk_{n}"]).astype(str)
        out[f"e{n}"] = _qbins(rate, edges[f"rate_{n}"]).astype(str)
        cats += [f"d{n}", f"k{n}", f"e{n}"]

    out["d7"], out["d13"], out["d25"] = out["d8"], out["d15"], out["d30"]
    out["k7"], out["k13"], out["k25"] = out["k8"], out["k15"], out["k30"]
    out["e7"], out["e13"], out["e25"] = out["e8"], out["e15"], out["e30"]

    def cross(name, *parts):
        s = out[parts[0]].astype(str)
        for p in parts[1:]:
            s = s + "|" + out[p].astype(str)
        out[name] = s
        cats.append(name)

    # alt cross list (names kept for readability) + dfx family
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

    for c in ("region", "source", "bin_pat", "A_reg_src", "A_k13_src", "A_d13_reg"):
        out[f"freq_{c}"] = out[c].map(out[c].value_counts()).astype(float)
    return out, cats


def w9_frame(raw: pd.DataFrame, edges: dict, stream_offset: int, n_views: int = 3):
    X, cats = build_w9(raw, edges)
    add_noise_view(X, cats, raw)
    # noise_view looks for d7; we already aliased it as a column (not in cats)
    if "d7" not in X.columns:
        X["d7"] = X["d8"]
    rk = raw.groupby("source")["condition"].rank(pct=True).fillna(0.5)
    add_jitter_views(X, cats, raw, rk, pd.to_numeric(raw["days"]),
                     n_views=n_views, n_bins=11, stream_offset=70 + stream_offset)
    for c in cats:
        X[c] = X[c].astype(str)
    num = [c for c in X.columns if c not in cats]
    X[num] = X[num].astype(float).fillna(-999.0)
    return X, cats

"""V5 world w11: main encoding world with alt's rank/rate grafted in.

Goal: keep main's proven strength (cond_r / ratio / cross2) and add alt's
rank-normalised condition + rate as a parallel expression so the arm is
both strong (≥0.694) and slightly decorrelated from frozen cat_d5/d6.

Diffs vs main ``catboost_frame``:
  * add cond_rk = rank(condition|source) and rate = days*(1-rk) numerics;
  * add alt-style quantile bins (7,13,25) and the strongest k*_src / e*_src
    crosses on top of the existing main cross list;
  * jitter stream family offset shifted (+40) so the noise views differ.

Label-free.  Does not remove any main column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features import (BIN_COLS, GRADE_MAP, DAYS_FIXED_EDGES, add_noise_view,
                      build, fit_edges, _qbins)
from jitter import add_jitter_views

W11_ALT_Q = (7, 13, 25)


def fit_edges_w11(df: pd.DataFrame) -> dict:
    edges = fit_edges(df)
    rk = df.groupby("source")["condition"].rank(pct=True).fillna(0.5)
    days = pd.to_numeric(df["days"])
    rate = days * (1.0 - rk)
    for n in W11_ALT_Q:
        qs = np.linspace(0, 1, n + 1)[1:-1]
        edges[f"w11_days_{n}"] = np.quantile(days.dropna(), qs)
        edges[f"w11_crk_{n}"] = np.quantile(rk, qs)
        edges[f"w11_rate_{n}"] = np.quantile(rate, qs)
    return edges


def w11_frame(raw: pd.DataFrame, edges: dict, stream_offset: int, n_views: int = 4):
    X, cats = build(raw, edges, "cross2")
    days = pd.to_numeric(raw["days"])
    rk = raw.groupby("source")["condition"].rank(pct=True).fillna(0.5)
    rate = days * (1.0 - rk)

    X["cond_rk"] = rk
    X["rate_alt"] = rate
    X["log_rate_alt"] = np.log1p(rate.clip(lower=0))

    for n in W11_ALT_Q:
        X[f"ak{n}"] = _qbins(rk, edges[f"w11_crk_{n}"]).astype(str)
        X[f"ae{n}"] = _qbins(rate, edges[f"w11_rate_{n}"]).astype(str)
        X[f"ad{n}"] = _qbins(days, edges[f"w11_days_{n}"]).astype(str)
        cats += [f"ak{n}", f"ae{n}", f"ad{n}"]

    def cross(name, *parts):
        s = X[parts[0]].astype(str)
        for p in parts[1:]:
            s = s + "|" + X[p].astype(str)
        X[name] = s
        cats.append(name)

    cross("W_ak7_src", "ak7", "source")
    cross("W_ak13_src", "ak13", "source")
    cross("W_ak25_src", "ak25", "source")
    cross("W_ae13_src", "ae13", "source")
    cross("W_ae13_reg", "ae13", "region")
    cross("W_ad13_src", "ad13", "source")
    cross("W_ad13_ak13", "ad13", "ak13")
    cross("W_ak13_reg", "ak13", "region")
    cross("W_ak7_age", "ak7", "age_cat")

    add_noise_view(X, cats, raw)
    # jitter on cond_r (main scale) with shifted stream
    from features import _derive
    der = _derive(raw, edges["__scale__"])
    add_jitter_views(X, cats, raw, der["cond_r"], days,
                     n_views=n_views, stream_offset=40 + stream_offset)
    for c in cats:
        X[c] = X[c].astype(str)
    num = [c for c in X.columns if c not in cats]
    X[num] = X[num].astype(float).fillna(-999.0)
    return X, cats

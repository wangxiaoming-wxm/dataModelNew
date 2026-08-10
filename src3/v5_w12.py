"""V5 world w12: exact ``build_alt`` plus grafted main cond_r / ratio.

alt already clears the 0.694 gate.  Grafting main's two carrying signals
should not make it weaker, and the extra expression may decorrelate it
enough from the frozen cat_alt bag to help ``max``.

Diffs vs ``altboost_frame``:
  * keep build_alt + noise view + jitter unchanged (same rate formula);
  * add cond_r / ratio / log_* numerics from main's per-source median scale;
  * add a short list of cr*_src / r*_src / dfx crosses on a (7,13,25) grid
    that matches alt's bins.

Label-free.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features import (BIN_COLS, GRADE_MAP, DAYS_FIXED_EDGES, add_noise_view,
                      build_alt, fit_edges_alt, _derive, _qbins)
from jitter import add_jitter_views


def fit_edges_w12(df: pd.DataFrame) -> dict:
    edges = fit_edges_alt(df)
    scale = df.groupby("source")["condition"].median()
    der = _derive(df, scale)
    edges["__scale__"] = scale
    for n in (7, 13, 25):
        qs = np.linspace(0, 1, n + 1)[1:-1]
        edges[f"condr_{n}"] = np.quantile(der["cond_r"], qs)
        edges[f"ratio_{n}"] = np.quantile(der["ratio"], qs)
    return edges


def w12_frame(raw: pd.DataFrame, edges: dict, stream_offset: int, n_views: int = 3):
    X, cats = build_alt(raw, edges)
    der = _derive(raw, edges["__scale__"])
    cond_r, ratio = der["cond_r"], der["ratio"]
    days = pd.to_numeric(raw["days"])

    X["cond_r"] = cond_r
    X["log_cond_r"] = np.log(cond_r.clip(lower=1e-9))
    X["ratio"] = ratio
    X["log_ratio"] = np.log(ratio.clip(lower=1e-9))
    X["ratio_p75"] = days / cond_r.clip(lower=1e-9) ** 0.75
    X["days_fx"] = np.digitize(days.to_numpy(dtype=float), DAYS_FIXED_EDGES).astype(str)
    cats.append("days_fx")

    for n in (7, 13, 25):
        X[f"cr{n}"] = _qbins(cond_r, edges[f"condr_{n}"]).astype(str)
        X[f"rr{n}"] = _qbins(ratio, edges[f"ratio_{n}"]).astype(str)
        cats += [f"cr{n}", f"rr{n}"]

    def cross(name, *parts):
        s = X[parts[0]].astype(str)
        for p in parts[1:]:
            s = s + "|" + X[p].astype(str)
        X[name] = s
        cats.append(name)

    cross("G_cr7_src", "cr7", "source")
    cross("G_cr13_src", "cr13", "source")
    cross("G_cr25_src", "cr25", "source")
    cross("G_cr13_reg", "cr13", "region")
    cross("G_rr7_src", "rr7", "source")
    cross("G_rr13_src", "rr13", "source")
    cross("G_rr25_src", "rr25", "source")
    cross("G_rr13_reg", "rr13", "region")
    cross("G_dfx_src", "days_fx", "source")
    cross("G_dfx_cr13", "days_fx", "cr13")
    cross("G_dfx_rr13", "days_fx", "rr13")
    cross("G_d13_cr13", "d13", "cr13")

    add_noise_view(X, cats, raw)
    rk = raw.groupby("source")["condition"].rank(pct=True).fillna(0.5)
    add_jitter_views(X, cats, raw, rk, days,
                     n_views=n_views, n_bins=13, stream_offset=50 + stream_offset)
    for c in cats:
        X[c] = X[c].astype(str)
    num = [c for c in X.columns if c not in cats]
    X[num] = X[num].astype(float).fillna(-999.0)
    return X, cats

"""Target-free feature engineering for the vehicle-insurance claim task.

Data-generating structure recovered during EDA (see docs/DATA_STRUCTURE.md):

* ``source`` is the latent vehicle model.  ``V``, ``x19``, ``code``, ``t3`` and
  ``x0..x17`` are deterministic functions of it plus additive uniform
  anonymisation noise; ``cc`` and ``max_g`` are its centre plus uniform noise.
* ``livability`` is a deterministic function of ``region``.
* ``x20`` is an affine function of ``condition`` plus uniform noise.
* ``x18`` is unconditional noise.

Every one of those residuals scores inside the permutation band against the
label, so the informative columns are only:
``days, condition, region, source, age_range`` and the eight binary flags.
``month``/``version``/``grades`` are kept as low-weight extras.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BIN_COLS = ["t1", "t2", "r1", "r2", "c1", "c2", "w1", "w2"]
GRADE_MAP = {"s": 1, "ss": 2, "sss": 3}


def _qbins(values: pd.Series, edges: np.ndarray) -> pd.Series:
    return pd.Series(np.digitize(values.to_numpy(), edges), index=values.index)


def fit_edges(df: pd.DataFrame, ns=(5, 10, 20, 40)) -> dict:
    """Quantile cut points. Label-free, so computing them on all rows is safe."""
    edges = {}
    for n in ns:
        qs = np.linspace(0, 1, n + 1)[1:-1]
        edges[f"days_{n}"] = np.quantile(df["days"].dropna(), qs)
        edges[f"cond_{n}"] = np.quantile(df["condition"].dropna(), qs)
    return edges


def build(df: pd.DataFrame, edges: dict, level: str = "cross") -> tuple[pd.DataFrame, list[str]]:
    """Return (frame, categorical_column_names) for a given FE level."""
    out = pd.DataFrame(index=df.index)

    # ---- numeric core -------------------------------------------------
    days = pd.to_numeric(df["days"])
    cond = pd.to_numeric(df["condition"])
    out["days"] = days
    out["log_days"] = np.log1p(days.clip(lower=0))
    out["condition"] = cond
    out["log_condition"] = np.log1p(cond.clip(lower=0))
    out["condition_missing"] = cond.isna().astype(int)
    out["age_range"] = df["age_range"].astype(float)
    for c in BIN_COLS:
        out[c] = df[c].astype(int)
    out["bin_sum"] = df[BIN_COLS].sum(axis=1)
    out["days_over_age"] = days / df["age_range"].astype(float)
    out["cond_x_days"] = cond * days
    out["cond_over_days"] = cond / (days.abs() + 1.0)

    # ---- base categoricals --------------------------------------------
    cats: list[str] = []
    out["region"] = df["region"].astype(str)
    out["source"] = df["source"].astype(str)
    cats += ["region", "source"]

    if level == "core":
        return out, cats

    out["month"] = df["month"].astype(str)
    out["version"] = df["version"].astype(str)
    out["grade_ord"] = df["grades"].map(GRADE_MAP).astype(float)
    cats += ["month", "version"]

    # ---- discretised drivers ------------------------------------------
    for n in (5, 10, 20, 40):
        out[f"days_q{n}"] = _qbins(days, edges[f"days_{n}"]).astype(str)
        cats.append(f"days_q{n}")
    for n in (5, 10, 20):
        out[f"cond_q{n}"] = _qbins(cond.fillna(-1), edges[f"cond_{n}"]).astype(str)
        cats.append(f"cond_q{n}")
    out["age_cat"] = df["age_range"].astype(str)
    out["bin_pat"] = df[BIN_COLS].astype(str).agg("".join, axis=1)
    cats += ["age_cat", "bin_pat"]

    if level == "flat":
        return out, cats

    # ---- interaction categoricals (CatBoost turns these into ordered TS)
    def cross(name: str, *parts: str) -> None:
        out[name] = out[parts[0]].astype(str)
        for p in parts[1:]:
            out[name] = out[name] + "|" + out[p].astype(str)
        cats.append(name)

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

    return out, cats

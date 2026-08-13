"""vz21: honest multi-family ensemble on the business features only.

Design rules (a direct reaction to what the fp_v8 audit found):
  * no id-derived feature of any kind
  * no target encoding outside a model's own fold-internal machinery
  * no weight or direction is ever chosen using the labels of the rows it is
    scored on
  * bin edges are fitted on train+test features (no labels), which is
    transductive but label-free, and is what the previous pipeline did too

The feature set is deliberately *leaner* than vz19's ~121 features / 81
categorical crosses: with 1,496 positives, dozens of high-cardinality crosses
are a variance source, not a signal source.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BIN_COLS = ["t1", "t2", "r1", "r2", "c1", "c2", "w1", "w2"]
GRADE_MAP = {"s": 1, "ss": 2, "sss": 3}
XCOLS = [f"x{i}" for i in range(19)]
QUANTS = (5, 10, 20)


# ---------------------------------------------------------------- features
def fit_edges(df: pd.DataFrame) -> dict:
    """Label-free binning/normalisation statistics fitted on train+test."""
    cond = pd.to_numeric(df["condition"], errors="coerce")
    days = pd.to_numeric(df["days"], errors="coerce")
    scale = cond.groupby(df["source"]).median()
    cond_r = (cond / df["source"].map(scale)).fillna(1.0)
    ratio = days / cond_r.clip(lower=1e-9)
    edges = {"__scale__": scale}
    for n in QUANTS:
        qs = np.linspace(0, 1, n + 1)[1:-1]
        edges[f"d_{n}"] = np.quantile(days.dropna(), qs)
        edges[f"c_{n}"] = np.quantile(cond.dropna(), qs)
        edges[f"cr_{n}"] = np.quantile(cond_r, qs)
        edges[f"ra_{n}"] = np.quantile(ratio, qs)
    return edges


def build(df: pd.DataFrame, edges: dict, include_x: bool = False):
    """Return (feature frame, list of categorical column names)."""
    out = pd.DataFrame(index=df.index)
    days = pd.to_numeric(df["days"], errors="coerce")
    cond = pd.to_numeric(df["condition"], errors="coerce")
    scale = edges["__scale__"]
    cond_r = (cond / df["source"].map(scale)).fillna(1.0)
    rank = cond.groupby(df["source"]).rank(pct=True).fillna(0.5)
    ratio = days / cond_r.clip(lower=1e-9)
    rate = days * (1.0 - rank)

    # --- core numerics: the two engineered "worlds" (ratio and rate) merged,
    # since a single model with both beats two models with one each.
    out["days"] = days
    out["log_days"] = np.log1p(days.clip(lower=0))
    out["condition"] = cond
    out["cond_missing"] = cond.isna().astype(int)
    out["cond_r"] = cond_r
    out["log_cond_r"] = np.log(cond_r.clip(lower=1e-9))
    out["cond_rank"] = rank
    out["ratio"] = ratio
    out["log_ratio"] = np.log(ratio.clip(lower=1e-9))
    out["ratio_p75"] = days / cond_r.clip(lower=1e-9) ** 0.75
    out["rate"] = rate
    out["log_rate"] = np.log1p(rate.clip(lower=0))
    out["cond_x_days"] = cond * days
    out["cond_over_days"] = cond / (days.abs() + 1.0)

    age = df["age_range"].astype(float)
    out["age_range"] = age
    out["days_over_age"] = days / age
    out["rate_over_age"] = rate / age
    out["grade_ord"] = df["grades"].map(GRADE_MAP).astype(float)

    for c in ("V", "cc", "max_g", "livability", "x19", "x20"):
        out[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("V", "cc", "max_g"):
        out[f"{c}_over_cr"] = out[c] / cond_r.clip(lower=1e-9)
        out[f"{c}_over_days"] = out[c] / (days.abs() + 1.0)

    for c in BIN_COLS:
        out[c] = df[c].astype(int)
    out["bin_sum"] = out[BIN_COLS].sum(axis=1)

    # t3 = "<number><letter>": split it instead of using 163 opaque levels
    t3 = df["t3"].astype(str)
    out["t3_num"] = pd.to_numeric(t3.str.extract(r"^([0-9.]+)")[0], errors="coerce")
    # x0..x18 are deliberately excluded: honest cross-half puts them at
    # 0.5124 (real but tiny) and a paired CatBoost test costs -0.0041
    # fold-mean when they are added. They are diluting, not informative.
    if include_x:
        for c in XCOLS:
            out[c] = pd.to_numeric(df[c], errors="coerce")

    # --- categoricals (a small, deliberate set) ---
    cats: list[str] = []

    def put(name, values):
        out[name] = pd.Series(np.asarray(values, dtype=object), index=df.index).astype(str)
        cats.append(name)

    put("region", df["region"])
    put("source", df["source"])
    put("month", df["month"])
    put("version", df["version"])
    put("code", df["code"])
    put("grades_c", df["grades"])
    put("age_cat", df["age_range"])
    put("t3_suf", t3.str.extract(r"([A-Za-z]+)$")[0].fillna("NA"))
    put("t3_c", t3)
    put("x19_c", df["x19"])
    put("x20_c", df["x20"])
    put("lv_c", df["livability"])
    put("bin_pat", out[BIN_COLS].astype(str).agg("".join, axis=1))

    for n in QUANTS:
        put(f"d{n}", np.digitize(days.to_numpy(float), edges[f"d_{n}"]))
        put(f"c{n}", np.digitize(cond.fillna(-1).to_numpy(float), edges[f"c_{n}"]))
        put(f"cr{n}", np.digitize(cond_r.to_numpy(float), edges[f"cr_{n}"]))
        put(f"r{n}", np.digitize(ratio.to_numpy(float), edges[f"ra_{n}"]))

    # a short, pre-registered list of 2-way crosses -- the interactions the
    # marginal analysis actually supports (days x condition x region/source)
    def cross(name, a, b):
        put(name, out[a].astype(str) + "|" + out[b].astype(str))

    cross("rs", "region", "source")
    cross("d10_r", "d10", "region")
    cross("d10_s", "d10", "source")
    cross("c10_s", "c10", "source")
    cross("cr10_r", "cr10", "region")
    cross("d10_c10", "d10", "c10")
    cross("r10_s", "r10", "source")
    cross("region_age", "region", "age_cat")
    cross("source_age", "source", "age_cat")

    for c in ("region", "source", "bin_pat", "rs", "t3_c"):
        out[f"f_{c}"] = out[c].map(out[c].value_counts()).astype(float)

    return out, cats


def make_matrices(train_df: pd.DataFrame, test_df: pd.DataFrame, include_x: bool = False):
    """Build features once for train+test together (label-free)."""
    raw = pd.concat([train_df.drop(columns=["label"], errors="ignore"), test_df], ignore_index=True)
    edges = fit_edges(raw)
    X, cats = build(raw, edges, include_x=include_x)
    ntr = len(train_df)
    return X.iloc[:ntr].reset_index(drop=True), X.iloc[ntr:].reset_index(drop=True), cats


def as_category(X: pd.DataFrame, cats: list[str], levels: dict | None = None):
    """Copy with categorical dtype, using a shared level set across splits."""
    out = X.copy()
    lv = {}
    for c in cats:
        cat_type = pd.CategoricalDtype(levels[c]) if levels else pd.CategoricalDtype(sorted(X[c].unique()))
        out[c] = out[c].astype(cat_type)
        lv[c] = list(out[c].cat.categories)
    return out, lv


def as_ordinal(X: pd.DataFrame, cats: list[str], levels: dict):
    """Integer codes for models without native categorical support."""
    out = X.copy()
    for c in cats:
        out[c] = pd.Categorical(out[c], categories=levels[c]).codes.astype(np.int32)
    return out

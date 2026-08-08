"""Target-free feature engineering for the vehicle-insurance claim task.

Data-generating structure recovered during EDA (see docs/DATA_STRUCTURE.md):

* ``source`` is the latent vehicle model.  ``V``, ``x19``, ``code``, ``t3`` and
  ``x0..x17`` are deterministic functions of it plus additive uniform
  anonymisation noise; ``cc`` and ``max_g`` are its centre plus uniform noise.
* ``livability`` is a deterministic function of ``region``.
* ``x20`` is an affine function of ``condition`` plus uniform noise.
* ``x18`` is unconditional noise.

Every one of those residuals scores inside the permutation band against the
label, both on its own and jointly, so the informative columns are only
``days``, ``condition``, ``region``, ``source``, ``age_range`` and the eight
binary flags, with ``month`` / ``version`` / ``grades`` as weak extras.

``condition`` is only meaningful relative to the vehicle model it belongs to
(CAR_1 sits at a third of the scale of the others), and the resulting
``days / condition_ratio`` rate is the single strongest ranker in the data
(AUC 0.620 on its own, against 0.593 for raw ``days``).

Nothing here touches the label, so fitting the quantile cut points and the
per-source condition scale on train+test is transductive but leakage-free.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BIN_COLS = ["t1", "t2", "r1", "r2", "c1", "c2", "w1", "w2"]
GRADE_MAP = {"s": 1, "ss": 2, "sss": 3}
DAYS_FIXED_EDGES = np.array([700, 2500, 5000, 7000, 9000, 10000], dtype=float)
QUANTS = (5, 10, 20, 40)


def _qbins(values: pd.Series, edges: np.ndarray) -> np.ndarray:
    return np.digitize(values.to_numpy(dtype=float), edges)


def _derive(df: pd.DataFrame, scale: pd.Series) -> pd.DataFrame:
    cond = pd.to_numeric(df["condition"])
    days = pd.to_numeric(df["days"])
    sm = df["source"].map(scale).astype(float)
    cond_r = (cond / sm).fillna(1.0)
    return pd.DataFrame(
        {"cond_r": cond_r, "ratio": days / cond_r.clip(lower=1e-9)}, index=df.index
    )


def fit_edges(df: pd.DataFrame) -> dict:
    """Quantile cut points and the per-source condition scale. Label-free."""
    scale = df.groupby("source")["condition"].median()
    der = _derive(df, scale)
    edges: dict = {"__scale__": scale}
    for n in QUANTS:
        qs = np.linspace(0, 1, n + 1)[1:-1]
        edges[f"days_{n}"] = np.quantile(df["days"].dropna(), qs)
        edges[f"cond_{n}"] = np.quantile(df["condition"].dropna(), qs)
        edges[f"condr_{n}"] = np.quantile(der["cond_r"], qs)
        edges[f"ratio_{n}"] = np.quantile(der["ratio"], qs)
    return edges


def build(df: pd.DataFrame, edges: dict, level: str = "cross") -> tuple[pd.DataFrame, list[str]]:
    """Return ``(frame, categorical_columns)``.

    ``num``    numeric columns only.
    ``base``   + the raw and discretised categoricals.
    ``cross``  + curated pair crosses (the working set).
    ``cross2`` + joint days x condition cells crossed with the segment columns.
    """
    out = pd.DataFrame(index=df.index)
    days = pd.to_numeric(df["days"])
    cond = pd.to_numeric(df["condition"])
    der = _derive(df, edges["__scale__"])
    cond_r, ratio = der["cond_r"], der["ratio"]

    # ---- numeric --------------------------------------------------------
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
    if level == "num":
        return out, cats

    # ---- categoricals ---------------------------------------------------
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

    if level == "base":
        return out, cats

    def cross(name: str, *parts: str) -> None:
        s = out[parts[0]].astype(str)
        for p in parts[1:]:
            s = s + "|" + out[p].astype(str)
        out[name] = s
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
    cross("r10_reg", "ratio_q10", "region")
    cross("r10_src", "ratio_q10", "source")
    cross("r10_age", "ratio_q10", "age_cat")
    cross("r20_reg", "ratio_q20", "region")
    cross("r10_pat", "ratio_q10", "bin_pat")
    cross("cr10_reg", "condr_q10", "region")
    cross("cr10_age", "condr_q10", "age_cat")

    if level == "cross":
        return out, cats

    # condition x source is by far the strongest interaction; give the model
    # several independent discretisations of it rather than a single one.
    cross("c5_src", "cond_q5", "source")
    cross("c20_src", "cond_q20", "source")
    cross("cr5_src", "condr_q5", "source")
    cross("cr10_src", "condr_q10", "source")
    cross("cr20_src", "condr_q20", "source")
    cross("cr5_reg", "condr_q5", "region")
    cross("cr20_reg", "condr_q20", "region")
    cross("c5_reg", "cond_q5", "region")
    # joint days x condition cells, and the same cells inside each segment
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

    # label-free frequency encodings of the segment columns
    for c in ("region", "source", "bin_pat", "reg_src", "d10_reg", "c10_src", "month", "version"):
        out[f"freq_{c}"] = out[c].map(out[c].value_counts()).astype(float)
    return out, cats


def add_noise_view(out: pd.DataFrame, cats: list[str], df: pd.DataFrame) -> None:
    """Add the organisers' redundant/noisy encodings of source and condition.

    They carry no signal of their own (their within-group residuals score inside
    the permutation band) but they act as jittered re-encodings of the real
    ``source`` and ``condition`` interactions, which is a cheap source of
    averaging inside a single CatBoost model.
    """
    out["x19_cat"] = df["x19"].astype(str)
    out["x20_cat"] = df["x20"].astype(str)
    out["liv_cat"] = df["livability"].astype(str)
    out["t3_cat"] = df["t3"].astype(str)
    out["code_cat"] = df["code"].astype(str)
    cats += ["x19_cat", "x20_cat", "liv_cat", "t3_cat", "code_cat"]
    out["cc"] = df["cc"].astype(float)
    out["max_g"] = df["max_g"].astype(float)
    out["V"] = df["V"].astype(float)
    out["x18"] = df["x18"].astype(float)
    for name, parts in [
        ("x20_src", ("x20_cat", "source")),
        ("x20_reg", ("x20_cat", "region")),
        ("x20_age", ("x20_cat", "age_cat")),
        ("x19_liv", ("x19_cat", "liv_cat")),
        ("liv_age", ("liv_cat", "age_cat")),
        ("reg_liv", ("region", "liv_cat")),
        ("t3_days5", ("t3_cat", "days_q5")),
        ("src_x20_age", ("source", "x20_cat", "age_cat")),
        ("reg_x20_age", ("region", "x20_cat", "age_cat")),
        ("reg_src_x19", ("region", "source", "x19_cat")),
    ]:
        s = out[parts[0]].astype(str)
        for p in parts[1:]:
            s = s + "|" + out[p].astype(str)
        out[name] = s
        cats.append(name)

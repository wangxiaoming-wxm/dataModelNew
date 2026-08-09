"""B8 gapv3 categorical features (fold-local; no TE).

Target residual failure modes vs B7 max3:
- high condition / shorter days misses
- hard regions 9685/f09d/6645/fafc
- x20 / grades=s / t3_sfx=M slices
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from insurance_claim.b6_gap_features import (
    DAYS_FIXED_EDGES,
    _bin_codes,
    _days_fixed,
    _quantile_edges,
    _t3_sfx,
)

GAPV3_CAT_COLS = (
    "gv3_hard_region",
    "gv3_x20_bin",
    "gv3_x20_region",
    "gv3_x20_days5",
    "gv3_cond_hi_days_lo",
    "gv3_cond5_region",
    "gv3_cond5_hard",
    "gv3_daysfix_x20",
    "gv3_grades_t3",
    "gv3_grades_region",
    "gv3_t3sfx_region_days5",
    "gv3_age_cond5",
    "gv3_liv_days5",
    "gv3_ratio5_hard",
    "gv3_midcond_source",
)


def fit_gapv3_edges(X_tr: pd.DataFrame) -> dict[str, np.ndarray]:
    days = pd.to_numeric(X_tr["days"], errors="coerce")
    cond = pd.to_numeric(X_tr["condition"], errors="coerce")
    ratio = cond / (days.abs() + 1.0)
    x20 = pd.to_numeric(X_tr["x20"], errors="coerce")
    liv = pd.to_numeric(X_tr.get("livability"), errors="coerce")
    return {
        "days5": _quantile_edges(days, 5),
        "cond5": _quantile_edges(cond, 5),
        "ratio5": _quantile_edges(ratio, 5),
        "x20_5": _quantile_edges(x20, 5),
        "liv5": _quantile_edges(liv, 5),
        "cond_med": np.array([float(cond.median())], dtype=float),
        "days_med": np.array([float(days.median())], dtype=float),
    }


def add_gapv3_cats(frame: pd.DataFrame, edges: dict[str, np.ndarray]) -> pd.DataFrame:
    out = frame.copy()
    days = pd.to_numeric(out["days"], errors="coerce")
    cond = pd.to_numeric(out["condition"], errors="coerce")
    ratio = cond / (days.abs() + 1.0)
    x20 = pd.to_numeric(out["x20"], errors="coerce")
    liv = pd.to_numeric(out.get("livability"), errors="coerce")
    age = pd.to_numeric(out.get("age_range"), errors="coerce")

    days5 = _bin_codes(days, edges["days5"], "d5")
    cond5 = _bin_codes(cond, edges["cond5"], "c5")
    ratio5 = _bin_codes(ratio, edges["ratio5"], "r5")
    x20b = _bin_codes(x20, edges["x20_5"], "x20")
    liv5 = _bin_codes(liv, edges["liv5"], "liv")
    days_fixed = _days_fixed(days)

    region = out["region"].astype(str)
    source = out["source"].astype(str)
    grades = out["grades"].astype(str)
    t3_sfx = _t3_sfx(out["t3"]) if "t3" in out.columns else pd.Series("__NONE__", index=out.index)

    hard = region.isin(["9685", "f09d", "6645", "fafc"]).map({True: "hard", False: "other"})
    cond_hi = cond >= float(edges["cond_med"][0])
    days_lo = days <= float(edges["days_med"][0])
    regime = np.where(cond_hi & days_lo, "condhi_dayslo", np.where(cond_hi, "condhi", "other"))

    midcond = cond5.isin([f"c5_{i}" for i in (2, 3)])  # central bins approx
    midcond_s = np.where(midcond, "mid", "edge")

    out["gv3_hard_region"] = hard.astype(str)
    out["gv3_x20_bin"] = x20b
    out["gv3_x20_region"] = (x20b + "|" + region).astype(str)
    out["gv3_x20_days5"] = (x20b + "|" + days5).astype(str)
    out["gv3_cond_hi_days_lo"] = pd.Series(regime, index=out.index).astype(str)
    out["gv3_cond5_region"] = (cond5 + "|" + region).astype(str)
    out["gv3_cond5_hard"] = (cond5 + "|" + hard.astype(str)).astype(str)
    out["gv3_daysfix_x20"] = (days_fixed.astype(str) + "|" + x20b).astype(str)
    out["gv3_grades_t3"] = (grades + "|" + t3_sfx).astype(str)
    out["gv3_grades_region"] = (grades + "|" + region).astype(str)
    out["gv3_t3sfx_region_days5"] = (t3_sfx + "|" + region + "|" + days5).astype(str)
    out["gv3_age_cond5"] = (age.fillna(-1).astype(int).astype(str) + "|" + cond5).astype(str)
    out["gv3_liv_days5"] = (liv5 + "|" + days5).astype(str)
    out["gv3_ratio5_hard"] = (ratio5 + "|" + hard.astype(str)).astype(str)
    out["gv3_midcond_source"] = (
        pd.Series(midcond_s, index=out.index).astype(str) + "|" + source
    ).astype(str)
    return out

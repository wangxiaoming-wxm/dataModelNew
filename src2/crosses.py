"""Configurable categorical cross generation over the informative columns only."""
from __future__ import annotations

import itertools
import pandas as pd

# Ordered by how much marginal signal the column carries.
PAIR_BASE = [
    "ratio_q10", "days_q10", "condr_q10", "region", "source",
    "age_cat", "bin_pat", "ratio_q5", "days_q5", "condr_q5",
    "cond_q10", "days_q20", "ratio_q20", "month", "version", "grades_c",
]

TRIPLES = [
    ("days_q5", "region", "source"),
    ("ratio_q5", "region", "source"),
    ("ratio_q5", "region", "age_cat"),
    ("days_q5", "region", "age_cat"),
    ("condr_q5", "region", "source"),
    ("ratio_q5", "source", "age_cat"),
    ("days_q5", "source", "bin_pat"),
    ("ratio_q5", "region", "bin_pat"),
]


def add_crosses(out: pd.DataFrame, cats: list[str], n_pair_base: int, n_triples: int) -> list[str]:
    base = [c for c in PAIR_BASE[:n_pair_base] if c in out.columns]
    new: list[str] = []
    for a, b in itertools.combinations(base, 2):
        name = f"X_{a}_{b}"
        out[name] = out[a].astype(str) + "|" + out[b].astype(str)
        new.append(name)
    for parts in TRIPLES[:n_triples]:
        if not all(p in out.columns for p in parts):
            continue
        name = "X3_" + "_".join(parts)
        s = out[parts[0]].astype(str)
        for p in parts[1:]:
            s = s + "|" + out[p].astype(str)
        out[name] = s
        new.append(name)
    cats.extend(new)
    return new

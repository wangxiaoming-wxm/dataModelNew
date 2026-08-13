#!/usr/bin/env python3
"""E7: how much of a leaderboard gap is actually resolvable?

The test set is 6,398 rows at a ~10% positive rate, i.e. ~640 positives. That
puts a hard noise floor under any AUC comparison. Two estimates:

  * Hanley-McNeil analytic standard error of a single AUC
  * bootstrap SE of the *difference* between two correlated submissions,
    which is the quantity that actually decides who ranks above whom
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/workspace/vz20/audit")
from common import ART  # noqa: E402


def hanley_mcneil(a, n1, n0):
    q1 = a / (2 - a)
    q2 = 2 * a * a / (1 + a)
    var = (a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a)) / (n1 * n0)
    return float(np.sqrt(var))


def main():
    n = 6398
    prior = 0.1002
    n1 = int(round(n * prior))
    n0 = n - n1
    out = {"n_test": n, "assumed_pos_rate": prior, "n_pos": n1, "n_neg": n0}

    out["se_single_auc"] = {f"{a:.3f}": hanley_mcneil(a, n1, n0) for a in (0.70, 0.715, 0.72, 0.749)}
    print("Hanley-McNeil SE of one AUC on this test set:")
    for k, v in out["se_single_auc"].items():
        print(f"  AUC {k}: SE = {v:.4f}")

    gaps = {
        "W62(0.71503) -> top3(0.72)": 0.72 - 0.71503,
        "W62(0.71503) -> champion(0.749)": 0.749 - 0.71503,
        "vz19(0.71298) -> W62(0.71503)": 0.71503 - 0.71298,
    }
    se = out["se_single_auc"]["0.715"]
    out["gaps_in_se_units"] = {k: {"gap": v, "in_SE": v / se} for k, v in gaps.items()}
    print(f"\nGaps expressed in units of that SE ({se:.4f}):")
    for k, v in out["gaps_in_se_units"].items():
        print(f"  {k:<36} {v['gap']:+.5f} = {v['in_SE']:.2f} SE")

    # SE of the *difference* between two submissions of a given rank
    # correlation, by simulation on a synthetic test set of the same size.
    rng = np.random.default_rng(0)
    sims = {}
    for rho in (0.90, 0.95, 0.99):
        diffs = []
        for _ in range(300):
            y = (rng.random(n) < prior).astype(int)
            base = rng.standard_normal(n) + 0.75 * y
            other = rho * base + np.sqrt(1 - rho**2) * (rng.standard_normal(n) + 0.75 * y)
            from common import fast_auc

            diffs.append(fast_auc(y, base) - fast_auc(y, other))
        sims[str(rho)] = float(np.std(diffs, ddof=1))
    out["se_of_difference_by_correlation"] = sims
    print("\nSE of the AUC *difference* between two submissions (simulated):")
    for k, v in sims.items():
        print(f"  correlation {k}: SE(diff) = {v:.4f}")

    out["interpretation"] = (
        "A single submission's AUC on this leaderboard carries ~0.012 of pure sampling noise, "
        "so the 0.005 gap from W62 to the top-3 line is well under half a standard error and the "
        "0.034 gap to the champion is under three. Two highly correlated submissions can still be "
        "compared far more precisely (SE of the difference ~0.002-0.005), which is why a genuine "
        "local improvement is worth chasing and a leaderboard-rank-chasing gamble is not."
    )
    (ART / "e6_lb_noise.json").write_text(json.dumps(out, indent=2) + "\n")
    print("\n" + out["interpretation"])


if __name__ == "__main__":
    main()

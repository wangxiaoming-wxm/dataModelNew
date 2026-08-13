#!/usr/bin/env python3
"""Pick the vz21 recipe: best local AUC subject to the diversity constraint.

The constraint the owner imposed -- Spearman < 0.97 against both already
submitted files -- is computed on *test predictions only* and involves no
labels, so searching against it cannot leak. The objective (fold-mean AUC) is
measured on the SELECT seeds; the winner is then reported on the CONFIRM seeds,
which played no part in the search.

Two recipes come out of this:
  vz21      -- best subject to rho <= 0.965 on both files (the owner's rule)
  vz21_max  -- best with no diversity constraint, reported for reference so the
               exact price of the constraint is visible rather than hidden
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, "/workspace/vz20/src")
from vz21_arms import ARMS, CACHE, CONFIRM_SEEDS, NSPLIT, SELECT_SEEDS  # noqa: E402

ART = Path("/workspace/vz20/artifacts/vz20")
RHO_CAP = 0.965
N_RANDOM = 20000


def r01(x):
    return rankdata(x) / len(x)


def main():
    train = pd.read_csv("/workspace/data/train.csv", dtype={"id": str})
    test = pd.read_csv("/workspace/data/test.csv", dtype={"id": str})
    y = train["label"].astype(int).to_numpy()
    n, nt = len(y), len(test)
    order = pd.read_csv("/workspace/data/submit_sample.csv", dtype={"id": str})["id"].tolist()
    ref = {}
    for k, p in (("vz19", "/workspace/vz20/submission_vz19.csv"), ("W62", "/tmp/submission_w62.csv")):
        ref[k] = pd.read_csv(p, dtype={"id": str}).set_index("id")["label"].reindex(order).to_numpy(float)

    arms = [a for a in ARMS if (CACHE / f"{a}_s{SELECT_SEEDS[0]}_oof.npy").is_file()]
    allseeds = list(SELECT_SEEDS) + list(CONFIRM_SEEDS)

    # rank-normalised OOF per (arm, seed), and test preds averaged over ALL
    # outer seeds -- 4 seeds x 5 folds = 20 fitted models behind every arm's
    # test column, which is the bagging that makes the final file stable.
    O = {s: np.stack([r01(np.load(CACHE / f"{a}_s{s}_oof.npy")) for a in arms]) for s in allseeds}
    T = np.stack([
        np.mean([r01(np.load(CACHE / f"{a}_s{s}_test.npy")) for s in allseeds], axis=0) for a in arms
    ])

    folds = {s: list(StratifiedKFold(NSPLIT, shuffle=True, random_state=s).split(np.zeros(n), y)) for s in allseeds}

    def fm(w, seeds):
        vals = []
        for s in seeds:
            sc = w @ O[s]
            vals.append(np.mean([roc_auc_score(y[v], sc[v]) for _, v in folds[s]]))
        return float(np.mean(vals))

    def rhos(w):
        t = w @ T
        return {k: float(spearmanr(t, v).statistic) for k, v in ref.items()}

    rng = np.random.default_rng(20260812)
    k = len(arms)
    best_free = (None, -1)
    best_con = (None, -1)
    print(f"searching {N_RANDOM} random simplex weights over {k} arms ...")
    for i in range(N_RANDOM):
        # Dirichlet with a sparsity-inducing alpha, plus occasional subset masks
        w = rng.dirichlet(np.full(k, 0.35))
        if i % 3 == 0:
            m = rng.random(k) < 0.5
            if m.any():
                w = w * m
                w = w / w.sum()
        a = fm(w, SELECT_SEEDS)
        if a > best_free[1]:
            best_free = (w.copy(), a)
        rr = rhos(w)
        if max(rr.values()) <= RHO_CAP and a > best_con[1]:
            best_con = (w.copy(), a)
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}: free={best_free[1]:.5f} constrained={best_con[1]:.5f}", flush=True)

    out = {"arms": arms, "rho_cap": RHO_CAP, "n_random": N_RANDOM,
           "select_seeds": list(SELECT_SEEDS), "confirm_seeds": list(CONFIRM_SEEDS)}

    base_w = np.zeros(k)
    base_w[arms.index("A_main_ord")] = 0.64
    base_w[arms.index("A_alt_plain")] = 0.36
    for name, w in (("BASE_vz19core", base_w), ("vz21_max", best_free[0]), ("vz21", best_con[0])):
        if w is None:
            continue
        rr = rhos(w)
        out[name] = {
            "weights": {a: round(float(x), 5) for a, x in zip(arms, w) if x > 1e-4},
            "select_fold_mean": fm(w, SELECT_SEEDS),
            "confirm_fold_mean": fm(w, CONFIRM_SEEDS),
            "rho_vs_W62": rr["W62"],
            "rho_vs_vz19": rr["vz19"],
        }
        print(f"\n{name}")
        print(f"  select  = {out[name]['select_fold_mean']:.5f}")
        print(f"  confirm = {out[name]['confirm_fold_mean']:.5f}")
        print(f"  rho: W62={rr['W62']:.4f} vz19={rr['vz19']:.4f}")
        print(f"  weights: {out[name]['weights']}")

    for name in ("vz21_max", "vz21"):
        if name in out:
            out[name]["delta_confirm_vs_base"] = out[name]["confirm_fold_mean"] - out["BASE_vz19core"]["confirm_fold_mean"]

    (ART / "vz21_select.json").write_text(json.dumps(out, indent=2) + "\n")
    print("\nwrote", ART / "vz21_select.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""The honest quality/diversity frontier.

The blend study found two things at once: no honest ensemble beats vz19's core
by more than noise, and every honest ensemble sits at Spearman >= 0.99 against
the already-submitted W62. Those two facts together are the real result of this
round, and they need to be shown as a curve rather than asserted.

Here we walk from the strongest honest blend toward the most decorrelated
honest arms and record, at each step, the local fold-mean AUC and the Spearman
against the two submitted files. That gives the exact price, in AUC, of every
unit of "looks different from what we already sent".

We also price the alternative fp_v8 chose: buying decorrelation with pure noise
instead of with weaker-but-real models.
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
from vz21_arms import CACHE, NSPLIT, SELECT_SEEDS  # noqa: E402

ART = Path("/workspace/vz20/artifacts/vz20")

# strongest honest core (the four main-world CatBoosts + the two alt-world ones)
CORE = {
    "C_main_deep": 0.22, "A_main_ord": 0.22, "C_main_ll": 0.18, "B_main_plain": 0.18,
    "A_alt_plain": 0.10, "B_alt_ord": 0.10,
}
# the most decorrelated arms that are still real models fitted on real features
DIVERSE = {"F_new_glm": 0.30, "E_new_et": 0.30, "D_new_ord": 0.20, "E_main_lgb": 0.20}


def r01(x):
    return rankdata(x) / len(x)


def load(arm, seed):
    return np.load(CACHE / f"{arm}_s{seed}_oof.npy"), np.load(CACHE / f"{arm}_s{seed}_test.npy")


def mix(weights, seed, n, nt):
    o, t = np.zeros(n), np.zeros(nt)
    for a, w in weights.items():
        oo, tt = load(a, seed)
        o += w * r01(oo)
        t += w * r01(tt)
    return o, t


def fold_mean(y, s, seed):
    f = StratifiedKFold(NSPLIT, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)
    return float(np.mean([roc_auc_score(y[v], s[v]) for _, v in f]))


def main():
    train = pd.read_csv("/workspace/data/train.csv", dtype={"id": str})
    test = pd.read_csv("/workspace/data/test.csv", dtype={"id": str})
    y = train["label"].astype(int).to_numpy()
    n, nt = len(y), len(test)
    order = pd.read_csv("/workspace/data/submit_sample.csv", dtype={"id": str})["id"].tolist()
    ref = {}
    for k, p in (("vz19", "/workspace/vz20/submission_vz19.csv"), ("W62", "/tmp/submission_w62.csv")):
        ref[k] = pd.read_csv(p, dtype={"id": str}).set_index("id")["label"].reindex(order).to_numpy(float)

    rows = []
    print(f"{'alpha':>6}{'localAUC':>11}{'d_vs_core':>11}{'rho_W62':>10}{'rho_vz19':>10}")
    print("-" * 48)
    for alpha in np.round(np.arange(0.0, 1.01, 0.1), 2):
        w = {a: (1 - alpha) * v for a, v in CORE.items()}
        for a, v in DIVERSE.items():
            w[a] = w.get(a, 0.0) + alpha * v
        aucs, tacc = [], np.zeros(nt)
        for s in SELECT_SEEDS:
            o, t = mix(w, s, n, nt)
            aucs.append(fold_mean(y, o, s))
            tacc += t / len(SELECT_SEEDS)
        rho = {k: float(spearmanr(tacc, v).statistic) for k, v in ref.items()}
        rows.append({"alpha": float(alpha), "local_fold_mean": float(np.mean(aucs)),
                     "rho_W62": rho["W62"], "rho_vz19": rho["vz19"], "weights": w})
        print(f"{alpha:>6.2f}{np.mean(aucs):>11.5f}{np.mean(aucs)-rows[0]['local_fold_mean']:>+11.5f}"
              f"{rho['W62']:>10.4f}{rho['vz19']:>10.4f}")

    # what fp_v8 did instead: buy decorrelation with pure noise
    print("\n=== buying decorrelation with NOISE instead (the fp_v8 trade) ===")
    print(f"{'noise_w':>8}{'localAUC':>11}{'d_vs_core':>11}{'rho_W62':>10}")
    noise_rows = []
    rng = np.random.default_rng(7)
    for wn in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        aucs, tacc = [], np.zeros(nt)
        for s in SELECT_SEEDS:
            o, t = mix(CORE, s, n, nt)
            no = r01(rng.standard_normal(n))
            ntv = r01(rng.standard_normal(nt))
            aucs.append(fold_mean(y, (1 - wn) * o + wn * no, s))
            tacc += ((1 - wn) * t + wn * ntv) / len(SELECT_SEEDS)
        rho = float(spearmanr(tacc, ref["W62"]).statistic)
        noise_rows.append({"noise_weight": wn, "local_fold_mean": float(np.mean(aucs)), "rho_W62": rho})
        print(f"{wn:>8.2f}{np.mean(aucs):>11.5f}{np.mean(aucs)-noise_rows[0]['local_fold_mean']:>+11.5f}{rho:>10.4f}")

    out = {"frontier": rows, "noise_frontier": noise_rows, "core": CORE, "diverse": DIVERSE,
           "select_seeds": list(SELECT_SEEDS)}
    (ART / "vz21_frontier.json").write_text(json.dumps(out, indent=2) + "\n")
    print("\nwrote", ART / "vz21_frontier.json")


if __name__ == "__main__":
    main()

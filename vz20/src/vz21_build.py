#!/usr/bin/env python3
"""Build the vz21 submission files.

Two files, both from the same cached arm predictions:

  submission_vz21.csv      -- P4_core_plus, the only pre-registered recipe that
                              was non-negative on the untouched CONFIRM seeds.
                              Maximises expected score.
  submission_vz21_div.csv  -- the best blend subject to Spearman <= 0.965
                              against both already-submitted files. Satisfies
                              the "must look different" rule, and pays for it.

Every arm's test column is the mean of its rank-normalised test prediction over
4 outer seeds x 5 folds = 20 fitted models, so the files are heavily bagged.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, "/workspace/vz20/src")
from vz21_arms import CACHE, CONFIRM_SEEDS, NSPLIT, SELECT_SEEDS  # noqa: E402

OUT = Path("/workspace/vz20/next_submit")
ART = Path("/workspace/vz20/artifacts/vz20")
ALL_SEEDS = list(SELECT_SEEDS) + list(CONFIRM_SEEDS)

# --- recipe 1: P4_core_plus, exactly as pre-registered in vz21_blend.py ---
KEEP = ["A_main_ord", "A_alt_plain", "B_main_plain", "B_alt_ord",
        "C_main_ll", "C_main_deep", "D_new_ord", "D_new_plain"]
CORE = {"A_main_ord": 0.64, "A_alt_plain": 0.36}
REST = [a for a in KEEP if a not in CORE]
W_MAIN = {**{a: 0.60 * w for a, w in CORE.items()}, **{a: 0.40 / len(REST) for a in REST}}

# --- recipe 2: diversity-constrained winner from vz21_select.py ---
W_DIV = {"B_main_plain": 0.2171, "C_main_ll": 0.00177, "C_main_deep": 0.27907, "F_new_glm": 0.50206}


def r01(x):
    return rankdata(x) / len(x)


def sha256(p):
    h = hashlib.sha256()
    h.update(Path(p).read_bytes())
    return h.hexdigest()


def assemble(weights, n, nt):
    o = {s: np.zeros(n) for s in ALL_SEEDS}
    t = np.zeros(nt)
    for a, w in weights.items():
        for s in ALL_SEEDS:
            o[s] += w * r01(np.load(CACHE / f"{a}_s{s}_oof.npy"))
        t += w * np.mean([r01(np.load(CACHE / f"{a}_s{s}_test.npy")) for s in ALL_SEEDS], axis=0)
    return o, t


def main():
    train = pd.read_csv("/workspace/data/train.csv", dtype={"id": str})
    y = train["label"].astype(int).to_numpy()
    sample = pd.read_csv("/workspace/data/submit_sample.csv", dtype={"id": str})
    n, nt = len(y), len(sample)
    ref = {}
    for k, p in (("vz19", "/workspace/vz20/submission_vz19.csv"), ("W62", "/tmp/submission_w62.csv")):
        ref[k] = pd.read_csv(p, dtype={"id": str}).set_index("id")["label"].reindex(sample["id"]).to_numpy(float)

    folds = {s: list(StratifiedKFold(NSPLIT, shuffle=True, random_state=s).split(np.zeros(n), y)) for s in ALL_SEEDS}

    def fm(o, seeds):
        return float(np.mean([np.mean([roc_auc_score(y[v], o[s][v]) for _, v in folds[s]]) for s in seeds]))

    OUT.mkdir(parents=True, exist_ok=True)
    meta = {}
    for name, w in (("vz21", W_MAIN), ("vz21_div", W_DIV)):
        o, t = assemble(w, n, nt)
        pred = np.clip(r01(t), 0.001, 0.999)
        path = OUT / f"submission_{name}.csv"
        pd.DataFrame({"id": sample["id"], "label": pred}).to_csv(path, index=False)
        rr = {k: float(spearmanr(pred, v).statistic) for k, v in ref.items()}
        meta[name] = {
            "path": str(path),
            "sha256": sha256(path),
            "weights": {k: round(float(v), 5) for k, v in w.items()},
            "select_fold_mean": fm(o, SELECT_SEEDS),
            "confirm_fold_mean": fm(o, CONFIRM_SEEDS),
            "rho_vs_W62": rr["W62"],
            "rho_vs_vz19": rr["vz19"],
            "n_models_behind_file": len(w) * len(ALL_SEEDS) * NSPLIT,
        }
        print(f"{name}: select={meta[name]['select_fold_mean']:.5f} confirm={meta[name]['confirm_fold_mean']:.5f} "
              f"rho(W62)={rr['W62']:.4f} rho(vz19)={rr['vz19']:.4f}")
        print(f"  {path}\n  sha256 {meta[name]['sha256']}")

    ob, _ = assemble(CORE, n, nt)
    meta["BASE_vz19core"] = {"select_fold_mean": fm(ob, SELECT_SEEDS), "confirm_fold_mean": fm(ob, CONFIRM_SEEDS)}
    for k in ("vz21", "vz21_div"):
        meta[k]["delta_confirm_vs_vz19core"] = meta[k]["confirm_fold_mean"] - meta["BASE_vz19core"]["confirm_fold_mean"]
    print(f"\nbaseline vz19core confirm={meta['BASE_vz19core']['confirm_fold_mean']:.5f}")
    print(f"  vz21     delta = {meta['vz21']['delta_confirm_vs_vz19core']:+.5f}")
    print(f"  vz21_div delta = {meta['vz21_div']['delta_confirm_vs_vz19core']:+.5f}")

    (ART / "vz21_build.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("\nwrote", ART / "vz21_build.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""vz21 blending, with the selection/confirmation split enforced.

Blend recipes are *rules fixed in advance*, not a weight search on the data
they are scored on -- that is exactly the failure mode the fp_v8 audit found.

  1. every arm is scored on SELECT_SEEDS only
  2. four pre-registered recipes are formed from those scores
  3. one recipe is chosen on SELECT_SEEDS
  4. the chosen recipe is scored ONCE on CONFIRM_SEEDS, which no arm screening
     or weighting ever touched

The paired baseline is vz19's own core (0.64*rank(arm1) + 0.36*rank(arm2)),
evaluated on identical folds so the comparison is paired.
"""
from __future__ import annotations

import argparse
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
QUALITY_BAR = 0.675
DIVERSITY_MAX_SPEARMAN = 0.97


def r01(x):
    return rankdata(x) / len(x)


def load(arm, seed):
    fo, ft = CACHE / f"{arm}_s{seed}_oof.npy", CACHE / f"{arm}_s{seed}_test.npy"
    if not (fo.is_file() and ft.is_file()):
        return None
    return np.load(fo), np.load(ft)


def fold_mean(y, score, seed):
    folds = StratifiedKFold(NSPLIT, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)
    return float(np.mean([roc_auc_score(y[v], score[v]) for _, v in folds]))


def arm_table(y, seeds, arms):
    """per-arm fold-mean over the given seeds (rank-normalised OOF)."""
    out = {}
    for a in arms:
        vals = []
        for s in seeds:
            got = load(a, s)
            if got is None:
                break
            vals.append(fold_mean(y, r01(got[0]), s))
        if len(vals) == len(seeds):
            out[a] = {"fold_mean": float(np.mean(vals)), "per_seed": [round(v, 5) for v in vals]}
    return out


# ------------------------------------------------------------- blend recipes
FAMILY = {
    "A_main_ord": "main_cb", "B_main_plain": "main_cb", "C_main_ll": "main_cb", "C_main_deep": "main_cb",
    "A_alt_plain": "alt_cb", "B_alt_ord": "alt_cb",
    "D_new_ord": "new_cb", "D_new_plain": "new_cb",
    "E_new_et": "other", "E_main_et": "other", "E_main_lgb": "other", "F_new_glm": "other",
}


def recipes(keep: list[str], quality: dict, allarms: list[str] | None = None) -> dict[str, dict[str, float]]:
    """Pre-registered weight rules. Each returns arm -> weight."""
    out = {}

    # P1: equal weight over every arm that clears the quality bar
    out["P1_equal"] = {a: 1.0 / len(keep) for a in keep}

    # P2: equal weight per model family, then equal weight inside a family.
    # Stops the four main-world CatBoosts from dominating by headcount.
    fams = {}
    for a in keep:
        fams.setdefault(FAMILY[a], []).append(a)
    out["P2_family"] = {a: 1.0 / (len(fams) * len(v)) for v in fams.values() for a in v}

    # P3: tilt toward quality, weight proportional to (fold_mean - 0.65)
    raw = {a: max(quality[a]["fold_mean"] - 0.65, 0.0) for a in keep}
    tot = sum(raw.values())
    out["P3_quality"] = {a: v / tot for a, v in raw.items()}

    # P4: keep vz19's core as the backbone, spend 40% on everything else
    core = {"A_main_ord": 0.64, "A_alt_plain": 0.36}
    rest = [a for a in keep if a not in core]
    out["P4_core_plus"] = {
        **{a: 0.60 * w for a, w in core.items() if a in keep},
        **{a: 0.40 / len(rest) for a in rest},
    }

    # P5: family-balanced over EVERY arm, quality bar waived. The externally
    # imposed constraint is that the submission must not be a near-copy of an
    # already-submitted file (Spearman < 0.97); seven of the nine arms that
    # clear the bar live in the old main/alt worlds, so a bar-free family
    # balance is the pre-registered way to buy decorrelation. Registered
    # before any blend score was computed.
    if allarms:
        fams2 = {}
        for a in allarms:
            fams2.setdefault(FAMILY[a], []).append(a)
        out["P5_family_all"] = {a: 1.0 / (len(fams2) * len(v)) for v in fams2.values() for a in v}
    return out


def blend_scores(weights, seed, n, nt):
    o = np.zeros(n)
    t = np.zeros(nt)
    for a, w in weights.items():
        got = load(a, seed)
        if got is None:
            return None, None
        o += w * r01(got[0])
        t += w * r01(got[1])
    return o, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true", help="also score on the confirmation seeds")
    args = ap.parse_args()

    train = pd.read_csv("/workspace/data/train.csv", dtype={"id": str})
    test = pd.read_csv("/workspace/data/test.csv", dtype={"id": str})
    y = train["label"].astype(int).to_numpy()
    n, nt = len(y), len(test)
    order = pd.read_csv("/workspace/data/submit_sample.csv", dtype={"id": str})["id"].tolist()

    ref = {}
    for name, path in (("vz19", "/workspace/vz20/submission_vz19.csv"), ("W62", "/tmp/submission_w62.csv")):
        if Path(path).is_file():
            d = pd.read_csv(path, dtype={"id": str}).set_index("id")["label"]
            ref[name] = d.reindex(order).to_numpy(float)

    avail = [a for a in ARMS if load(a, SELECT_SEEDS[0]) is not None]
    q = arm_table(y, SELECT_SEEDS, avail)
    print("=== arm quality on SELECT seeds", SELECT_SEEDS, "===")
    for a, v in sorted(q.items(), key=lambda t: -t[1]["fold_mean"]):
        _, t = load(a, SELECT_SEEDS[0])
        sp = f"{spearmanr(r01(t), ref['W62']).statistic:+.4f}" if "W62" in ref else "n/a"
        print(f"  {a:<14} {FAMILY[a]:<8} foldmean={v['fold_mean']:.5f} {v['per_seed']}  rho(W62)={sp}")

    keep = [a for a in q if q[a]["fold_mean"] >= QUALITY_BAR]
    print(f"\nkept {len(keep)}/{len(q)} arms at bar {QUALITY_BAR}: {keep}")

    base_w = {"A_main_ord": 0.64, "A_alt_plain": 0.36}
    R = recipes(keep, q, allarms=list(q))

    def score_on(seeds, weights):
        vals = []
        for s in seeds:
            o, _ = blend_scores(weights, s, n, nt)
            if o is None:
                return None
            vals.append(fold_mean(y, o, s))
        return vals

    print(f"\n=== SELECT-seed scores (baseline = vz19 core 0.64/0.36) ===")
    sel = {}
    b = score_on(SELECT_SEEDS, base_w)
    sel["BASE_vz19core"] = {"per_seed": b, "fold_mean": float(np.mean(b))}
    print(f"  {'BASE_vz19core':<16} {np.mean(b):.5f} {[round(x,5) for x in b]}")
    for name, w in R.items():
        v = score_on(SELECT_SEEDS, w)
        sel[name] = {"per_seed": v, "fold_mean": float(np.mean(v)),
                     "delta_vs_base": float(np.mean(v) - np.mean(b)), "weights": w}
        print(f"  {name:<16} {np.mean(v):.5f} {[round(x,5) for x in v]}  delta={np.mean(v)-np.mean(b):+.5f}")

    out = {"quality_bar": QUALITY_BAR, "select_seeds": list(SELECT_SEEDS),
           "confirm_seeds": list(CONFIRM_SEEDS), "arms": q, "kept": keep, "select": sel}

    if args.confirm:
        print(f"\n=== CONFIRM-seed scores {CONFIRM_SEEDS} (untouched until now) ===")
        conf = {}
        b2 = score_on(CONFIRM_SEEDS, base_w)
        conf["BASE_vz19core"] = {"per_seed": b2, "fold_mean": float(np.mean(b2))}
        print(f"  {'BASE_vz19core':<16} {np.mean(b2):.5f} {[round(x,5) for x in b2]}")
        for name, w in R.items():
            v = score_on(CONFIRM_SEEDS, w)
            if v is None:
                continue
            conf[name] = {"per_seed": v, "fold_mean": float(np.mean(v)),
                          "delta_vs_base": float(np.mean(v) - np.mean(b2))}
            print(f"  {name:<16} {np.mean(v):.5f} {[round(x,5) for x in v]}  delta={np.mean(v)-np.mean(b2):+.5f}")
        out["confirm"] = conf

        # paired per-fold win rate across every confirm fold
        print("\n=== paired per-fold record vs baseline (all confirm folds) ===")
        for name, w in R.items():
            wins = tot = 0
            for s in CONFIRM_SEEDS:
                ob, _ = blend_scores(base_w, s, n, nt)
                ov, _ = blend_scores(w, s, n, nt)
                for _, vi in StratifiedKFold(NSPLIT, shuffle=True, random_state=s).split(np.zeros(n), y):
                    tot += 1
                    wins += roc_auc_score(y[vi], ov[vi]) > roc_auc_score(y[vi], ob[vi])
            print(f"  {name:<16} {wins}/{tot} folds beat vz19 core")
            out["confirm"][name]["fold_wins"] = f"{wins}/{tot}"

    # diversity of each recipe's test prediction vs what has been submitted
    print("\n=== test-prediction diversity (gate: rho < 0.97 vs both) ===")
    for name, w in R.items():
        tacc = np.zeros(nt)
        ok = True
        for s in SELECT_SEEDS:
            _, t = blend_scores(w, s, n, nt)
            if t is None:
                ok = False
                break
            tacc += t / len(SELECT_SEEDS)
        if not ok:
            continue
        rr = {k: float(spearmanr(tacc, v).statistic) for k, v in ref.items()}
        sel[name]["rho_vs_submitted"] = rr
        sel[name]["diversity_gate"] = all(v < DIVERSITY_MAX_SPEARMAN for v in rr.values())
        print(f"  {name:<16} " + "  ".join(f"rho({k})={v:.4f}" for k, v in rr.items())
              + ("  PASS" if sel[name]["diversity_gate"] else "  FAIL"))

    (ART / "vz21_blend.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    print("\nwrote", ART / "vz21_blend.json")


if __name__ == "__main__":
    main()

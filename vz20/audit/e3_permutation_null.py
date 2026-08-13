#!/usr/bin/env python3
"""E3: permutation null for the honest cross-half transfer of id components.

E2 showed the honest cross-half AUC of every id family sits at ~0.5 and that
the fp_v8 "flip by full-label AUC" step is what manufactures the signal. This
script makes that quantitative:

  * "honest" sign estimator upgraded to inner-OOF inside the fitting half
    (the strongest legitimate version of the fp_v8 idea: direction is learned,
    but only from data the evaluation half never sees).
  * a null distribution from N random permutations of the id column, so the
    real-id result gets a z-score instead of an eyeball.

If the real-id z-score is not clearly positive, the id axis carries nothing
that can transfer to test, and fp_v8's +0.0118 "gain" is selection bias.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, "/workspace/vz20/audit")
from common import ART, TE_SMOOTH, fast_auc, load_data  # noqa: E402
from e2_crosshalf import build_families  # noqa: E402

FAMILIES = (
    "byte",
    "byte_hi",
    "bit64",
    "bytepair_xor",
    "bytepair_and",
    "withinbyte_bit_and",
    "crossbyte_bit_xor",
    "withinbyte_tri_and",
    "bytepair_concat",
    "CONTROL_xfeatures",
)
SEEDS = (11, 22, 33)
N_PERM = 20


def _table(codes, yv, n_levels, prior, smooth=TE_SMOOTH):
    s = np.bincount(codes, weights=yv.astype(float), minlength=n_levels)
    c = np.bincount(codes, minlength=n_levels).astype(float)
    return (s + smooth * prior) / (c + smooth), c


def crosshalf(keys, y, idx_fit, idx_eval, mode):
    """mode: 'none' | 'oof_sign' (honest) | 'eval_leak' (fp_v8 style)."""
    y_fit, y_eval = y[idx_fit], y[idx_eval]
    prior = float(y_fit.mean())
    acc = np.zeros(len(idx_eval), dtype=float)
    inner = None
    if mode == "oof_sign":
        inner = list(StratifiedKFold(5, shuffle=True, random_state=7).split(np.zeros(len(idx_fit)), y_fit))
    for tr_codes, _t, n_levels in keys:
        cf, ce = tr_codes[idx_fit], tr_codes[idx_eval]
        table, cnt = _table(cf, y_fit, n_levels, prior)
        v_eval = np.where(cnt[ce] > 0, table[ce], prior)
        if mode == "oof_sign":
            # direction learned honestly *inside* the fitting half
            oof = np.empty(len(idx_fit))
            for tri, vai in inner:
                tb, tc = _table(cf[tri], y_fit[tri], n_levels, float(y_fit[tri].mean()))
                oof[vai] = np.where(tc[cf[vai]] > 0, tb[cf[vai]], prior)
            if fast_auc(y_fit, oof) < 0.5:
                v_eval = -v_eval
        elif mode == "eval_leak":
            if fast_auc(y_eval, v_eval) < 0.5:
                v_eval = -v_eval
        acc += rankdata(v_eval) / len(v_eval)
    return fast_auc(y_eval, acc / len(keys))


def family_score(keys, y, mode, seeds=SEEDS):
    out = []
    for seed in seeds:
        for ia, ib in StratifiedKFold(2, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
            out.append(crosshalf(keys, y, ia, ib, mode))
    return np.asarray(out)


def main():
    train, test, _ = load_data()
    y = train["label"].astype(int).to_numpy()

    fam_real = build_families(train, test)
    real = {}
    print("=== real ids ===", flush=True)
    for name in FAMILIES:
        a_h = family_score(fam_real[name], y, "oof_sign")
        a_n = family_score(fam_real[name], y, "none")
        a_l = family_score(fam_real[name], y, "eval_leak")
        real[name] = {
            "honest_oof_sign_mean": float(a_h.mean()),
            "honest_oof_sign_std": float(a_h.std(ddof=1)),
            "noflip_mean": float(a_n.mean()),
            "eval_leak_mean": float(a_l.mean()),
            "leak_inflation": float(a_l.mean() - a_h.mean()),
        }
        print(f"{name:<22} honest={a_h.mean():.4f} noflip={a_n.mean():.4f} LEAK={a_l.mean():.4f}", flush=True)

    print(f"\n=== null: {N_PERM} id permutations ===", flush=True)
    null = {n: {"honest": [], "leak": []} for n in FAMILIES}
    t0 = time.time()
    for p in range(N_PERM):
        rng = np.random.default_rng(1000 + p)
        tr_sh = train.copy()
        tr_sh["id"] = train["id"].to_numpy()[rng.permutation(len(train))]
        fam_sh = build_families(tr_sh, test)
        for name in FAMILIES:
            null[name]["honest"].append(float(family_score(fam_sh[name], y, "oof_sign", seeds=SEEDS[:2]).mean()))
            null[name]["leak"].append(float(family_score(fam_sh[name], y, "eval_leak", seeds=SEEDS[:2]).mean()))
        print(f"  perm {p+1}/{N_PERM}  ({time.time()-t0:.0f}s)", flush=True)

    summary = {}
    print(f"\n{'family':<22}{'real':>9}{'null_mu':>10}{'null_sd':>9}{'z':>8}{'p_ge':>8}{'realLEAK':>10}{'nullLEAK':>10}")
    print("-" * 86)
    for name in FAMILIES:
        h = np.asarray(null[name]["honest"])
        lk = np.asarray(null[name]["leak"])
        r = real[name]["honest_oof_sign_mean"]
        z = float((r - h.mean()) / h.std(ddof=1)) if h.std(ddof=1) > 0 else 0.0
        p = float((np.sum(h >= r) + 1) / (len(h) + 1))
        summary[name] = {
            **real[name],
            "null_honest_mean": float(h.mean()),
            "null_honest_std": float(h.std(ddof=1)),
            "z_vs_null": z,
            "p_one_sided": p,
            "null_leak_mean": float(lk.mean()),
            "leak_is_explained_by_null": bool(abs(real[name]["eval_leak_mean"] - lk.mean()) < 2 * lk.std(ddof=1)),
        }
        print(
            f"{name:<22}{r:>9.4f}{h.mean():>10.4f}{h.std(ddof=1):>9.4f}{z:>8.2f}{p:>8.3f}"
            f"{real[name]['eval_leak_mean']:>10.4f}{lk.mean():>10.4f}"
        )

    out = {"n_perm": N_PERM, "seeds": list(SEEDS), "families": summary}
    (ART / "e3_permutation_null.json").write_text(json.dumps(out, indent=2) + "\n")
    print("\nwrote", ART / "e3_permutation_null.json")


if __name__ == "__main__":
    main()

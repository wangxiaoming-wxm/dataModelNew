#!/usr/bin/env python3
"""E2/E3/E4: honest cross-half transfer for every id component family.

Protocol per family (this is the whole point of the audit):

  * split train into halves A/B (stratified, seeded)
  * fit EVERYTHING on A only: the TE table AND the sign of each key
  * apply the frozen table+sign to B, rank-normalise, average over keys
  * score AUC on B.  Then swap A/B.  Repeat over seeds.

fp_v8 instead decides the sign with `roc_auc_score(y, oof) < 0.5` on the FULL
label vector and then reports the AUC of that same vector. We reproduce that
variant too ("insample_flip") to quantify how much of its score is pure
selection bias.

Controls:
  * shuffled ids (id column permuted against y) -> must land on 0.5
  * real business features (x0..x18 binned) -> positive control, must be >0.5
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, "/workspace/vz20/audit")
from common import ART, TE_SMOOTH, fast_auc, id_bits, id_bytes, load_data  # noqa: E402

SEEDS = (11, 22, 33, 44, 55)


# --------------------------------------------------------------------------
# key families
# --------------------------------------------------------------------------
def build_families(train, test):
    """dict name -> list of (train_codes, test_codes, n_levels) key columns."""
    tb, teb = id_bytes(train["id"]), id_bytes(test["id"])
    tbit, tebit = id_bits(tb), id_bits(teb)
    fam = {}

    fam["byte"] = [(tb[:, i].astype(np.int64), teb[:, i].astype(np.int64), 256) for i in range(8)]
    fam["byte_hi"] = [((tb[:, i] >> 4).astype(np.int64), (teb[:, i] >> 4).astype(np.int64), 16) for i in range(8)]
    fam["bit64"] = [(tbit[:, j].astype(np.int64), tebit[:, j].astype(np.int64), 2) for j in range(64)]

    xor, and_ = [], []
    for i in range(8):
        for j in range(i + 1, 8):
            xor.append(((tb[:, i] ^ tb[:, j]).astype(np.int64), (teb[:, i] ^ teb[:, j]).astype(np.int64), 256))
            and_.append(((tb[:, i] & tb[:, j]).astype(np.int64), (teb[:, i] & teb[:, j]).astype(np.int64), 256))
    fam["bytepair_xor"] = xor
    fam["bytepair_and"] = and_

    wa, wo, wx = [], [], []
    for b in range(8):
        for i in range(8):
            for j in range(i + 1, 8):
                a1, a2 = tbit[:, b * 8 + i], tbit[:, b * 8 + j]
                e1, e2 = tebit[:, b * 8 + i], tebit[:, b * 8 + j]
                wa.append(((a1 & a2).astype(np.int64), (e1 & e2).astype(np.int64), 2))
                wo.append(((a1 | a2).astype(np.int64), (e1 | e2).astype(np.int64), 2))
                wx.append(((a1 ^ a2).astype(np.int64), (e1 ^ e2).astype(np.int64), 2))
    fam["withinbyte_bit_and"] = wa
    fam["withinbyte_bit_or"] = wo
    fam["withinbyte_bit_xor"] = wx

    ca, co, cx = [], [], []
    for bit in range(8):
        for b1 in range(8):
            for b2 in range(b1 + 1, 8):
                a1, a2 = tbit[:, b1 * 8 + bit], tbit[:, b2 * 8 + bit]
                e1, e2 = tebit[:, b1 * 8 + bit], tebit[:, b2 * 8 + bit]
                ca.append(((a1 & a2).astype(np.int64), (e1 & e2).astype(np.int64), 2))
                co.append(((a1 | a2).astype(np.int64), (e1 | e2).astype(np.int64), 2))
                cx.append(((a1 ^ a2).astype(np.int64), (e1 ^ e2).astype(np.int64), 2))
    fam["crossbyte_bit_and"] = ca
    fam["crossbyte_bit_or"] = co
    fam["crossbyte_bit_xor"] = cx

    tri = []
    for b in range(8):
        for i in range(8):
            for j in range(i + 1, 8):
                for k in range(j + 1, 8):
                    tri.append(
                        (
                            (tbit[:, b * 8 + i] & tbit[:, b * 8 + j] & tbit[:, b * 8 + k]).astype(np.int64),
                            (tebit[:, b * 8 + i] & tebit[:, b * 8 + j] & tebit[:, b * 8 + k]).astype(np.int64),
                            2,
                        )
                    )
    fam["withinbyte_tri_and"] = tri

    pc = []
    for i in range(8):
        for j in range(i + 1, 8):
            pc.append(
                (
                    (tb[:, i].astype(np.int64) * 256 + tb[:, j]).astype(np.int64),
                    (teb[:, i].astype(np.int64) * 256 + teb[:, j]).astype(np.int64),
                    65536,
                )
            )
    fam["bytepair_concat"] = pc

    # positive control: genuine tabular features, 20-quantile binned
    xs = []
    for i in range(19):
        col = f"x{i}"
        v = train[col].to_numpy(float)
        edges = np.quantile(v, np.linspace(0, 1, 21)[1:-1])
        xs.append((np.digitize(v, edges).astype(np.int64), np.digitize(test[col].to_numpy(float), edges).astype(np.int64), 21))
    fam["CONTROL_xfeatures"] = xs

    return fam


# --------------------------------------------------------------------------
# evaluation modes
# --------------------------------------------------------------------------
def _te_table(codes, y, n_levels, smooth=TE_SMOOTH):
    prior = float(y.mean())
    s = np.bincount(codes, weights=y.astype(float), minlength=n_levels)
    c = np.bincount(codes, minlength=n_levels).astype(float)
    return (s + smooth * prior) / (c + smooth), c, prior


def crosshalf_pool(keys, y, idx_fit, idx_eval, flip_mode="fit_half"):
    """Fit TE + sign on idx_fit, score idx_eval. Returns (auc, pooled score).

    flip_mode:
      "fit_half"  - honest: sign chosen from the fitting half only
      "none"      - no sign flipping at all
      "eval_leak" - fp_v8 style: sign chosen using the labels of the eval rows
    """
    y_fit, y_eval = y[idx_fit], y[idx_eval]
    acc = np.zeros(len(idx_eval), dtype=float)
    for tr_codes, _te_codes, n_levels in keys:
        cf, ce = tr_codes[idx_fit], tr_codes[idx_eval]
        table, cnt, prior = _te_table(cf, y_fit, n_levels)
        v_eval = np.where(cnt[ce] > 0, table[ce], prior)
        if flip_mode == "fit_half":
            v_fit = np.where(cnt[cf] > 0, table[cf], prior)
            if fast_auc(y_fit, v_fit) < 0.5:
                v_eval = -v_eval
        elif flip_mode == "eval_leak":
            if fast_auc(y_eval, v_eval) < 0.5:
                v_eval = -v_eval
        acc += rankdata(v_eval) / len(v_eval)
    acc /= len(keys)
    return fast_auc(y_eval, acc), acc


def run_family(name, keys, y, seeds=SEEDS):
    res = {}
    for mode in ("fit_half", "none", "eval_leak"):
        aucs = []
        for seed in seeds:
            skf = StratifiedKFold(2, shuffle=True, random_state=seed)
            for ia, ib in skf.split(np.zeros(len(y)), y):
                aucs.append(crosshalf_pool(keys, y, ia, ib, mode)[0])
        a = np.asarray(aucs)
        res[mode] = {
            "mean": float(a.mean()),
            "std": float(a.std(ddof=1)),
            "min": float(a.min()),
            "max": float(a.max()),
            "n_runs": int(a.size),
            "n_above_half": int((a > 0.5).sum()),
            "t_stat_vs_half": float((a.mean() - 0.5) / (a.std(ddof=1) / np.sqrt(a.size))) if a.std(ddof=1) > 0 else 0.0,
        }
    return res


def main():
    train, test, _ = load_data()
    y = train["label"].astype(int).to_numpy()
    fam = build_families(train, test)

    out = {"n_train": int(len(y)), "seeds": list(SEEDS), "note": "cross-half: fit on one half, evaluate on the other"}

    print(f"{'family':<24}{'n_keys':>7}{'honest':>10}{'std':>8}{'>0.5':>7}{'noflip':>10}{'evalLEAK':>10}{'t':>8}")
    print("-" * 84)
    real = {}
    for name, keys in fam.items():
        t0 = time.time()
        r = run_family(name, keys, y)
        r["n_keys"] = len(keys)
        r["seconds"] = round(time.time() - t0, 1)
        real[name] = r
        print(
            f"{name:<24}{len(keys):>7}{r['fit_half']['mean']:>10.4f}{r['fit_half']['std']:>8.4f}"
            f"{r['fit_half']['n_above_half']:>4}/{r['fit_half']['n_runs']:<3}"
            f"{r['none']['mean']:>10.4f}{r['eval_leak']['mean']:>10.4f}{r['fit_half']['t_stat_vs_half']:>8.2f}",
            flush=True,
        )
    out["real_ids"] = real

    # ---------------- E4: shuffled-id control ----------------
    print("\n=== E4: id column permuted against y (must collapse to 0.5) ===", flush=True)
    rng = np.random.default_rng(20260812)
    perm = rng.permutation(len(train))
    tr_sh = train.copy()
    tr_sh["id"] = train["id"].to_numpy()[perm]
    fam_sh = build_families(tr_sh, test)
    shuf = {}
    for name in ("byte", "bit64", "bytepair_xor", "bytepair_and", "crossbyte_bit_xor", "withinbyte_bit_and", "CONTROL_xfeatures"):
        keys = fam_sh[name]
        r = run_family(name, keys, y, seeds=SEEDS[:3])
        shuf[name] = r
        print(
            f"{name:<24}{len(keys):>7}{r['fit_half']['mean']:>10.4f}{r['fit_half']['std']:>8.4f}"
            f"{r['fit_half']['n_above_half']:>4}/{r['fit_half']['n_runs']:<3}"
            f"{r['none']['mean']:>10.4f}{r['eval_leak']['mean']:>10.4f}",
            flush=True,
        )
    out["shuffled_ids"] = shuf

    ART.mkdir(parents=True, exist_ok=True)
    (ART / "e2_crosshalf.json").write_text(json.dumps(out, indent=2) + "\n")
    print("\nwrote", ART / "e2_crosshalf.json")


if __name__ == "__main__":
    main()

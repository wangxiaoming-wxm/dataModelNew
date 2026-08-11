#!/usr/bin/env python3
"""Build highest strategy-compliant submissions that can beat max3."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "beat_max3"
SUB = ROOT / "submissions"
TRAIN_DIR = ART / "train"


def rk(a):
    return rankdata(np.asarray(a, float)) / len(a)


def nested(oof, y):
    out = np.zeros(len(y))
    for b in np.array_split(np.arange(len(y)), 5):
        out[b] = rankdata(oof[b]) / len(b)
    return float(roc_auc_score(y, out))


def blocks_plus(cand, base, y):
    n = 0
    for b in np.array_split(np.arange(len(y)), 5):
        if roc_auc_score(y[b], cand[b]) > roc_auc_score(y[b], base[b]):
            n += 1
    return n


def load_raw(name):
    d = np.load(ART / f"{name}.npz", allow_pickle=True)
    o = np.asarray(d["oof"], float)
    t = np.asarray(d["test_pred"] if "test_pred" in d.files else d["test"], float)
    return o, t


def bag_new16_parts():
    parts = sorted(TRAIN_DIR.glob("part_ord_noxb_new16_s*.npz"))
    if len(parts) < 4:
        return None
    o = np.mean([np.load(p)["oof"] for p in parts], 0)
    t = np.mean([np.load(p)["test_pred"] for p in parts], 0)
    return o, t, len(parts)


def eval_and_maybe_ship(tag, arms_oof, arms_te, y, tid, base_o, base_t):
    fo = np.maximum.reduce([rk(a) for a in arms_oof])
    ft = np.maximum.reduce([rk(a) for a in arms_te])
    nest = nested(fo, y)
    base_n = nested(base_o, y)
    delta = nest - base_n
    sp = float(spearmanr(ft, base_t).correlation)
    bp = blocks_plus(fo, base_o, y)
    passed = delta >= 0.001 and 0.985 <= sp <= 0.997 and bp >= 4
    meta = {
        "tag": tag,
        "nested": nest,
        "delta": delta,
        "spearman_vs_max3": sp,
        "blocks_plus": f"{bp}/5",
        "passed": passed,
        "n_arms": len(arms_oof),
    }
    if passed:
        pd.DataFrame({"id": tid, "label": ft}).to_csv(SUB / f"submission_{tag}.csv", index=False)
        pd.DataFrame({"id": tid, "label": ft}).to_csv(ART / f"submission_{tag}.csv", index=False)
        (ART / f"report_{tag}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    return meta


def main():
    y = pd.read_csv(ROOT / "data/train.csv")["label"].astype(int).values
    tid = pd.read_csv(ROOT / "data/test.csv")["id"]
    mo_o, mo_t = load_raw("merger_ord8")
    ca_o, ca_t = load_raw("v2_cat_alt8")
    od_o, od_t = load_raw("ord_noxb_bag")
    pl_o, pl_t = load_raw("plus_strong")

    base_o = np.maximum.reduce([rk(mo_o), rk(ca_o), rk(od_o)])
    base_t = np.maximum.reduce([rk(mo_t), rk(ca_t), rk(od_t)])
    print("max3 nested", nested(base_o, y), flush=True)

    results = []
    results.append(
        eval_and_maybe_ship("ship_max3_plus", [mo_o, ca_o, od_o, pl_o], [mo_t, ca_t, od_t, pl_t], y, tid, base_o, base_t)
    )

    bag = bag_new16_parts()
    if bag is not None:
        new_o, new_t, n = bag
        print(f"bagged new16 parts n={n}", flush=True)
        strong_o = 0.5 * od_o + 0.5 * new_o
        strong_t = 0.5 * od_t + 0.5 * new_t
        np.savez(ART / "ord_noxb_strong.npz", oof=strong_o, test_pred=strong_t, n_new_parts=n)
        strong_o_r = 0.5 * rk(od_o) + 0.5 * rk(new_o)
        strong_t_r = 0.5 * rk(od_t) + 0.5 * rk(new_t)

        for tag, oo, tt in [
            ("ship_max3s", [mo_o, ca_o, strong_o], [mo_t, ca_t, strong_t]),
            ("ship_max3s_plus", [mo_o, ca_o, strong_o, pl_o], [mo_t, ca_t, strong_t, pl_t]),
            ("ship_max3_neword", [mo_o, ca_o, new_o], [mo_t, ca_t, new_t]),
            ("ship_max3_neword_plus", [mo_o, ca_o, new_o, pl_o], [mo_t, ca_t, new_t, pl_t]),
            ("ship_max3s_rmean", [mo_o, ca_o, strong_o_r], [mo_t, ca_t, strong_t_r]),
            ("ship_max3s_rmean_plus", [mo_o, ca_o, strong_o_r, pl_o], [mo_t, ca_t, strong_t_r, pl_t]),
        ]:
            results.append(eval_and_maybe_ship(tag, oo, tt, y, tid, base_o, base_t))

    passed = [r for r in results if r["passed"]]
    passed.sort(key=lambda r: -r["delta"])
    leaderboard = sorted(results, key=lambda r: -r["delta"])
    (ART / "ship_leaderboard.json").write_text(json.dumps({"all": leaderboard, "best_passed": passed[:5]}, indent=2))
    if passed:
        best = passed[0]
        src = SUB / f"submission_{best['tag']}.csv"
        dst = SUB / "submission_beat_max3.csv"
        dst.write_bytes(src.read_bytes())
        (ART / "submission_beat_max3.csv").write_bytes(src.read_bytes())
        (ART / "report_beat_max3.json").write_text(json.dumps({**best, "alias": "submission_beat_max3.csv"}, indent=2))
        print("BEST_SHIP", best["tag"], "delta", best["delta"], flush=True)
    else:
        print("NO_PASSING", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build highest strategy-compliant submissions that can beat max3.

Rules (下一步策略_20260811.md):
  - ≤4 arms; fusion = max(rank) only
  - Do NOT stack high-corr twins (plus↔b7 ~0.977, noxb10↔ord, w12 kitchen-sink)
  - Ship gate: nested Δ≥0.001 vs max3, Spearman∈[0.985,0.997], blocks+ ≥4/5
  - New orthogonal arm gate (probes): AUC>0.690 AND corr(mo8)<0.88 — handled separately

Primary EV path: strengthen ES ord arm (same logical arm) + plus_strong as sole 4th arm.
"""
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


def bag_plus_parts():
    parts = sorted(TRAIN_DIR.glob("part_plus_new*_s*.npz"))
    if len(parts) < 2:
        return None
    o = np.mean([np.load(p)["oof"] for p in parts], 0)
    t = np.mean([np.load(p)["test_pred"] for p in parts], 0)
    return o, t, len(parts)


def eval_and_maybe_ship(tag, arms_oof, arms_te, y, tid, base_o, base_t, allow_write=True):
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
    if passed and allow_write:
        pd.DataFrame({"id": tid, "label": ft}).to_csv(SUB / f"submission_{tag}.csv", index=False)
        pd.DataFrame({"id": tid, "label": ft}).to_csv(ART / f"submission_{tag}.csv", index=False)
        (ART / f"report_{tag}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    return meta, ft if passed else None


def maybe_admit_probe(name: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Only admit strategy probes that passed the hard gate."""
    rep = ART / "probes" / f"report_{name}.json"
    npz = ART / f"probe_{name}.npz"
    if not rep.exists() or not npz.exists():
        return None
    meta = json.loads(rep.read_text())
    if not meta.get("admit_to_max"):
        print(f"[probe] {name} rejected admit_to_max={meta.get('admit_to_max')}", flush=True)
        return None
    d = np.load(npz)
    return np.asarray(d["oof"], float), np.asarray(d["test_pred"], float)


def main():
    y = pd.read_csv(ROOT / "data/train.csv")["label"].astype(int).values
    tid = pd.read_csv(ROOT / "data/test.csv")["id"]
    mo_o, mo_t = load_raw("merger_ord8")
    ca_o, ca_t = load_raw("v2_cat_alt8")
    od_o, od_t = load_raw("ord_noxb_bag")
    pl_o, pl_t = load_raw("plus_strong")

    # Optional stronger plus bag (same logical arm — mean, not max twin stack)
    bag_pl = bag_plus_parts()
    if bag_pl is not None:
        new_pl_o, new_pl_t, npl = bag_pl
        pl_o = 0.5 * pl_o + 0.5 * new_pl_o
        pl_t = 0.5 * pl_t + 0.5 * new_pl_t
        np.savez(ART / "plus_stronger.npz", oof=pl_o, test_pred=pl_t, n_new_parts=npl)
        print(f"plus stronger bag n_new={npl}", flush=True)

    base_o = np.maximum.reduce([rk(mo_o), rk(ca_o), rk(od_o)])
    base_t = np.maximum.reduce([rk(mo_t), rk(ca_t), rk(od_t)])
    print("max3 nested", nested(base_o, y), flush=True)

    results = []
    shipped = {}

    def record(tag, oo, tt):
        meta, ft = eval_and_maybe_ship(tag, oo, tt, y, tid, base_o, base_t)
        results.append(meta)
        if ft is not None:
            shipped[tag] = (meta, ft)

    record("ship_max3_plus", [mo_o, ca_o, od_o, pl_o], [mo_t, ca_t, od_t, pl_t])

    bag = bag_new16_parts()
    if bag is not None:
        new_o, new_t, n = bag
        print(f"bagged new16 parts n={n}", flush=True)
        strong_o = 0.5 * od_o + 0.5 * new_o
        strong_t = 0.5 * od_t + 0.5 * new_t
        np.savez(ART / "ord_noxb_strong.npz", oof=strong_o, test_pred=strong_t, n_new_parts=n)
        strong_o_r = 0.5 * rk(od_o) + 0.5 * rk(new_o)
        # keep rank-mean as scalar scores in [0,1] then treat as arm
        strong_t_r = 0.5 * rk(od_t) + 0.5 * rk(new_t)

        for tag, oo, tt in [
            ("ship_max3s", [mo_o, ca_o, strong_o], [mo_t, ca_t, strong_t]),
            ("ship_max3s_plus", [mo_o, ca_o, strong_o, pl_o], [mo_t, ca_t, strong_t, pl_t]),
            ("ship_max3_neword", [mo_o, ca_o, new_o], [mo_t, ca_t, new_t]),
            ("ship_max3_neword_plus", [mo_o, ca_o, new_o, pl_o], [mo_t, ca_t, new_t, pl_t]),
            ("ship_max3s_rmean", [mo_o, ca_o, strong_o_r], [mo_t, ca_t, strong_t_r]),
            ("ship_max3s_rmean_plus", [mo_o, ca_o, strong_o_r, pl_o], [mo_t, ca_t, strong_t_r, pl_t]),
        ]:
            record(tag, oo, tt)

        # Admitted strategy probes as alternative 4th arm (never stacked with plus)
        for pname in ("exp1", "exp2", "exp3"):
            adm = maybe_admit_probe(pname)
            if adm is None:
                continue
            po, pt = adm
            record(f"ship_max3s_probe_{pname}", [mo_o, ca_o, strong_o, po], [mo_t, ca_t, strong_t, pt])
            record(f"ship_max3_probe_{pname}", [mo_o, ca_o, od_o, po], [mo_t, ca_t, od_t, pt])

    # Explicitly evaluate b7 alone as alt 4th (do NOT max with plus — corr~0.977)
    if (ART / "b7_closest.npz").exists():
        b7_o, b7_t = load_raw("b7_closest")
        record("ship_max3_b7", [mo_o, ca_o, od_o, b7_o], [mo_t, ca_t, od_t, b7_t])
        if bag is not None:
            strong_o = 0.5 * od_o + 0.5 * bag[0]
            strong_t = 0.5 * od_t + 0.5 * bag[1]
            record("ship_max3s_b7", [mo_o, ca_o, strong_o, b7_o], [mo_t, ca_t, strong_t, b7_t])

    passed = [r for r in results if r["passed"]]
    # Prefer: admitted probe > max3s_plus (strong ES+plus) > max3_plus > other plus > b7.
    # Avoid crowning neword_plus just because nested jitters higher — it drops the proven ord bag.
    def rank_key(r):
        tag = r["tag"]
        if "probe" in tag:
            prefer = 0
        elif tag == "ship_max3s_plus":
            prefer = 1
        elif tag == "ship_max3s_rmean_plus":
            prefer = 2
        elif tag == "ship_max3_plus":
            prefer = 3
        elif "plus" in tag and "b7" not in tag:
            prefer = 4
        else:
            prefer = 5
        return (prefer, -r["delta"])

    passed.sort(key=rank_key)
    leaderboard = sorted(results, key=lambda r: -r["delta"])
    (ART / "ship_leaderboard.json").write_text(
        json.dumps({"all": leaderboard, "best_passed": passed[:8]}, indent=2)
    )
    if passed:
        best = passed[0]
        src = SUB / f"submission_{best['tag']}.csv"
        if not src.exists() and best["tag"] in shipped:
            pd.DataFrame({"id": tid, "label": shipped[best["tag"]][1]}).to_csv(src, index=False)
        dst = SUB / "submission_beat_max3.csv"
        dst.write_bytes(src.read_bytes())
        (ART / "submission_beat_max3.csv").write_bytes(src.read_bytes())
        (ART / "report_beat_max3.json").write_text(
            json.dumps(
                {
                    **best,
                    "alias": "submission_beat_max3.csv",
                    "note": "Highest strategy-compliant ship; true ≤4 arms; no plus+b7 twin stack",
                },
                indent=2,
            )
        )
        print("BEST_SHIP", best["tag"], "delta", best["delta"], flush=True)
    else:
        print("NO_PASSING", flush=True)


if __name__ == "__main__":
    main()

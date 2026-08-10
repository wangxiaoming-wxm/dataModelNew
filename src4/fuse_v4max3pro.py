"""DEPRECATED for the official V4max3pro submission.

Use src4/build_submission_v4max3pro.py instead. This file remains as an
exploratory multi-arm scanner from the research phase; its older
artifacts/v4max3pro/fuse_report.json is intentionally removed.

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
ART_BASE = ROOT / "artifacts" / "v4max3"
ART_PRO = ROOT / "artifacts" / "v4max3pro"
SUB = ROOT / "submissions"
DATA = ROOT / "data"
N_BLOCKS = 5

# Pre-registered arm inventory. Tags: honest | es | plus10
ARM_SPECS = {
    "merger_ord8": ("v4max3/merger_ord8.npz", "honest"),
    "v2_cat_alt8": ("v4max3/v2_cat_alt8.npz", "honest"),
    "ord_noxb_bag": ("v4max3/ord_noxb_bag.npz", "es"),
    "ordered_bag": ("v4max3/ordered_bag.npz", "es"),
    "plus_v10": ("v4max3/plus_v10.npz", "plus10"),
    "b7_closest": ("v4max3/b7_closest.npz", "es"),
    # optional upgrades written by trainers
    "merger_ord_es": ("v4max3pro/merger_ord_es.npz", "es"),
    "cat_alt_es": ("v4max3pro/cat_alt_es.npz", "es"),
    "plus_ord5": ("v4max3pro/plus_ord5.npz", "es"),
    "plus_plain5": ("v4max3pro/plus_plain5.npz", "es"),
    "plus10_ord": ("v4max3pro/plus10_ord.npz", "plus10"),
    "alt2fix8": ("v4max3pro/alt2fix8.npz", "honest"),
    "main_ord_es": ("v4max3pro/main_ord_es.npz", "es"),
    "main10_ord_es": ("v4max3pro/main10_ord_es.npz", "plus10"),
    "hybrid10": ("v4max3pro/hybrid10.npz", "plus10"),
    "alt10": ("v4max3pro/alt10.npz", "plus10"),
    "noxb10": ("v4max3pro/noxb10.npz", "plus10"),
    "plus_v10_8": ("v4max3pro/plus_v10_8.npz", "plus10"),
    "plus_strong": ("v4max3pro/plus_strong.npz", "plus10"),
    "plus10_ord": ("v4max3pro/plus10_ord.npz", "plus10"),
}


def _load_npz(path: Path):
    d = np.load(path, allow_pickle=True)
    oof = np.asarray(d["oof"], dtype=float)
    if "test_pred" in d:
        te = np.asarray(d["test_pred"], dtype=float)
    elif "test" in d:
        te = np.asarray(d["test"], dtype=float)
    else:
        raise KeyError(f"no test key in {path}")
    return oof, te


def load_arm(name: str):
    rel, tag = ARM_SPECS[name]
    path = ROOT / "artifacts" / rel
    if not path.exists():
        return None
    oof, te = _load_npz(path)
    if oof.shape[0] != 14930 or te.shape[0] != 6398:
        return None
    return {
        "name": name,
        "tag": tag,
        "oof_raw": oof,
        "te_raw": te,
        "oof": rankdata(oof) / len(oof),
        "te": rankdata(te) / len(te),
        "oof_auc": float(roc_auc_score(load_arm.y, oof)),
    }


def nested_auc(y, oof, n_blocks=N_BLOCKS):
    n = len(y)
    out = np.zeros(n)
    for b in np.array_split(np.arange(n), n_blocks):
        out[b] = rankdata(oof[b]) / len(b)
    return float(roc_auc_score(y, out))


def apply_rule(kind, members, which, arms):
    stacked = np.vstack([arms[m][which] for m in members])
    if kind == "max":
        return stacked.max(axis=0)
    if kind == "rmean":
        return rankdata(stacked.mean(axis=0)) / stacked.shape[1]
    if kind == "mean":
        return stacked.mean(axis=0)
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--min-delta", type=float, default=0.0015)
    args = ap.parse_args()

    y = pd.read_csv(DATA / "train.csv")["label"].astype(int).values
    load_arm.y = y
    test = pd.read_csv(DATA / "test.csv")
    base_sub = pd.read_csv(SUB / "submission_v4_max3.csv")["label"].values

    arms = {}
    for name in ARM_SPECS:
        a = load_arm(name)
        if a is None:
            continue
        arms[name] = a
        print(f"  [{a['tag']:7s}] {name:16s} oof={a['oof_auc']:.5f}")

    # frozen max3
    max3_members = ["merger_ord8", "v2_cat_alt8", "ord_noxb_bag"]
    assert all(m in arms for m in max3_members)
    max3_oof = apply_rule("max", max3_members, "oof", arms)
    max3_te = apply_rule("max", max3_members, "te", arms)
    max3_nested = nested_auc(y, max3_oof)
    print(f"\nmax3 nested={max3_nested:.5f} full={roc_auc_score(y, max3_oof):.5f}")
    print(f"max3 vs frozen submission spearman={spearmanr(max3_te, base_sub).correlation:.6f}")

    names = list(arms)
    results = []
    for r in range(2, min(len(names), 5) + 1):
        for combo in combinations(names, r):
            # must keep at least 2 of the max3 core to stay on-recipe
            core_hit = sum(1 for c in combo if c in max3_members)
            if core_hit < 2:
                continue
            for kind in ("max", "rmean"):
                oof = apply_rule(kind, combo, "oof", arms)
                te = apply_rule(kind, combo, "te", arms)
                nest = nested_auc(y, oof)
                full = float(roc_auc_score(y, oof))
                tags = {arms[c]["tag"] for c in combo}
                honest_only = tags <= {"honest"}
                label = f"{kind}{r}_" + "+".join(combo)
                results.append(
                    {
                        "label": label,
                        "kind": kind,
                        "members": list(combo),
                        "nested": nest,
                        "full": full,
                        "delta_vs_max3": nest - max3_nested,
                        "honest_only": honest_only,
                        "tags": sorted(tags),
                        "spearman_vs_max3": float(spearmanr(te, max3_te).correlation),
                        "te": te,
                    }
                )

    results.sort(key=lambda d: -d["nested"])
    print(f"\n{'rule':70s} {'nested':>8s} {'delta':>8s} {'sp':>7s} tags")
    for r in results[:30]:
        print(
            f"{r['label'][:70]:70s} {r['nested']:8.5f} {r['delta_vs_max3']:+8.5f} "
            f"{r['spearman_vs_max3']:7.4f} {','.join(r['tags'])}"
        )

    # prefer recipes that include the three max3 arms (upgrade path)
    upgrades = [
        r
        for r in results
        if r["kind"] == "max"
        and set(max3_members).issubset(r["members"])
        and r["delta_vs_max3"] >= 0
    ]
    best_upgrade = upgrades[0] if upgrades else None
    best_any = results[0]

    # admission — prefer upgrades that keep the exact max3 trio and add new arms.
    # Reject "admit" if the only lift comes from recycling b7_closest (known optimistic).
    admit = False
    chosen = None
    if best_upgrade and best_upgrade["delta_vs_max3"] >= args.min_delta:
        members = set(best_upgrade["members"])
        recycles_only = members <= set(max3_members) | {"ordered_bag", "b7_closest", "plus_v10"}
        has_new_train = any(
            m.startswith(("plus_ord", "plus_plain", "plus_strong", "plus10", "plus_v10_", "merger_ord_es", "cat_alt_es", "alt2fix", "main_ord", "main10", "hybrid10", "alt10", "noxb10"))
            for m in members
        )
        if 0.97 <= best_upgrade["spearman_vs_max3"] <= 0.9995 and (has_new_train or not recycles_only):
            # still require new trained arms before burning the last submit
            if has_new_train:
                admit = True
                chosen = best_upgrade
            else:
                print("NOTE: inventory-only lift — exploratory, not admit for last submit.")

    # shuffled sanity on chosen / best upgrade
    sanity = None
    probe = chosen or best_upgrade or best_any
    if probe is not None:
        rng = np.random.default_rng(0)
        y_s = rng.permutation(y)
        sanity = float(roc_auc_score(y_s, apply_rule(probe["kind"], probe["members"], "oof", arms)))

    expected_lb = None
    if probe is not None:
        expected_lb = probe["nested"] + (0.71222 - max3_nested)

    report = {
        "max3_nested": max3_nested,
        "max3_public_lb": 0.71222,
        "cv_to_lb_gap": 0.71222 - max3_nested,
        "target_lb": 0.7155,
        "needed_nested": 0.7155 - (0.71222 - max3_nested),
        "min_delta_gate": args.min_delta,
        "arms_present": {n: {"oof_auc": arms[n]["oof_auc"], "tag": arms[n]["tag"]} for n in arms},
        "best_any": {k: probe[k] for k in ("label", "nested", "full", "delta_vs_max3", "tags", "spearman_vs_max3", "members")}
        if (probe := best_any)
        else None,
        "best_upgrade": {
            k: best_upgrade[k]
            for k in ("label", "nested", "full", "delta_vs_max3", "tags", "spearman_vs_max3", "members")
        }
        if best_upgrade
        else None,
        "admit": admit,
        "chosen": {
            k: chosen[k]
            for k in ("label", "nested", "full", "delta_vs_max3", "tags", "spearman_vs_max3", "members")
        }
        if chosen
        else None,
        "expected_lb_if_same_gap": expected_lb,
        "shuffled_sanity_auc": sanity,
        "top10": [
            {
                k: r[k]
                for k in ("label", "nested", "full", "delta_vs_max3", "tags", "spearman_vs_max3", "members")
            }
            for r in results[:10]
        ],
    }
    ART_PRO.mkdir(parents=True, exist_ok=True)
    (ART_PRO / "fuse_report.json").write_text(json.dumps(report, indent=2))
    print("\n" + json.dumps({k: report[k] for k in report if k != "top10"}, indent=2))

    if args.write:
        # always write exploratory best-upgrade for inspection
        for tag, rec in [("upgrade", best_upgrade), ("any", best_any)]:
            if rec is None:
                continue
            out = SUB / f"submission_v4max3pro_{tag}.csv"
            pd.DataFrame({"id": test["id"], "label": np.clip(rec["te"], 0.001, 0.999)}).to_csv(
                out, index=False
            )
            print(f"wrote {out} nested={rec['nested']:.5f} delta={rec['delta_vs_max3']:+.5f}")
        if admit and chosen is not None:
            out = SUB / "submission_v4max3pro.csv"
            pd.DataFrame({"id": test["id"], "label": np.clip(chosen["te"], 0.001, 0.999)}).to_csv(
                out, index=False
            )
            print(f"ADMIT wrote {out}")
        else:
            print("NOT ADMITTED — do not burn the last submission slot on this yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

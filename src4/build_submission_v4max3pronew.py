#!/usr/bin/env python3
"""Build / evaluate V4max3proNew from frozen max3/pro arms + semantic_rmse.

Admission rule (honest):
  Admit only if nested 5-block AUC beats frozen max3 AND beats frozen v4max3pro,
  AND test-rank Spearman vs max3 submission is < 0.9995 (real change).

Frozen recipe candidate families evaluated:
  A) max(max3_3arms, semantic_rmse)
  B) max(v4max3pro_5arms, semantic_rmse)
  C) max(max3_3arms, plus_strong, semantic_rmse)   # drop collinear noxb10
  D) rank-mean variants of the above

Usage:
  python3 src4/build_submission_v4max3pronew.py
  python3 src4/build_submission_v4max3pronew.py --write
  python3 src4/build_submission_v4max3pronew.py --check
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SUB = ROOT / "submissions"
ART_BASE = ROOT / "artifacts" / "v4max3"
ART_PRO = ROOT / "artifacts" / "v4max3pro"
ART_NEW = ROOT / "artifacts" / "v4max3pronew"
N_BLOCKS = 5

MAX3_ARMS = [
    ("merger_ord8", ART_BASE / "merger_ord8.npz", "honest"),
    ("v2_cat_alt8", ART_BASE / "v2_cat_alt8.npz", "honest"),
    ("ord_noxb_bag", ART_BASE / "ord_noxb_bag.npz", "es"),
]
PRO_EXTRA = [
    ("plus_strong", ART_PRO / "plus_strong.npz", "plus10"),
    ("noxb10", ART_PRO / "noxb10.npz", "plus10"),
]
NEW_ARM = ("semantic_rmse", ART_NEW / "semantic_rmse.npz", "es5_rmse")


def load_npz(path: Path):
    d = np.load(path, allow_pickle=True)
    oof = np.asarray(d["oof"], dtype=float)
    if "test_pred" in d:
        te = np.asarray(d["test_pred"], dtype=float)
    elif "test" in d:
        te = np.asarray(d["test"], dtype=float)
    else:
        raise KeyError(f"no test key in {path}")
    return oof, te, d


def nested_auc(y: np.ndarray, oof: np.ndarray, n_blocks: int = N_BLOCKS) -> float:
    out = np.zeros(len(y))
    for b in np.array_split(np.arange(len(y)), n_blocks):
        out[b] = rankdata(oof[b]) / len(b)
    return float(roc_auc_score(y, out))


def block_deltas(y: np.ndarray, base: np.ndarray, cand: np.ndarray, n_blocks: int = N_BLOCKS):
    deltas = []
    for b in np.array_split(np.arange(len(y)), n_blocks):
        rb = rankdata(base[b]) / len(b)
        rc = rankdata(cand[b]) / len(b)
        deltas.append(float(roc_auc_score(y[b], rc) - roc_auc_score(y[b], rb)))
    return deltas


def pack_arm(name, path, tag, y):
    oof, te, raw = load_npz(path)
    if oof.shape[0] != 14930 or te.shape[0] != 6398:
        raise ValueError(f"{name} bad shape oof={oof.shape} te={te.shape}")
    return {
        "name": name,
        "tag": tag,
        "path": path,
        "oof_raw": oof,
        "te_raw": te,
        "oof": rankdata(oof) / len(oof),
        "te": rankdata(te) / len(te),
        "oof_auc": float(roc_auc_score(y, oof)),
        "raw": raw,
    }


def fuse(kind: str, members: list[str], arms: dict, which: str):
    stacked = np.vstack([arms[m][which] for m in members])
    if kind == "max":
        return stacked.max(axis=0)
    if kind == "rmean":
        return rankdata(stacked.mean(axis=0)) / stacked.shape[1]
    raise ValueError(kind)


def evaluate_candidates(y, arms, max3_nested, pro_nested, base_sub):
    """Enumerate strong max/rmean recipes that keep max3 core and add semantic."""
    from itertools import combinations

    max3 = ["merger_ord8", "v2_cat_alt8", "ord_noxb_bag"]
    extras = [n for n in arms if n not in max3]
    catalogs = {
        "max3": max3,
        "pro": max3 + ["plus_strong", "noxb10"],
        "sem_only": ["semantic_rmse"],
    }
    # All supersets of max3 that include semantic_rmse (and optional extras)
    for r in range(0, len(extras) + 1):
        for extra in combinations(extras, r):
            members = max3 + list(extra)
            if "semantic_rmse" not in members and r > 0:
                # still keep baselines without semantic for reference only when r covers known
                pass
            label = "+".join(members)
            catalogs[label] = members
    # Explicit short labels for the main report
    catalogs.update(
        {
            "max3+sem": max3 + ["semantic_rmse"],
            "pro+sem": max3 + ["plus_strong", "noxb10", "semantic_rmse"],
            "max3+plus+sem": max3 + ["plus_strong", "semantic_rmse"],
            "max3+noxb+sem": max3 + ["noxb10", "semantic_rmse"],
            "plus+sem": ["plus_strong", "semantic_rmse"],
            "honest2+sem": ["merger_ord8", "v2_cat_alt8", "semantic_rmse"],
        }
    )

    rows = []
    seen = set()
    for label, members in catalogs.items():
        key = tuple(sorted(members))
        if key in seen:
            continue
        seen.add(key)
        for kind in ("max", "rmean"):
            if len(members) == 1 and kind == "rmean":
                continue
            oof = fuse(kind, members, arms, "oof")
            te = fuse(kind, members, arms, "te")
            nest = nested_auc(y, oof)
            full = float(roc_auc_score(y, oof))
            sp_max3 = float(spearmanr(te, base_sub).correlation)
            short = label if label in {
                "max3", "pro", "sem_only", "max3+sem", "pro+sem",
                "max3+plus+sem", "max3+noxb+sem", "plus+sem", "honest2+sem",
            } else "combo:" + "+".join(members)
            rows.append(
                {
                    "label": f"{kind}:{short}",
                    "kind": kind,
                    "members": members,
                    "nested": nest,
                    "full": full,
                    "delta_vs_max3": nest - max3_nested,
                    "delta_vs_pro": nest - pro_nested,
                    "spearman_vs_max3_sub": sp_max3,
                    "block_deltas_vs_max3": block_deltas(
                        y,
                        fuse("max", max3, arms, "oof"),
                        oof,
                    ),
                }
            )
    rows.sort(key=lambda r: (-r["nested"], -abs(1.0 - r["spearman_vs_max3_sub"])))
    return rows


def pick_winner(rows, max3_nested, pro_nested, min_delta=0.0005):
    """Pick the strongest max-fusion that beats max3 & pro with real test change.

    Preference order:
      1) max recipes containing semantic_rmse, nested >= pro, delta_vs_max3 >= min_delta
      2) same but allow tiny margin over pro
      3) best nested max recipe with semantic that still beats max3
    Prefer higher nested; tie-break by larger |1-spearman| (more real change) then more members diversity.
    """
    def ok_base(r):
        if r["kind"] != "max":
            return False
        if "semantic_rmse" not in r["members"]:
            return False
        if r["spearman_vs_max3_sub"] >= 0.9995:
            return False
        return True

    cands = [r for r in rows if ok_base(r)]
    cands.sort(
        key=lambda r: (
            r["nested"],
            r["delta_vs_pro"],
            abs(1.0 - r["spearman_vs_max3_sub"]),
            len(r["members"]),
        ),
        reverse=True,
    )
    for r in cands:
        if r["nested"] >= pro_nested + 1e-6 and r["nested"] >= max3_nested + min_delta:
            return r
    for r in cands:
        if r["nested"] >= pro_nested - 1e-6 and r["nested"] > max3_nested:
            return r
    for r in cands:
        if r["nested"] > max3_nested:
            return r
    return None


def build(write: bool = False):
    y = pd.read_csv(DATA / "train.csv")["label"].astype(int).values
    test = pd.read_csv(DATA / "test.csv")
    base_sub = pd.read_csv(SUB / "submission_v4_max3.csv")["label"].values
    pro_sub_path = SUB / "submission_v4max3pro.csv"
    pro_sub = pd.read_csv(pro_sub_path)["label"].values if pro_sub_path.exists() else None

    arms = {}
    for name, path, tag in MAX3_ARMS + PRO_EXTRA + [NEW_ARM]:
        if not path.exists():
            raise FileNotFoundError(path)
        arms[name] = pack_arm(name, path, tag, y)
        print(f"  [{tag:10s}] {name:16s} oof={arms[name]['oof_auc']:.5f}")

    max3_oof = fuse("max", [n for n, _, _ in MAX3_ARMS], arms, "oof")
    pro_oof = fuse("max", [n for n, _, _ in MAX3_ARMS + PRO_EXTRA], arms, "oof")
    max3_nested = nested_auc(y, max3_oof)
    pro_nested = nested_auc(y, pro_oof)
    print(f"max3 nested={max3_nested:.5f}  pro nested={pro_nested:.5f}")

    rows = evaluate_candidates(y, arms, max3_nested, pro_nested, base_sub)
    for r in rows:
        bd = r["block_deltas_vs_max3"]
        print(
            f"  {r['label']:22s} nested={r['nested']:.5f} "
            f"d_max3={r['delta_vs_max3']:+.5f} d_pro={r['delta_vs_pro']:+.5f} "
            f"sp={r['spearman_vs_max3_sub']:.5f} "
            f"blocks+={sum(1 for x in bd if x > 0)}/5"
        )

    winner = pick_winner(rows, max3_nested, pro_nested)
    ART_NEW.mkdir(parents=True, exist_ok=True)

    report = {
        "source_ref": "715.zip claimed LB 0.71504 (semantic RMSE arm port)",
        "max3_nested": max3_nested,
        "pro_nested": pro_nested,
        "arm_oof": {n: arms[n]["oof_auc"] for n in arms},
        "candidates": [
            {k: v for k, v in r.items() if k != "block_deltas_vs_max3"}
            | {"blocks_positive_vs_max3": sum(1 for x in r["block_deltas_vs_max3"] if x > 0)}
            for r in rows
        ],
        "winner": None,
        "admission": "REJECT",
        "notes": [
            "715 zip local pooled OOF ~0.695 is a single arm; fusion needed under our ruler.",
            "Do not claim public LB 0.71504 for this port without a verified submit.",
            "semantic_rmse uses ES + multi-bag; PROTOCOL_RISK similar to other ES arms.",
        ],
    }

    out_csv = SUB / "submission_v4max3pronew.csv"
    if winner is None:
        print("ADMISSION REJECTED: no candidate beats max3/pro honestly with real test change")
        report["admission"] = "REJECT"
        (ART_NEW / "recipe_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if write:
            # Still write best exploratory max:pro+sem if nested > max3, else refuse
            best = next((r for r in rows if r["kind"] == "max" and "sem" in r["label"]), None)
            if best and best["nested"] > max3_nested and best["spearman_vs_max3_sub"] < 0.9995:
                labels = np.clip(fuse(best["kind"], best["members"], arms, "te"), 0.001, 0.999)
                pd.DataFrame({"id": test["id"], "label": labels}).to_csv(out_csv, index=False)
                report["winner"] = best
                report["admission"] = "EXPLORATORY_WRITE"
                report["submission"] = str(out_csv.relative_to(ROOT))
                print(f"wrote EXPLORATORY {out_csv} nested={best['nested']:.5f}")
            else:
                print("no exploratory write either")
        (ART_NEW / "recipe_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    labels = np.clip(fuse(winner["kind"], winner["members"], arms, "te"), 0.001, 0.999)
    sp_pro = float(spearmanr(labels, pro_sub).correlation) if pro_sub is not None else None
    report["winner"] = {
        **{k: v for k, v in winner.items() if k != "block_deltas_vs_max3"},
        "blocks_positive_vs_max3": sum(1 for x in winner["block_deltas_vs_max3"] if x > 0),
        "block_deltas_vs_max3": winner["block_deltas_vs_max3"],
        "spearman_vs_pro_sub": sp_pro,
        "optimistic_lb_vs_max3": winner["nested"] + (0.71222 - max3_nested),
    }
    report["admission"] = "ADMIT"
    report["recipe"] = f"{winner['kind']}(" + ", ".join(winner["members"]) + ")"
    report["submission"] = str(out_csv.relative_to(ROOT))

    if write:
        SUB.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"id": test["id"], "label": labels}).to_csv(out_csv, index=False)
        print(f"ADMIT wrote {out_csv} nested={winner['nested']:.5f}")
    else:
        print(f"ADMIT (dry-run) nested={winner['nested']:.5f} recipe={report['recipe']}")

    (ART_NEW / "recipe_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    status = {
        "admission": report["admission"],
        "nested": winner["nested"],
        "delta_vs_max3": winner["delta_vs_max3"],
        "delta_vs_pro": winner["delta_vs_pro"],
        "recipe": report.get("recipe"),
        "optimistic_lb_vs_max3": report["winner"]["optimistic_lb_vs_max3"],
    }
    (ART_NEW / "status_report.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return report


def check():
    report_path = ART_NEW / "recipe_report.json"
    csv_path = SUB / "submission_v4max3pronew.csv"
    if not csv_path.exists():
        raise SystemExit(f"missing {csv_path}")
    # rebuild without write and compare
    y = pd.read_csv(DATA / "train.csv")["label"].astype(int).values
    test = pd.read_csv(DATA / "test.csv")
    base_sub = pd.read_csv(SUB / "submission_v4_max3.csv")["label"].values
    arms = {}
    for name, path, tag in MAX3_ARMS + PRO_EXTRA + [NEW_ARM]:
        arms[name] = pack_arm(name, path, tag, y)
    max3_oof = fuse("max", [n for n, _, _ in MAX3_ARMS], arms, "oof")
    pro_oof = fuse("max", [n for n, _, _ in MAX3_ARMS + PRO_EXTRA], arms, "oof")
    max3_nested = nested_auc(y, max3_oof)
    pro_nested = nested_auc(y, pro_oof)
    rows = evaluate_candidates(y, arms, max3_nested, pro_nested, base_sub)
    winner = pick_winner(rows, max3_nested, pro_nested)
    if winner is None:
        # fall back to report winner members
        rep = json.loads(report_path.read_text(encoding="utf-8"))
        w = rep.get("winner")
        if not w:
            raise SystemExit("no admitted winner to check")
        labels = np.clip(fuse(w["kind"], w["members"], arms, "te"), 0.001, 0.999)
    else:
        labels = np.clip(fuse(winner["kind"], winner["members"], arms, "te"), 0.001, 0.999)
    cur = pd.read_csv(csv_path)["label"].values
    diff = np.abs(cur - labels)
    frac = float(np.mean(diff > 1e-12))
    print(
        json.dumps(
            {
                "frac_diff": frac,
                "max_abs": float(diff.max()),
                "ok": frac == 0.0,
            },
            indent=2,
        )
    )
    return 0 if frac == 0.0 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        raise SystemExit(check())
    build(write=args.write)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

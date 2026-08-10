#!/usr/bin/env python3
"""Independent audit of V4max3proNew — recalculate from raw artifacts, ignore JSON narrative."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path("/workspace")
DATA = ROOT / "data"
SUB = ROOT / "submissions"
ART_BASE = ROOT / "artifacts" / "v4max3"
ART_PRO = ROOT / "artifacts" / "v4max3pro"
ART_NEW = ROOT / "artifacts" / "v4max3pronew"
N_BLOCKS = 5
RNG = np.random.default_rng(42)


def load_npz(path: Path):
    d = np.load(path, allow_pickle=True)
    oof = np.asarray(d["oof"], dtype=float)
    if "test_pred" in d:
        te = np.asarray(d["test_pred"], dtype=float)
    elif "test" in d:
        te = np.asarray(d["test"], dtype=float)
    else:
        raise KeyError(path)
    return oof, te


def nested_auc(y, oof, n_blocks=N_BLOCKS):
    out = np.zeros(len(y))
    for b in np.array_split(np.arange(len(y)), n_blocks):
        out[b] = rankdata(oof[b]) / len(b)
    return float(roc_auc_score(y, out))


def block_aucs(y, oof, n_blocks=N_BLOCKS):
    out = []
    for b in np.array_split(np.arange(len(y)), n_blocks):
        out.append(float(roc_auc_score(y[b], rankdata(oof[b]) / len(b))))
    return out


def block_deltas(y, base, cand, n_blocks=N_BLOCKS):
    deltas = []
    for b in np.array_split(np.arange(len(y)), n_blocks):
        rb = rankdata(base[b]) / len(b)
        rc = rankdata(cand[b]) / len(b)
        deltas.append(float(roc_auc_score(y[b], rc) - roc_auc_score(y[b], rb)))
    return deltas


def fuse_max(ranked_list):
    return np.vstack(ranked_list).max(axis=0)


def rank01(x):
    return rankdata(x) / len(x)


def main():
    y = pd.read_csv(DATA / "train.csv")["label"].astype(int).values
    assert len(y) == 14930

    arms_spec = {
        "merger_ord8": (ART_BASE / "merger_ord8.npz", "honest"),
        "v2_cat_alt8": (ART_BASE / "v2_cat_alt8.npz", "honest"),
        "ord_noxb_bag": (ART_BASE / "ord_noxb_bag.npz", "es"),
        "plus_strong": (ART_PRO / "plus_strong.npz", "plus10_es"),
        "noxb10": (ART_PRO / "noxb10.npz", "plus10_es"),
        "semantic_rmse": (ART_NEW / "semantic_rmse.npz", "es5x5x10"),
        "semantic_logloss": (ART_NEW / "semantic_logloss.npz", "es5x5x10"),
    }

    arms = {}
    print("=" * 72)
    print("ARM RAW OOF AUC + NESTED + PROTOCOL")
    print("=" * 72)
    for name, (path, tag) in arms_spec.items():
        oof, te = load_npz(path)
        assert oof.shape == (14930,) and te.shape == (6398,), (name, oof.shape, te.shape)
        raw_auc = float(roc_auc_score(y, oof))
        nest = nested_auc(y, oof)
        # label shuffle sanity
        y_shuf = y.copy()
        RNG.shuffle(y_shuf)
        shuf_auc = float(roc_auc_score(y_shuf, oof))
        arms[name] = {
            "tag": tag,
            "oof_raw": oof,
            "te_raw": te,
            "oof": rank01(oof),
            "te": rank01(te),
            "raw_auc": raw_auc,
            "nested": nest,
            "shuf_auc": shuf_auc,
        }
        print(
            f"{name:18s} tag={tag:10s} raw={raw_auc:.5f} nested={nest:.5f} "
            f"shuf_y_auc={shuf_auc:.5f}"
        )

    max3_members = ["merger_ord8", "v2_cat_alt8", "ord_noxb_bag"]
    pro_members = max3_members + ["plus_strong", "noxb10"]
    new_members = pro_members + ["semantic_rmse"]

    max3_oof = fuse_max([arms[m]["oof"] for m in max3_members])
    pro_oof = fuse_max([arms[m]["oof"] for m in pro_members])
    new_oof = fuse_max([arms[m]["oof"] for m in new_members])
    sem_oof = arms["semantic_rmse"]["oof"]

    max3_te = fuse_max([arms[m]["te"] for m in max3_members])
    pro_te = fuse_max([arms[m]["te"] for m in pro_members])
    new_te = np.clip(fuse_max([arms[m]["te"] for m in new_members]), 0.001, 0.999)

    # submissions
    sub_max3 = pd.read_csv(SUB / "submission_v4_max3.csv")["label"].values
    sub_pro = pd.read_csv(SUB / "submission_v4max3pro.csv")["label"].values
    sub_new = pd.read_csv(SUB / "submission_v4max3pronew.csv")["label"].values

    # check rebuild equals submission
    frac_diff_new = float(np.mean(np.abs(sub_new - new_te) > 1e-12))
    max_abs_new = float(np.max(np.abs(sub_new - new_te)))
    # also check max3/pro rebuilt vs sub (clip)
    max3_te_clip = np.clip(max3_te, 0.001, 0.999)
    pro_te_clip = np.clip(pro_te, 0.001, 0.999)
    frac_diff_max3 = float(np.mean(np.abs(sub_max3 - max3_te_clip) > 1e-12))
    frac_diff_pro = float(np.mean(np.abs(sub_pro - pro_te_clip) > 1e-12))

    print("\n" + "=" * 72)
    print("FUSION NESTED / RAW / DELTAS")
    print("=" * 72)
    rows = {}
    for label, oof in [("max3", max3_oof), ("pro", pro_oof), ("New", new_oof), ("semantic_rmse", sem_oof)]:
        raw = float(roc_auc_score(y, oof))
        nest = nested_auc(y, oof)
        ba = block_aucs(y, oof)
        rows[label] = {"raw": raw, "nested": nest, "block_aucs": ba}
        print(f"{label:14s} raw={raw:.5f} nested={nest:.5f} blocks={[round(x,5) for x in ba]}")

    d_new_max3 = rows["New"]["nested"] - rows["max3"]["nested"]
    d_new_pro = rows["New"]["nested"] - rows["pro"]["nested"]
    d_pro_max3 = rows["pro"]["nested"] - rows["max3"]["nested"]
    d_sem_max3 = rows["semantic_rmse"]["nested"] - rows["max3"]["nested"]

    bd_new_max3 = block_deltas(y, max3_oof, new_oof)
    bd_new_pro = block_deltas(y, pro_oof, new_oof)
    bd_pro_max3 = block_deltas(y, max3_oof, pro_oof)

    print(f"\nΔnested New-max3 = {d_new_max3:+.5f}  blocks+={sum(1 for x in bd_new_max3 if x>0)}/5  deltas={[round(x,5) for x in bd_new_max3]}")
    print(f"Δnested New-pro  = {d_new_pro:+.5f}  blocks+={sum(1 for x in bd_new_pro if x>0)}/5  deltas={[round(x,5) for x in bd_new_pro]}")
    print(f"Δnested pro-max3 = {d_pro_max3:+.5f}  blocks+={sum(1 for x in bd_pro_max3 if x>0)}/5  deltas={[round(x,5) for x in bd_pro_max3]}")
    print(f"Δnested sem-max3 = {d_sem_max3:+.5f}")

    # Block-level bootstrap P(Δ>0): resample 5 blocks with replacement
    def boot_p_pos(deltas, n=10000):
        deltas = np.asarray(deltas)
        means = []
        for _ in range(n):
            idx = RNG.integers(0, len(deltas), size=len(deltas))
            means.append(deltas[idx].mean())
        means = np.asarray(means)
        return float(np.mean(means > 0)), float(np.mean(means)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    p_new_max3, m_new_max3, lo_nm, hi_nm = boot_p_pos(bd_new_max3)
    p_new_pro, m_new_pro, lo_np, hi_np = boot_p_pos(bd_new_pro)
    p_pro_max3, m_pro_max3, lo_pm, hi_pm = boot_p_pos(bd_pro_max3)

    print("\n" + "=" * 72)
    print("BLOCK BOOTSTRAP P(mean Δ>0)  [n=10000, block-level]")
    print("=" * 72)
    print(f"New vs max3: P(Δ>0)={p_new_max3:.3f} mean={m_new_max3:+.5f} CI95=[{lo_nm:+.5f},{hi_nm:+.5f}]")
    print(f"New vs pro : P(Δ>0)={p_new_pro:.3f} mean={m_new_pro:+.5f} CI95=[{lo_np:+.5f},{hi_np:+.5f}]")
    print(f"pro vs max3: P(Δ>0)={p_pro_max3:.3f} mean={m_pro_max3:+.5f} CI95=[{lo_pm:+.5f},{hi_pm:+.5f}]")

    print("\n" + "=" * 72)
    print("TEST SPEARMAN (submissions)")
    print("=" * 72)
    sp_new_max3 = float(spearmanr(sub_new, sub_max3).correlation)
    sp_new_pro = float(spearmanr(sub_new, sub_pro).correlation)
    sp_pro_max3 = float(spearmanr(sub_pro, sub_max3).correlation)
    print(f"New vs max3 sub: {sp_new_max3:.5f}")
    print(f"New vs pro  sub: {sp_new_pro:.5f}")
    print(f"pro vs max3 sub: {sp_pro_max3:.5f}")
    print(f"rebuild New frac_diff={frac_diff_new} max_abs={max_abs_new}")
    print(f"rebuild max3 frac_diff={frac_diff_max3}  pro frac_diff={frac_diff_pro}")

    # label shuffle on fusions
    print("\n" + "=" * 72)
    print("LABEL SHUFFLE SANITY (fusion OOF AUC should ~0.5)")
    print("=" * 72)
    for label, oof in [("max3", max3_oof), ("pro", pro_oof), ("New", new_oof), ("semantic_rmse", sem_oof)]:
        ys = y.copy()
        RNG.shuffle(ys)
        print(f"{label:14s} shuf_auc={float(roc_auc_score(ys, oof)):.5f}")

    print("\n" + "=" * 72)
    print("COLLINEARITY / COMPLEMENTARITY (OOF Spearman on ranks)")
    print("=" * 72)
    pairs = [
        ("noxb10", "ord_noxb_bag"),
        ("semantic_rmse", "merger_ord8"),
        ("semantic_rmse", "v2_cat_alt8"),
        ("semantic_rmse", "ord_noxb_bag"),
        ("semantic_rmse", "plus_strong"),
        ("semantic_rmse", "noxb10"),
        ("semantic_rmse", "max3_fuse"),
        ("semantic_rmse", "pro_fuse"),
        ("semantic_logloss", "semantic_rmse"),
    ]
    # also te spearman
    fuse_oofs = {"max3_fuse": max3_oof, "pro_fuse": pro_oof}
    fuse_tes = {"max3_fuse": max3_te, "pro_fuse": pro_te}
    for a, b in pairs:
        oa = arms[a]["oof"] if a in arms else fuse_oofs[a]
        ob = arms[b]["oof"] if b in arms else fuse_oofs[b]
        ta = arms[a]["te"] if a in arms else fuse_tes[a]
        tb = arms[b]["te"] if b in arms else fuse_tes[b]
        spo = float(spearmanr(oa, ob).correlation)
        spt = float(spearmanr(ta, tb).correlation)
        print(f"{a:18s} vs {b:18s}  OOF_sp={spo:.5f}  TE_sp={spt:.5f}")

    # fusion gain vs single arm: is New nested lift from picking optimistic combo?
    print("\n" + "=" * 72)
    print("FUSION GAIN ANALYSIS")
    print("=" * 72)
    print(f"semantic_rmse nested={rows['semantic_rmse']['nested']:.5f}")
    print(f"max3 nested={rows['max3']['nested']:.5f}")
    print(f"pro nested={rows['pro']['nested']:.5f}")
    print(f"New nested={rows['New']['nested']:.5f}")
    print(f"New - semantic_rmse nested gap = {rows['New']['nested'] - rows['semantic_rmse']['nested']:+.5f}")
    print(f"New - pro nested (semantic add) = {d_new_pro:+.5f}")
    # max3+sem without pro extras
    max3sem_oof = fuse_max([arms[m]["oof"] for m in max3_members + ["semantic_rmse"]])
    print(f"max3+sem nested={nested_auc(y, max3sem_oof):.5f}  (vs max3 Δ={nested_auc(y, max3sem_oof)-rows['max3']['nested']:+.5f})")
    # plus+sem
    plussem = fuse_max([arms["plus_strong"]["oof"], arms["semantic_rmse"]["oof"]])
    print(f"plus+sem nested={nested_auc(y, plussem):.5f}")

    # optimistic LB extrapolation (NOT verified)
    claimed_max3_lb = 0.71222
    gap = claimed_max3_lb - rows["max3"]["nested"]
    opt_new = rows["New"]["nested"] + gap
    opt_pro = rows["pro"]["nested"] + gap
    print("\n" + "=" * 72)
    print("OPTIMISTIC EXTRAPOLATION (NOT VERIFIED LB)")
    print("=" * 72)
    print(f"assumed max3 LB={claimed_max3_lb}  nested={rows['max3']['nested']:.5f}  gap={gap:.5f}")
    print(f"opt New LB ≈ {opt_new:.5f}   opt pro LB ≈ {opt_pro:.5f}")
    print(f"claimed 715 zip LB=0.71504  — cannot verify; single-arm zip only")

    # compare to doc
    doc = {
        "max3_nested": 0.70307,
        "pro_nested": 0.70522,
        "new_nested": 0.70557,
        "delta_vs_max3": 0.00250,
        "delta_vs_pro": 0.00035,
        "sp_vs_max3": 0.98920,
        "sp_vs_pro": 0.99727,
    }
    print("\n" + "=" * 72)
    print("DOC CONSISTENCY (rounded)")
    print("=" * 72)
    checks = [
        ("max3_nested", rows["max3"]["nested"], doc["max3_nested"]),
        ("pro_nested", rows["pro"]["nested"], doc["pro_nested"]),
        ("new_nested", rows["New"]["nested"], doc["new_nested"]),
        ("d_max3", d_new_max3, doc["delta_vs_max3"]),
        ("d_pro", d_new_pro, doc["delta_vs_pro"]),
        ("sp_max3", sp_new_max3, doc["sp_vs_max3"]),
        ("sp_pro", sp_new_pro, doc["sp_vs_pro"]),
    ]
    for name, got, expect in checks:
        ok = abs(got - expect) < 5.5e-5  # allow rounding to 5 decimals
        print(f"{name:12s} got={got:.5f} doc={expect:.5f} match5dp={ok}")

    # admission gate critique
    print("\n" + "=" * 72)
    print("ADMISSION GATE CRITIQUE")
    print("=" * 72)
    min_delta = 0.0005
    beats_max3 = rows["New"]["nested"] >= rows["max3"]["nested"] + min_delta
    beats_pro = rows["New"]["nested"] >= rows["pro"]["nested"] + 1e-6
    real_change = sp_new_max3 < 0.9995
    print(f"beats max3 by >=0.0005? {beats_max3} ({d_new_max3:+.5f})")
    print(f"beats pro by >0? {beats_pro} ({d_new_pro:+.5f})")
    print(f"sp vs max3 < 0.9995? {real_change} ({sp_new_max3:.5f})")
    print(f"BUT vs pro Spearman={sp_new_pro:.5f} (nearly identical submit)")
    print(f"Δ vs pro is only {d_new_pro:+.5f} — tiny; one negative block vs pro={sum(1 for x in bd_new_pro if x<=0)}/5 nonpos")

    # selection risk: many candidates screened
    rep = json.loads((ART_NEW / "recipe_report.json").read_text())
    n_cands = len(rep.get("candidates", []))
    print(f"candidate recipes screened in report: {n_cands}")
    # count how many beat pro
    n_beat_pro = sum(1 for c in rep["candidates"] if c.get("delta_vs_pro", -1) > 0)
    print(f"candidates with delta_vs_pro>0: {n_beat_pro}")

    out = {
        "max3": rows["max3"],
        "pro": rows["pro"],
        "New": rows["New"],
        "semantic_rmse": rows["semantic_rmse"],
        "d_new_max3": d_new_max3,
        "d_new_pro": d_new_pro,
        "bd_new_max3": bd_new_max3,
        "bd_new_pro": bd_new_pro,
        "sp_new_max3": sp_new_max3,
        "sp_new_pro": sp_new_pro,
        "p_new_max3": p_new_max3,
        "p_new_pro": p_new_pro,
        "frac_diff_new": frac_diff_new,
        "opt_new": opt_new,
    }
    Path("/tmp/audit_v4max3pronew_summary.json").write_text(json.dumps(out, indent=2))
    print("\nWrote /tmp/audit_v4max3pronew_summary.json")


if __name__ == "__main__":
    main()

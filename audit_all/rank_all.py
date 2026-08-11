#!/usr/bin/env python3
"""Independent honest ranking across ALL branch/zip candidates.

Ruler: 5-block nested AUC via np.array_split contiguous blocks, re-rank inside block.
Does NOT call any delivered fuse scripts. Fusion = rank -> elementwise max (-> optional clip).
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path("/workspace")
DATA = ROOT / "data"
OPUS = Path("/tmp/audit_all/opus/20260810-cursor-opus5")
ZCODE = Path("/tmp/audit_all/zcode/20260808-zcode-cursor")
V4P = Path("/tmp/audit_all/v4max3pro")
V5 = Path("/tmp/audit_all/v5")
OUT = Path("/tmp/audit_all/out")
OUT.mkdir(parents=True, exist_ok=True)

Y = pd.read_csv(DATA / "train.csv")["label"].astype(int).values
TEST = pd.read_csv(DATA / "test.csv")
TRAIN_SHA = hashlib.sha256((DATA/"train.csv").read_bytes()).hexdigest()
TEST_SHA = hashlib.sha256((DATA/"test.csv").read_bytes()).hexdigest()
SUMS = (DATA/"SHA256SUMS.txt").read_text() if (DATA/"SHA256SUMS.txt").exists() else ""

def rank01(a):
    a = np.asarray(a, dtype=float)
    return rankdata(a) / len(a)

def nested_auc(oof, y, n_blocks=5):
    out = np.zeros(len(y))
    for b in np.array_split(np.arange(len(y)), n_blocks):
        out[b] = rankdata(oof[b]) / len(b)
    return float(roc_auc_score(y, out))

def load_arm(path: Path):
    d = np.load(path, allow_pickle=True)
    oof = np.asarray(d["oof"], dtype=float)
    key = "test_pred" if "test_pred" in d.files else ("test" if "test" in d.files else None)
    te = np.asarray(d[key], dtype=float) if key else None
    ys = np.asarray(d["y"], dtype=int) if "y" in d.files else None
    return oof, te, ys, list(d.files)

def fuse_max(arm_paths, clip=True):
    arm_info = []
    oofs, tes = [], []
    for p in arm_paths:
        oof, te, ys, files = load_arm(p)
        info = {
            "arm": p.name,
            "oof_auc": round(float(roc_auc_score(Y, oof)), 6),
            "nested_single": round(nested_auc(rank01(oof), Y), 6),
            "y_match": None if ys is None else bool(np.array_equal(ys, Y)),
            "has_test": te is not None,
            "keys": files,
        }
        arm_info.append(info)
        oofs.append(rank01(oof))
        if te is not None:
            tes.append(rank01(te))
    f_oof = np.maximum.reduce(oofs)
    f_te = np.maximum.reduce(tes) if tes and len(tes)==len(oofs) else None
    labels = None
    if f_te is not None:
        labels = np.clip(f_te, 0.001, 0.999) if clip else f_te
    return {
        "arms": arm_info,
        "nested_5block": nested_auc(f_oof, Y),
        "full_oof": float(roc_auc_score(Y, f_oof)),
        "nested_10block": nested_auc(f_oof, Y, 10),
        "labels": labels,
        "f_oof": f_oof,
    }

def check_csv(csv_path, labels):
    if csv_path is None or not Path(csv_path).exists():
        return None
    df = pd.read_csv(csv_path)
    committed = df["label"].values
    rep = {
        "rows": int(len(committed)),
        "id_aligned": bool(np.array_equal(df["id"].values, TEST["id"].values)),
        "range": [float(committed.min()), float(committed.max())],
        "nan": int(np.isnan(committed).sum()),
        "sha256": hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
    }
    if labels is not None:
        absdiff = np.abs(committed - labels)
        rep.update({
            "repro_frac_diff": float(np.mean(absdiff > 1e-12)),
            "repro_max_abs": float(absdiff.max()),
            "repro_spearman": float(spearmanr(committed, labels).correlation),
        })
    return rep, committed

# -------- candidates with rebuildable OOF --------
CAND = []

def add(name, source, arms, csv, clip, protocol, es_arms, notes=""):
    CAND.append(dict(name=name, source=source, arms=arms, csv=csv, clip=clip,
                     protocol=protocol, es_arms=es_arms, notes=notes))

# opus zip
add("opus/v4_honest", "zip:20260810-cursor-opus5",
    [OPUS/"v4_honest/artifacts/merger_ord8.npz", OPUS/"v4_honest/artifacts/v2_cat_alt8.npz"],
    OPUS/"v4_honest/submissions/submission_v4_honest.csv", True, "honest_fixed_trees", 0)
add("opus/v4_max3", "zip:20260810-cursor-opus5",
    [OPUS/"v4_max3/artifacts/merger_ord8.npz", OPUS/"v4_max3/artifacts/v2_cat_alt8.npz", OPUS/"v4_max3/artifacts/ord_noxb_bag.npz"],
    OPUS/"v4_max3/submissions/submission_v4_max3.csv", True, "mixed_ES", 1)
add("opus/v5_honest", "zip:20260810-cursor-opus5",
    [OPUS/"v5_honest/artifacts/merger_ord8.npz", OPUS/"v5_honest/artifacts/v2_cat_alt8.npz", OPUS/"v5_honest/artifacts/arm_gap.npz"],
    OPUS/"v5_honest/submissions/submission_v5_honest.csv", True, "honest_fixed_trees", 0)

# zcode zip (same max3/honest + max4)
Z = ZCODE/"artifacts"
ZS = ZCODE/"submissions"
add("zcode/v4_honest", "zip:20260808-zcode-cursor",
    [Z/"merger_ord8.npz", Z/"v2_cat_alt8.npz"],
    ZS/"submission_v4_honest.csv", True, "honest_fixed_trees", 0)
add("zcode/v4_max3", "zip:20260808-zcode-cursor",
    [Z/"merger_ord8.npz", Z/"v2_cat_alt8.npz", Z/"ord_noxb_bag.npz"],
    ZS/"submission_v4_max3.csv", True, "mixed_ES", 1)
add("zcode/v4_max4", "zip:20260808-zcode-cursor",
    [Z/"merger_ord8.npz", Z/"v2_cat_alt8.npz", Z/"ord_noxb_bag.npz", Z/"ordered_bag.npz"],
    ZS/"submission_v4_max4.csv", True, "mixed_ES", 2,
    notes="ordered_bag also uses ES per zcode REPORT")
# zcode v5_honest if arms exist
if (Z/"v5_audit/arm_gap.npz").exists():
    add("zcode/v5_audit_max3_gap", "zip:20260808-zcode-cursor",
        [Z/"v5_audit/arm_cat_d5.npz", Z/"v5_audit/arm_cat_d6.npz", Z/"v5_audit/arm_cat_alt.npz", Z/"v5_audit/arm_gap.npz"],
        ZS/"submission_v5_honest.csv" if (ZS/"submission_v5_honest.csv").exists() else None,
        True, "honest_fixed_trees", 0, notes="v5_audit arms; csv may be different recipe")

# repo v4max3 / pro / pronew
add("repo/v4_max3", "branch:cursor/v4max3pro-f126",
    [V4P/"artifacts/v4max3/merger_ord8.npz", V4P/"artifacts/v4max3/v2_cat_alt8.npz", V4P/"artifacts/v4max3/ord_noxb_bag.npz"],
    V4P/"submissions/submission_v4_max3.csv", True, "mixed_ES", 1)
add("repo/v4max3pro", "branch:cursor/v4max3pro-f126",
    [V4P/"artifacts/v4max3/merger_ord8.npz", V4P/"artifacts/v4max3/v2_cat_alt8.npz", V4P/"artifacts/v4max3/ord_noxb_bag.npz",
     V4P/"artifacts/v4max3pro/plus_strong.npz", V4P/"artifacts/v4max3pro/noxb10.npz"],
    V4P/"submissions/submission_v4max3pro.csv", True, "mixed_ES", 3)
add("repo/v4max3pronew", "branch:cursor/v4max3pronew-f126",
    [V4P/"artifacts/v4max3/merger_ord8.npz", V4P/"artifacts/v4max3/v2_cat_alt8.npz", V4P/"artifacts/v4max3/ord_noxb_bag.npz",
     V4P/"artifacts/v4max3pro/plus_strong.npz", V4P/"artifacts/v4max3pro/noxb10.npz",
     V4P/"artifacts/v4max3pronew/semantic_rmse.npz"],
    V4P/"submissions/submission_v4max3pronew.csv", True, "mixed_ES", 3,
    notes="pro + semantic_rmse; README says logloss diversity rejected")

# main V2/V3/V4 from json arms (not npz)
def load_json_arm(path):
    d = json.loads(Path(path).read_text())
    # common keys
    for k in ("oof", "oof_pred", "pred_oof"):
        if k in d:
            oof = np.asarray(d[k], dtype=float)
            break
    else:
        raise KeyError(path)
    te = None
    for k in ("test", "test_pred", "pred_test"):
        if k in d:
            te = np.asarray(d[k], dtype=float)
            break
    return oof, te

def fuse_max_json(paths, clip=True):
    arm_info=[]
    oofs=[]; tes=[]
    for p in paths:
        oof, te = load_json_arm(p)
        arm_info.append({"arm": Path(p).name, "oof_auc": round(float(roc_auc_score(Y,oof)),6),
                         "nested_single": round(nested_auc(rank01(oof),Y),6)})
        oofs.append(rank01(oof))
        if te is not None: tes.append(rank01(te))
    f_oof = np.maximum.reduce(oofs)
    f_te = np.maximum.reduce(tes) if tes and len(tes)==len(oofs) else None
    labels = np.clip(f_te,0.001,0.999) if (f_te is not None and clip) else f_te
    return {"arms":arm_info,"nested_5block":nested_auc(f_oof,Y),"full_oof":float(roc_auc_score(Y,f_oof)),
            "nested_10block":nested_auc(f_oof,Y,10),"labels":labels,"f_oof":f_oof}

results = {}
fusion_te = {}
fusion_oof = {}

print("DATA sha train", TRAIN_SHA[:16], "test", TEST_SHA[:16])
print("SUMS contains train?", ("train.csv" in SUMS))

for c in CAND:
    missing = [str(p) for p in c["arms"] if not Path(p).exists()]
    if missing:
        results[c["name"]] = {"error": "missing_arms", "missing": missing}
        print("SKIP", c["name"], missing)
        continue
    r = fuse_max(c["arms"], clip=c["clip"])
    csv_rep = None
    committed = None
    if c["csv"] and Path(c["csv"]).exists():
        csv_rep, committed = check_csv(c["csv"], r["labels"])
    # permutation sanity on fusion oof
    rng = np.random.RandomState(0)
    yperm = rng.permutation(Y)
    perm_auc = float(roc_auc_score(yperm, r["f_oof"]))
    entry = {
        "source": c["source"],
        "protocol": c["protocol"],
        "es_arms": c["es_arms"],
        "notes": c["notes"],
        "arms": r["arms"],
        "nested_5block": round(r["nested_5block"], 6),
        "full_oof": round(r["full_oof"], 6),
        "nested_10block": round(r["nested_10block"], 6),
        "perm_auc": round(perm_auc, 5),
        "csv": csv_rep,
    }
    results[c["name"]] = entry
    fusion_oof[c["name"]] = r["f_oof"]
    if committed is not None:
        fusion_te[c["name"]] = committed
    print(f"== {c['name']:28s} nested={r['nested_5block']:.5f} full={r['full_oof']:.5f} proto={c['protocol']} es={c['es_arms']}")

# main branch V2/V3/V4 from fusion reports + arm json
MAIN = {}
for tag, report, arms, csv, proto in [
    ("main/v2", ROOT/"artifacts/v2/fusion_report.json",
     [ROOT/"artifacts/v2/arm_cat_d5.npz", ROOT/"artifacts/v2/arm_cat_d6.npz", ROOT/"artifacts/v2/arm_cat_alt.npz"],
     ROOT/"submissions/submission_v2.csv", "honest_fixed_trees"),
    ("main/v3", ROOT/"artifacts/v3/fusion_report.json",
     [ROOT/"artifacts/v3_f10/arm_cat_d5_f10.npz", ROOT/"artifacts/v3_f10/arm_cat_d6_f10.npz", ROOT/"artifacts/v3_f10/arm_cat_alt_f10.npz"],
     ROOT/"submissions/submission_v3.csv", "honest_fixed_trees"),
    ("main/v4", ROOT/"artifacts/v4/fusion_report_v4.json",
     [ROOT/"artifacts/v4/arm_cat_d5.json", ROOT/"artifacts/v4/arm_cat_d6.json", ROOT/"artifacts/v4/arm_cat_alt.json"],
     ROOT/"submissions/submission_v4.csv", "honest_fixed_trees"),
]:
    # try npz first for v2/v3
    try:
        if all(Path(p).suffix=='.npz' and Path(p).exists() for p in arms):
            r = fuse_max(arms, clip=False)  # main submissions may be unclipped ranks or clipped - check
            # recompute with clip for csv match attempt
            r_clip = fuse_max(arms, clip=True)
        elif all(Path(p).exists() for p in arms):
            # json arms - need flexible loader
            oofs=[]; tes=[]; arm_info=[]
            for p in arms:
                d=json.loads(Path(p).read_text())
                # find oof
                oof=None
                for k in ("oof","oof_pred","pred_oof","oof_rank"):
                    if k in d: oof=np.asarray(d[k],float); break
                if oof is None and "seeds" in d:
                    # bagged seeds?
                    pass
                te=None
                for k in ("test","test_pred","pred_test"):
                    if k in d: te=np.asarray(d[k],float); break
                if oof is None:
                    raise KeyError(f"no oof in {p} keys={list(d)[:20]}")
                arm_info.append({"arm":Path(p).name,"oof_auc":round(float(roc_auc_score(Y,oof)),6),
                                 "nested_single":round(nested_auc(rank01(oof),Y),6),"top_keys":list(d)[:15]})
                oofs.append(rank01(oof))
                if te is not None: tes.append(rank01(te))
            f_oof=np.maximum.reduce(oofs)
            f_te=np.maximum.reduce(tes) if tes and len(tes)==len(oofs) else None
            r={"arms":arm_info,"nested_5block":nested_auc(f_oof,Y),"full_oof":float(roc_auc_score(Y,f_oof)),
               "nested_10block":nested_auc(f_oof,Y,10),"labels": (np.clip(f_te,0.001,0.999) if f_te is not None else None),
               "f_oof":f_oof,"labels_raw":f_te}
            r_clip=r
        else:
            raise FileNotFoundError(arms)
        # also read official fusion report nested if present
        report_nested=None
        if Path(report).exists():
            jr=json.loads(Path(report).read_text())
            report_nested = jr.get("nested_oof_mean") or jr.get("nested_oof") or jr.get("nested")
            # dig
            if report_nested is None:
                for k in ("nested_selection","audit","summary"):
                    if k in jr and isinstance(jr[k], dict):
                        report_nested = jr[k].get("nested_oof_mean") or jr[k].get("nested_oof")
                        if report_nested: break
        csv_rep, committed = check_csv(csv, r.get("labels"))
        # try raw labels too
        if csv_rep and csv_rep.get("repro_frac_diff",1)>0 and r.get("labels_raw") is not None:
            csv_rep2, _ = check_csv(csv, r["labels_raw"])
            if csv_rep2 and csv_rep2.get("repro_frac_diff",1) < csv_rep.get("repro_frac_diff",1):
                csv_rep = csv_rep2
                csv_rep["matched_as"]="raw_rank_max"
        elif csv_rep and csv_rep.get("repro_frac_diff",1)>0:
            csv_rep_c, _ = check_csv(csv, r_clip.get("labels"))
            if csv_rep_c and csv_rep_c.get("repro_frac_diff",1) < csv_rep.get("repro_frac_diff",1):
                csv_rep = csv_rep_c
                csv_rep["matched_as"]="clipped"
        entry={
            "source":"branch:main",
            "protocol":proto,
            "es_arms":0,
            "rebuild_nested_5block_views_max": round(r["nested_5block"],6),
            "rebuild_full_oof": round(r["full_oof"],6),
            "report_nested_field": report_nested,
            "arms": r["arms"],
            "csv": csv_rep,
            "perm_auc": round(float(roc_auc_score(np.random.RandomState(0).permutation(Y), r["f_oof"])),5),
            "notes":"rebuild uses only views_max of listed arms; official nested may use nested rule selection over more rules",
        }
        results[tag]=entry
        fusion_oof[tag]=r["f_oof"]
        if committed is not None: fusion_te[tag]=committed
        print(f"== {tag:28s} rebuild_nested={r['nested_5block']:.5f} report={report_nested} proto={proto}")
    except Exception as e:
        results[tag]={"error":str(e)}
        print("FAIL", tag, e)

# V5 from branch artifacts
try:
    arms=[V5/"artifacts/v5/arm_cat_d5.npz", V5/"artifacts/v5/arm_cat_d6.npz", V5/"artifacts/v5/arm_cat_alt.npz", V5/"artifacts/v5/arm_gap.npz"]
    r=fuse_max(arms, clip=True)
    csv_rep, committed = check_csv(V5/"submissions/submission_v5.csv", r["labels"])
    # also try unclipped
    if csv_rep and csv_rep["repro_frac_diff"]>0:
        r2=fuse_max(arms, clip=False)
        csv_rep2, committed = check_csv(V5/"submissions/submission_v5.csv", r2["labels"])
        if csv_rep2["repro_frac_diff"] < csv_rep["repro_frac_diff"]:
            r, csv_rep = r2, csv_rep2
    # read audit
    audit=json.loads((V5/"artifacts/v5/audit.json").read_text()) if (V5/"artifacts/v5/audit.json").exists() else {}
    fr=json.loads((V5/"artifacts/v5/fusion_report.json").read_text()) if (V5/"artifacts/v5/fusion_report.json").exists() else {}
    results["repo/v5"]={
        "source":"branch:cursor/honest-auc-push-f126",
        "protocol":"honest_fixed_trees",
        "es_arms":0,
        "nested_5block": round(r["nested_5block"],6),
        "full_oof": round(r["full_oof"],6),
        "arms": r["arms"],
        "csv": csv_rep,
        "audit_nested_mean": audit.get("nested_oof_mean") or audit.get("nested_mean"),
        "fusion_report_keys": list(fr)[:20],
        "fusion_nested": fr.get("nested_oof") or fr.get("nested_oof_mean") or fr.get("nested"),
        "perm_auc": round(float(roc_auc_score(np.random.RandomState(0).permutation(Y), r["f_oof"])),5),
    }
    fusion_oof["repo/v5"]=r["f_oof"]
    if committed is not None: fusion_te["repo/v5"]=committed
    print(f"== {'repo/v5':28s} nested={r['nested_5block']:.5f} audit={results['repo/v5'].get('audit_nested_mean')} fr={results['repo/v5'].get('fusion_nested')}")
except Exception as e:
    results["repo/v5"]={"error":str(e)}
    print("FAIL repo/v5", e)

# Cross compare vs LB anchor max3
anchor_name = None
for cand in ("repo/v4_max3","opus/v4_max3","zcode/v4_max3"):
    if cand in fusion_te:
        anchor_name=cand; break
cross={}
if anchor_name:
    anchor=fusion_te[anchor_name]
    ra=rankdata(anchor)
    print(f"\n== vs ANCHOR {anchor_name} ==")
    for name,v in fusion_te.items():
        sp=float(spearmanr(anchor,v).correlation)
        identical=bool(np.array_equal(anchor,v))
        byte_same = results.get(name,{}).get("csv",{}).get("sha256")==results.get(anchor_name,{}).get("csv",{}).get("sha256")
        cross[name]={"spearman":round(sp,6),"identical":identical,"same_sha256":byte_same}
        print(f"  {name:28s} spearman={sp:.6f} identical={identical} same_sha={byte_same}")

# pairwise nested deltas among rebuildable
print("\n== nested ladder ==")
ladder=[]
for name,o in fusion_oof.items():
    n=nested_auc(o,Y)
    proto=results.get(name,{}).get("protocol","?")
    es=results.get(name,{}).get("es_arms","?")
    ladder.append((n,name,proto,es))
ladder.sort(reverse=True)
for n,name,proto,es in ladder:
    print(f"  {n:.5f}  {name:28s}  protocol={proto}  es_arms={es}")

# verified public LB (only facts documented as measured)
verified_lb = {
    "zcode/v3_max3_CLAIMED_IN_REPORT": 0.71184,  # documented in zcode REPORT as measured
    "opus/v4_max3": 0.71222,  # documented + prior audit; same bytes as repo/zcode max3
    "repo/v4_max3": 0.71222,
    "zcode/v4_max3": 0.71222,
    "main/v2": 0.70878,
    "b7_closest": 0.70722,
}
# note: v4_honest LB 0.71104 is CLAIMED in v5_honest README but prior audit says not independently confirmed by user

out = {
    "data_sha": {"train": TRAIN_SHA, "test": TEST_SHA},
    "ruler": "5-block nested AUC, np.array_split contiguous, re-rank inside block; fusion=max(rank)",
    "per_candidate": results,
    "vs_anchor": cross,
    "ladder": [{"nested":round(n,6),"name":name,"protocol":proto,"es_arms":es} for n,name,proto,es in ladder],
    "verified_public_lb_facts": verified_lb,
    "unverified_lb_claims": {
        "v4_honest_LB_0.71104": "claimed in opus v5_honest README; prior audit5 FINDINGS marks as unverified",
        "v5_honest_expected_LB_0.7136": "extrapolation from unverified anchor; not a measurement",
        "v4max3pro_extrapolated_LB": "extrapolation; not measured",
    },
}
(OUT/"rank_all.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
print("\nWrote", OUT/"rank_all.json")

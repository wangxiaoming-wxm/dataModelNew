"""V4 fusion: pre-registered rules + multi-seed nested selection.

Headline number is the mean nested-selection AUC over 20 block seeds
(random_state = 1000..1019). Rules that reference missing arms are skipped.

Delivered submission rule (as of stop): ``views_max_10_20_r16_r16b``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

STRONG = ("cat_d5", "cat_d6", "cat_alt")
STRONG_F20 = ("cat_d5_f20", "cat_d6_f20", "cat_alt_f20")
STRONG_R16 = ("cat_d5_r16", "cat_d6_r16", "cat_alt_r16")
STRONG_S16 = ("cat_d5_s16", "cat_d6_s16", "cat_alt_s16")
ALT_R16B = ("cat_alt_r16b",)
SUB85 = ("cat_d6_sf85", "cat_alt_sf85")
ALT_D5 = ("cat_alt_d5",)
# Opus/zcode honest arms (fixed trees, no ES). Copied into artifacts/v4 as
# arm_merger_ord8.npz / arm_v2_cat_alt8.npz / arm_gap_v5.npz.
OPUS_MA = ("merger_ord8", "v2_cat_alt8")
OPUS_GAP = ("gap_v5",)
V4_CORE = STRONG + STRONG_F20 + STRONG_R16 + ALT_R16B

# Only rules whose member arms exist in artifacts/v4 are kept here.
# Killed experiments (w6–w12, rit, pair, mid, rank, …) are documented in
# docs/V4.md and must not re-enter without a fresh pre-registration + audit.
# V4-ext opus rules were pre-registered BEFORE seeing nested scores (Phase1 plan).
RULES: dict[str, dict[str, float]] = {
    "views_max": {"__max__": 1.0, **{k: 1.0 for k in STRONG}},
    "views_mean": {k: 1 / 3 for k in STRONG},
    "views_half": {"cat_d5": 0.25, "cat_d6": 0.25, "cat_alt": 0.50},
    "cat_pair_max": {"__max__": 1.0, "cat_d5": 1.0, "cat_d6": 1.0},
    "cat_d5_only": {"cat_d5": 1.0},
    "views_max_f20": {"__max__": 1.0, **{k: 1.0 for k in STRONG_F20}},
    "views_mean_f20": {k: 1 / 3 for k in STRONG_F20},
    "views_half_f20": {"cat_d5_f20": 0.25, "cat_d6_f20": 0.25, "cat_alt_f20": 0.50},
    "views_max_10_20": {"__max__": 1.0, **{k: 1.0 for k in STRONG + STRONG_F20}},
    "views_max_r16": {"__max__": 1.0, **{k: 1.0 for k in STRONG_R16}},
    "views_max_10_r16": {"__max__": 1.0, **{k: 1.0 for k in STRONG + STRONG_R16}},
    "views_max_s16": {"__max__": 1.0, **{k: 1.0 for k in STRONG_S16}},
    "views_max_10_20_r16": {
        "__max__": 1.0,
        **{k: 1.0 for k in STRONG + STRONG_F20 + STRONG_R16},
    },
    "views_max_s16_f20": {
        "__max__": 1.0,
        **{k: 1.0 for k in STRONG_S16 + STRONG_F20},
    },
    "views_max_10_20_r16_r16b": {
        "__max__": 1.0,
        **{k: 1.0 for k in STRONG + STRONG_F20 + STRONG_R16 + ALT_R16B},
    },
    "views_max_r16_r16b": {
        "__max__": 1.0,
        **{k: 1.0 for k in STRONG_R16 + ALT_R16B},
    },
    "sub85_max": {"__max__": 1.0, **{k: 1.0 for k in SUB85}},
    "views_max_sub85": {"__max__": 1.0, **{k: 1.0 for k in STRONG + SUB85}},
    "views_max_alt_d5": {"__max__": 1.0, **{k: 1.0 for k in STRONG + ALT_D5}},
    "views_max_10_20_alt_d5": {
        "__max__": 1.0,
        **{k: 1.0 for k in STRONG + STRONG_F20 + ALT_D5},
    },
    "four_max_w5": {"__max__": 1.0, **{k: 1.0 for k in STRONG + ("cat_w5",)}},
    # --- V4-ext Phase1: V4 core ∪ opus honest arms (pre-registered) ---
    "views_max_v4_m": {
        "__max__": 1.0,
        **{k: 1.0 for k in V4_CORE + ("merger_ord8",)},
    },
    "views_max_v4_a": {
        "__max__": 1.0,
        **{k: 1.0 for k in V4_CORE + ("v2_cat_alt8",)},
    },
    "views_max_v4_ma": {
        "__max__": 1.0,
        **{k: 1.0 for k in V4_CORE + OPUS_MA},
    },
    "views_max_v4_mag": {
        "__max__": 1.0,
        **{k: 1.0 for k in V4_CORE + OPUS_MA + OPUS_GAP},
    },
    "opus_v5_honest_max": {
        "__max__": 1.0,
        **{k: 1.0 for k in OPUS_MA + OPUS_GAP},
    },
    # --- Phase2: w12 (main∪alt joint FE), admitted after fast+bag gate ---
    "views_max_v4_mag_w12": {
        "__max__": 1.0,
        **{k: 1.0 for k in V4_CORE + OPUS_MA + OPUS_GAP + ("cat_w12_d5",)},
    },
    "views_max_v4_ma_w12": {
        "__max__": 1.0,
        **{k: 1.0 for k in V4_CORE + OPUS_MA + ("cat_w12_d5",)},
    },
    "views_max_v4_w12": {
        "__max__": 1.0,
        **{k: 1.0 for k in V4_CORE + ("cat_w12_d5",)},
    },
}



def _r(v: np.ndarray) -> np.ndarray:
    return rankdata(v) / (len(v) + 1.0)


def apply_rule(rule: dict[str, float], ranks: dict[str, np.ndarray]) -> np.ndarray | None:
    if "__max__" in rule:
        members = [k for k in rule if k != "__max__" and k in ranks]
        if len(members) < 1:
            return None
        return np.max(np.vstack([ranks[k] for k in members]), axis=0)
    members = [k for k, w in rule.items() if k in ranks]
    if not members:
        return None
    s = sum(rule[k] for k in members)
    return sum((rule[k] / s) * ranks[k] for k in members)


def nested_run(rules, ranks, y, seed):
    assembled = np.zeros(len(y))
    picks = []
    for inner, held in StratifiedKFold(5, shuffle=True, random_state=seed).split(
        np.zeros(len(y)), y
    ):
        best, best_auc = None, -np.inf
        for n, r in rules.items():
            pred = apply_rule(r, {k: v[inner] for k, v in ranks.items()})
            if pred is None:
                continue
            a = roc_auc_score(y[inner], pred)
            if a > best_auc:
                best, best_auc = n, a
        picks.append(best)
        assembled[held] = apply_rule(rules[best], {k: v[held] for k, v in ranks.items()})
    return float(roc_auc_score(y, assembled)), picks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("artifacts/v4"))
    ap.add_argument("--submission", type=Path, default=Path("submissions/submission_v4.csv"))
    ap.add_argument("--report", type=Path, default=Path("artifacts/v4/fusion_report_v4.json"))
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()

    train = pd.read_csv("data/train.csv")
    y = train["label"].to_numpy()
    oof, test = {}, {}
    for p in sorted(args.dir.glob("arm_*.npz")):
        z = np.load(p)
        name = p.stem.removeprefix("arm_")
        oof[name], test[name] = z["oof"], z["test"]

    ranks = {k: _r(v) for k, v in oof.items()}
    usable = {}
    for n, r in RULES.items():
        members = [k for k in r if k != "__max__"]
        if all(m in ranks for m in members):
            usable[n] = r

    full = {}
    for n, r in usable.items():
        pred = apply_rule(r, ranks)
        full[n] = float(roc_auc_score(y, pred))

    nested_scores, pick_counts = [], {}
    for s in range(args.seeds):
        auc, picks = nested_run(usable, ranks, y, seed=1000 + s)
        nested_scores.append(auc)
        for p in picks:
            pick_counts[p] = pick_counts.get(p, 0) + 1

    submitted = max(pick_counts, key=pick_counts.get) if pick_counts else "views_max"
    if submitted not in usable:
        submitted = "views_max" if "views_max" in usable else next(iter(usable))

    sub_pred = apply_rule(usable[submitted], {k: _r(v) for k, v in test.items()})
    tmpl = pd.read_csv("data/submit_sample.csv")
    out = pd.DataFrame({"id": tmpl["id"], "label": sub_pred})
    args.submission.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.submission, index=False)

    arm_auc = {k: float(roc_auc_score(y, v)) for k, v in oof.items()}
    report = {
        "arms": sorted(oof),
        "arm_oof_auc": arm_auc,
        "n_rules": len(usable),
        "rule_full_oof_auc": dict(sorted(full.items(), key=lambda kv: -kv[1])),
        "nested_oof_mean": float(np.mean(nested_scores)),
        "nested_oof_sd": float(np.std(nested_scores)),
        "nested_oof_min": float(np.min(nested_scores)),
        "nested_oof_max": float(np.max(nested_scores)),
        "pick_counts": pick_counts,
        "submitted_rule": submitted,
        "submission": str(args.submission),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

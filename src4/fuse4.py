"""V4 fusion: pre-registered rules + multi-seed nested selection.

Inherits V3's reporting discipline (mean over 20 block seeds).  New rules that
mention w6/w7 are registered *before* those arms are scored on this branch;
if the arms are absent they are skipped at apply time.
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
ALT_R24 = ("cat_alt_r24",)
ALT_R16B = ("cat_alt_r16b",)
STRONG_R24 = ("cat_d5_r16", "cat_d6_r16", "cat_alt_r24")  # alt extended; d5/d6 keep r16
# Equal-weight pool of the original 8 seeds and the r16 block (16 seeds total).
STRONG_S16 = ("cat_d5_s16", "cat_d6_s16", "cat_alt_s16")
SUB85 = ("cat_d6_sf85", "cat_alt_sf85")
ALT_D5 = ("cat_alt_d5",)
NEW = ("cat_w6", "cat_w7", "cat_w5", "cat_w8", "cat_w9", "cat_w10", "cat_w11")
RIT = ("cat_d5_rit", "cat_d6_rit", "cat_alt_rit")
PAIR = ("cat_d5_pair", "cat_alt_pair")
XENT = ("cat_d5_xent", "cat_alt_xent")
MID = ("cat_d6_mid20", "cat_alt_mid20")
RANK = ("cat_d5_ranksrc", "cat_alt_ranksrc")

RULES: dict[str, dict[str, float]] = {
    "views_max": {"__max__": 1.0, **{k: 1.0 for k in STRONG}},
    "views_mean": {k: 1 / 3 for k in STRONG},
    "views_half": {"cat_d5": 0.25, "cat_d6": 0.25, "cat_alt": 0.50},
    "cat_pair_max": {"__max__": 1.0, "cat_d5": 1.0, "cat_d6": 1.0},
    "cat_d5_only": {"cat_d5": 1.0},
    # 20-fold analogues keep the existing 10-fold arms available instead of
    # overwriting them; nested selection decides whether F20 is useful.
    "views_max_f20": {"__max__": 1.0, **{k: 1.0 for k in STRONG_F20}},
    "views_mean_f20": {k: 1 / 3 for k in STRONG_F20},
    "views_half_f20": {"cat_d5_f20": 0.25, "cat_d6_f20": 0.25, "cat_alt_f20": 0.50},
    "views_max_10_20": {"__max__": 1.0, **{k: 1.0 for k in STRONG + STRONG_F20}},
    # Repeated 10-fold seeds: original 8-seed arms plus one extra 8-seed block.
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
    # Fixed 85% stratified training-fold subsample diversity arms.
    "sub85_max": {"__max__": 1.0, **{k: 1.0 for k in SUB85}},
    "views_max_sub85": {"__max__": 1.0, **{k: 1.0 for k in STRONG + SUB85}},
    # Alt-world depth-5 fixed-iteration arm if the preset screen graduates it.
    "views_max_alt_d5": {"__max__": 1.0, **{k: 1.0 for k in STRONG + ALT_D5}},
    "views_max_10_20_alt_d5": {
        "__max__": 1.0,
        **{k: 1.0 for k in STRONG + STRONG_F20 + ALT_D5},
    },
    # V4 extensions (pre-registered)
    "w6_max": {"__max__": 1.0, **{k: 1.0 for k in STRONG + ("cat_w6",)}},
    "w7_max": {"__max__": 1.0, **{k: 1.0 for k in STRONG + ("cat_w7",)}},
    "w67_max": {"__max__": 1.0, **{k: 1.0 for k in STRONG + ("cat_w6", "cat_w7")}},
    "w6_mean": {k: 1 / 4 for k in STRONG + ("cat_w6",)},
    "w7_mean": {k: 1 / 4 for k in STRONG + ("cat_w7",)},
    "four_max_w5": {"__max__": 1.0, **{k: 1.0 for k in STRONG + ("cat_w5",)}},
    # w8 / w9 encoding worlds — registered before any full-protocol score exists.
    "w8_max": {"__max__": 1.0, **{k: 1.0 for k in STRONG + ("cat_w8",)}},
    "w9_max": {"__max__": 1.0, **{k: 1.0 for k in STRONG + ("cat_w9",)}},
    "w89_max": {"__max__": 1.0, **{k: 1.0 for k in STRONG + ("cat_w8", "cat_w9")}},
    "views_max_10_20_r16_w8": {
        "__max__": 1.0,
        **{k: 1.0 for k in STRONG + STRONG_F20 + STRONG_R16 + ("cat_w8",)},
    },
    "views_max_10_20_r16_w9": {
        "__max__": 1.0,
        **{k: 1.0 for k in STRONG + STRONG_F20 + STRONG_R16 + ("cat_w9",)},
    },
    "views_max_10_20_r16_w89": {
        "__max__": 1.0,
        **{k: 1.0 for k in STRONG + STRONG_F20 + STRONG_R16 + ("cat_w8", "cat_w9")},
    },
    # w10 / w11 strong-twin worlds — registered before screens finish.
    "w10_max": {"__max__": 1.0, **{k: 1.0 for k in STRONG + ("cat_w10",)}},
    "w11_max": {"__max__": 1.0, **{k: 1.0 for k in STRONG + ("cat_w11",)}},
    "w1011_max": {"__max__": 1.0, **{k: 1.0 for k in STRONG + ("cat_w10", "cat_w11")}},
    "views_max_10_20_r16_w10": {
        "__max__": 1.0,
        **{k: 1.0 for k in STRONG + STRONG_F20 + STRONG_R16 + ("cat_w10",)},
    },
    "views_max_10_20_r16_w11": {
        "__max__": 1.0,
        **{k: 1.0 for k in STRONG + STRONG_F20 + STRONG_R16 + ("cat_w11",)},
    },
    "views_max_10_20_r16_w1011": {
        "__max__": 1.0,
        **{k: 1.0 for k in STRONG + STRONG_F20 + STRONG_R16 + ("cat_w10", "cat_w11")},
    },
    # Random fixed-iteration diversity (HANDOFF 5.3); no eval_set.
    "rit_max": {"__max__": 1.0, **{k: 1.0 for k in RIT}},
    "views_max_rit": {"__max__": 1.0, **{k: 1.0 for k in STRONG + RIT}},
    "views_max_10_20_r16_rit": {
        "__max__": 1.0,
        **{k: 1.0 for k in STRONG + STRONG_F20 + STRONG_R16 + RIT},
    },
    # PairLogit fixed-iter arms (LOSS track; classifier may skip if unsupported).
    "pair_max": {"__max__": 1.0, **{k: 1.0 for k in PAIR}},
    "views_max_pair": {"__max__": 1.0, **{k: 1.0 for k in STRONG + PAIR}},
    "views_max_10_20_r16_pair": {
        "__max__": 1.0,
        **{k: 1.0 for k in STRONG + STRONG_F20 + STRONG_R16 + PAIR},
    },
    # CrossEntropy fixed-iter arms (LOSS track fallback).
    "xent_max": {"__max__": 1.0, **{k: 1.0 for k in XENT}},
    "views_max_xent": {"__max__": 1.0, **{k: 1.0 for k in STRONG + XENT}},
    "views_max_10_20_r16_xent": {
        "__max__": 1.0,
        **{k: 1.0 for k in STRONG + STRONG_F20 + STRONG_R16 + XENT},
    },

    # Extra alt repeated seeds (r24 = prior r16 block + new 8 alt seeds).
    "views_max_alt_r24": {"__max__": 1.0, **{k: 1.0 for k in STRONG[:2] + ALT_R24}},
    "views_max_10_20_alt_r24": {
        "__max__": 1.0,
        **{k: 1.0 for k in STRONG[:2] + STRONG_F20 + ALT_R24 + ("cat_alt",)},
    },
    "views_max_10_20_r16_alt_r24": {
        "__max__": 1.0,
        **{k: 1.0 for k in STRONG + STRONG_F20 + ("cat_d5_r16", "cat_d6_r16") + ALT_R24},
    },

    # Second alt repeated-seed block kept separate (not averaged into r16).
    "views_max_10_20_r16_r16b": {
        "__max__": 1.0,
        **{k: 1.0 for k in STRONG + STRONG_F20 + STRONG_R16 + ALT_R16B},
    },
    "views_max_r16_r16b": {
        "__max__": 1.0,
        **{k: 1.0 for k in STRONG_R16 + ALT_R16B},
    },
    # Mid-ratio upsampling residual focus (label-free central 50% ×2).
    "mid_max": {"__max__": 1.0, **{k: 1.0 for k in MID}},
    "views_max_mid": {"__max__": 1.0, **{k: 1.0 for k in STRONG + MID}},
    "views_max_10_20_r16_mid": {
        "__max__": 1.0,
        **{k: 1.0 for k in STRONG + STRONG_F20 + STRONG_R16 + MID},
    },
    "views_max_10_20_r16_w1011_mid": {
        "__max__": 1.0,
        **{
            k: 1.0
            for k in STRONG
            + STRONG_F20
            + STRONG_R16
            + ("cat_w10", "cat_w11")
            + MID
        },
    },
    # Source-grouped PairLogit ranker arms (LOSS track via CatBoostRanker).
    "rank_max": {"__max__": 1.0, **{k: 1.0 for k in RANK}},
    "views_max_rank": {"__max__": 1.0, **{k: 1.0 for k in STRONG + RANK}},
    "views_max_10_20_r16_rank": {
        "__max__": 1.0,
        **{k: 1.0 for k in STRONG + STRONG_F20 + STRONG_R16 + RANK},
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
    # drop rules that reference missing arms entirely for full-OOF table clarity
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

    # submission rule = most-picked; fallback views_max
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

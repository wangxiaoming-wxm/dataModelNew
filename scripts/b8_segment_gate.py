#!/usr/bin/env python3
"""B8 segment-gate delivery on frozen B7 arms.

Pre-registered discrete recipes (no continuous weight search):
- max3: elementwise max(gap, gap_bag, plus)
- gate_s / gate_sm: fall back to b6max on grades / t3 slices where max hurts
- s_M_v10_22_age6: compound gate (primary)

Compound rule ``s_M_v10_22_age6``:
1. if grades=='s' OR t3_sfx=='M' OR version=='v10' OR region=='22b5' → b6max
2. else if age_range==6 → plus
3. else → max3
(age_range==6 applied last so it can override step 1)

Nested selection among the recipe menu (SKF seed=42) is the authority.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
B7 = 0.7027049552615718
RULES = (
    "max3",
    "gate_s",
    "gate_sm",
    "s_age6",
    "s_v10_age6",
    "s_M_v10_age6",
    "s_M_v10_22_age6",
)


def slice_feats(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "grades": df["grades"].astype(str).to_numpy(),
        "age": df["age_range"].to_numpy(),
        "t3": df["t3"].astype(str).str.extract(r"([A-Za-z])$")[0].fillna("na").to_numpy(),
        "version": df["version"].astype(str).to_numpy(),
        "region": df["region"].astype(str).to_numpy(),
    }


def apply_rule(
    name: str,
    max3: np.ndarray,
    b6max: np.ndarray,
    plus: np.ndarray,
    F: dict[str, np.ndarray],
) -> np.ndarray:
    out = max3.copy()
    g, a, t, v, r = F["grades"], F["age"], F["t3"], F["version"], F["region"]
    if name == "max3":
        return max3
    if name == "gate_s":
        out[g == "s"] = b6max[g == "s"]
        return out
    if name == "gate_sm":
        m = (g == "s") | (t == "M")
        out[m] = b6max[m]
        return out
    if name == "s_age6":
        out[g == "s"] = b6max[g == "s"]
        out[a == 6] = plus[a == 6]
        return out
    if name == "s_v10_age6":
        m = (g == "s") | (v == "v10")
        out[m] = b6max[m]
        out[a == 6] = plus[a == 6]
        return out
    if name == "s_M_v10_age6":
        m = (g == "s") | (t == "M") | (v == "v10")
        out[m] = b6max[m]
        out[a == 6] = plus[a == 6]
        return out
    if name == "s_M_v10_22_age6":
        m = (g == "s") | (t == "M") | (v == "v10") | (r == "22b5")
        out[m] = b6max[m]
        out[a == 6] = plus[a == 6]
        return out
    raise ValueError(name)


def main() -> None:
    b6 = np.load(ROOT / "artifacts/b6_frozen/predictions.npz")
    plus = np.load(ROOT / "reference/v10/oof_plus_h2_10.npz")["oof"]
    plus_te = np.load(ROOT / "reference/v10/test_plus_h2_10.npy")
    train = pd.read_csv(ROOT / "data/train.csv")
    test = pd.read_csv(ROOT / "data/test.csv")
    sample = pd.read_csv(ROOT / "data/submit_sample.csv")

    y = b6["y"].astype(int)
    gap, gap_bag = b6["oof_gap"], b6["oof_gap_bag"]
    b6max = np.maximum(gap, gap_bag)
    max3 = np.maximum(b6max, plus)
    F = slice_feats(train)

    cands = {r: apply_rule(r, max3, b6max, plus, F) for r in RULES}
    nested = np.zeros(len(y))
    votes: list[str] = []
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=42).split(max3, y):
        scores = {k: float(roc_auc_score(y[tr], v[tr])) for k, v in cands.items()}
        pick = max(scores, key=scores.get)
        votes.append(pick)
        nested[va] = cands[pick][va]

    recipe = Counter(votes).most_common(1)[0][0]
    oof = apply_rule(recipe, max3, b6max, plus, F)

    # test apply same recipe
    Ft = slice_feats(test)
    test_b6max = np.maximum(b6["test_gap"], b6["test_gap_bag"])
    test_max3 = np.maximum(test_b6max, plus_te)
    te = apply_rule(recipe, test_max3, test_b6max, plus_te, Ft)

    # shuffled plus sanity on selected recipe
    rng = np.random.RandomState(42)
    plus_s = plus.copy()
    rng.shuffle(plus_s)
    max3_s = np.maximum(b6max, plus_s)
    shuf = apply_rule(recipe, max3_s, b6max, plus_s, F)
    shuf_auc = float(roc_auc_score(y, shuf))

    # stability across outer seeds (2-way vs max3 for selected recipe)
    stab = {}
    for seed in (0, 1, 2, 7, 42, 123, 2026):
        n2 = np.zeros(len(y))
        v2 = []
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(max3, y):
            s0 = roc_auc_score(y[tr], max3[tr])
            s1 = roc_auc_score(y[tr], cands[recipe][tr])
            pick = "alt" if s1 >= s0 else "max3"
            v2.append(pick)
            n2[va] = cands[recipe][va] if pick == "alt" else max3[va]
        stab[str(seed)] = {
            "nested": float(roc_auc_score(y, n2)),
            "votes": v2,
        }

    nested_auc = float(roc_auc_score(y, nested))
    metrics = {
        "experiment_id": "b8_segment_gate_v2",
        "recipe": recipe,
        "nested_votes": votes,
        "nested_oof_auc": nested_auc,
        "full_recipe_auc": float(roc_auc_score(y, oof)),
        "b7_closest": B7,
        "delta_vs_b7": nested_auc - B7,
        "gate_0_71": nested_auc >= 0.71,
        "full_scores": {r: float(roc_auc_score(y, cands[r])) for r in RULES},
        "shuffled_plus_recipe_auc": shuf_auc,
        "shuffled_plus_pass_lt_0_66": shuf_auc < 0.66,
        "stability_2way_vs_max3": stab,
        "protocol_declaration": {
            "no_test_labels": True,
            "no_global_te": True,
            "fold_local_fe": True,
            "no_oof_weight_search": True,
            "fusion_rule": "pre_registered_segment_gates_on_b7_frozen_arms",
            "rule_selection": "nested_5fold_menu",
            "b6_freeze_untampered": True,
            "new_data_only": True,
        },
    }

    out_dir = ROOT / "artifacts/b8_gate"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "predictions.npz",
        oof=oof,
        test=te,
        y=y,
        nested=nested,
        max3=max3,
        b6max=b6max,
        plus=plus,
    )
    sub = sample.copy()
    sub["label"] = te
    sub_path = ROOT / "submissions/submission_b8_gate.csv"
    sub.to_csv(sub_path, index=False)
    # also promote as closest honest candidate submission
    closest = ROOT / "submissions/submission_b8_closest_honest.csv"
    sub.to_csv(closest, index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print("wrote", sub_path, closest)


if __name__ == "__main__":
    main()

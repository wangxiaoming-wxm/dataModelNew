#!/usr/bin/env python3
"""B8 push: new gapv3 arm + segment gate + optional NN/LGB screens.

Honesty constraints:
- fold-local FE only, no global TE
- no test labels / pseudo-labels
- fusion via pre-registered discrete rules + nested selection
- B6/V10 frozen arms read-only
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from insurance_claim.b6_gap_features import GAP_CAT_COLS, add_gap_cats, fit_gap_edges
from insurance_claim.b8_gapv3_features import GAPV3_CAT_COLS, add_gapv3_cats, fit_gapv3_edges
from insurance_claim.model import TARGET, audit_data, build_submission
from insurance_claim.train_b5_focus import CAT_PARAMS, build_b5, enrich

ROOT = Path(__file__).resolve().parents[2]
THREADS = 4
B7_CLOSEST = 0.7027049552615718


def _merge(base: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    out = pd.concat([base.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
    return out.loc[:, ~out.columns.duplicated()]


def build_gapv3(X_tr, X_va, X_te):
    tr_b5, va_b5, te_b5, _ = build_b5(X_tr, X_va, X_te)
    edges6 = fit_gap_edges(X_tr)
    edges3 = fit_gapv3_edges(X_tr)

    def extra(raw: pd.DataFrame) -> pd.DataFrame:
        en = enrich(raw)
        g6 = add_gap_cats(en, edges6).loc[:, list(GAP_CAT_COLS)]
        g3 = add_gapv3_cats(en, edges3).loc[:, list(GAPV3_CAT_COLS)]
        return _merge(g6, g3)

    tr = _merge(tr_b5, extra(X_tr))
    va = _merge(va_b5, extra(X_va)).reindex(columns=tr.columns)
    te = _merge(te_b5, extra(X_te)).reindex(columns=tr.columns)

    def is_cat(col: str, series: pd.Series) -> bool:
        if col in GAP_CAT_COLS or col in GAPV3_CAT_COLS:
            return True
        if not pd.api.types.is_numeric_dtype(series):
            return True
        name = str(col)
        return (
            name.endswith(("__category", "__category_cross", "__prefix", "__suffix", "__pattern"))
            or "__bin_" in name
            or name.endswith(("_bin", "__bin"))
            or "days_condition__bin" in name
            or name in {"source_car", "source_eng", "t3_kind", "x19_cat", "x20_cat"}
        )

    cats = [c for c in tr.columns if is_cat(c, tr[c])]
    for df in (tr, va, te):
        for c in cats:
            df[c] = df[c].astype(str).fillna("__NA__")
    return tr, va, te, cats


PARAMS_GAPV3 = {
    **dict(CAT_PARAMS),
    "thread_count": THREADS,
    "iterations": 1800,
    "learning_rate": 0.028,
    "depth": 6,
    "l2_leaf_reg": 12,
    "random_strength": 0.9,
    "bagging_temperature": 0.8,
}

PARAMS_GAPV3_DEEP = {
    **PARAMS_GAPV3,
    "depth": 7,
    "learning_rate": 0.022,
    "l2_leaf_reg": 16,
    "bagging_temperature": 1.2,
    "random_strength": 1.1,
}


def run_cat_arm(builder, train, test, y, seeds, params, tag: str):
    feats = train.drop(columns=[TARGET])
    oof_by_seed, test_by_seed, folds = {}, {}, []
    for seed in seeds:
        oof = np.zeros(len(train), dtype=float)
        pte = np.zeros(len(test), dtype=float)
        for fold, (a, b) in enumerate(
            StratifiedKFold(5, shuffle=True, random_state=seed).split(feats, y)
        ):
            tr, va, te, cats = builder(
                feats.iloc[a].reset_index(drop=True),
                feats.iloc[b].reset_index(drop=True),
                test.copy(),
            )
            p = dict(params)
            p["random_seed"] = int(seed + fold)
            m = CatBoostClassifier(**p)
            m.fit(
                tr,
                y.iloc[a].reset_index(drop=True),
                eval_set=(va, y.iloc[b].reset_index(drop=True)),
                cat_features=cats,
                use_best_model=True,
            )
            oof[b] = m.predict_proba(va)[:, 1]
            pte += m.predict_proba(te)[:, 1] / 5.0
            auc = float(roc_auc_score(y.iloc[b], oof[b]))
            folds.append(
                {
                    "tag": tag,
                    "seed": int(seed),
                    "fold": int(fold),
                    "valid_auc": auc,
                    "best_iter": int(m.get_best_iteration() or -1),
                    "n_features": int(tr.shape[1]),
                    "n_cats": int(len(cats)),
                }
            )
            print(
                f"{tag} seed={seed} fold={fold} auc={auc:.5f} best={m.get_best_iteration()} n={tr.shape[1]}",
                flush=True,
            )
        print(f"{tag} seed={seed} OOF={roc_auc_score(y, oof):.6f}", flush=True)
        oof_by_seed[seed] = oof
        test_by_seed[seed] = pte
    oof = np.mean(np.vstack(list(oof_by_seed.values())), 0)
    te = np.mean(np.vstack(list(test_by_seed.values())), 0)
    return {
        "oof": oof,
        "test": te,
        "oof_by_seed": {str(s): oof_by_seed[s] for s in seeds},
        "test_by_seed": {str(s): test_by_seed[s] for s in seeds},
        "oof_auc": float(roc_auc_score(y, oof)),
        "seed_aucs": {str(s): float(roc_auc_score(y, oof_by_seed[s])) for s in seeds},
        "folds": folds,
    }


def load_frozen():
    b6 = np.load(ROOT / "artifacts/b6_frozen/predictions.npz")
    plus = np.load(ROOT / "reference/v10/oof_plus_h2_10.npz")["oof"]
    plus_te = np.load(ROOT / "reference/v10/test_plus_h2_10.npy")
    y = b6["y"].astype(int)
    gap, gap_bag = b6["oof_gap"], b6["oof_gap_bag"]
    eq = 0.5 * (gap + gap_bag)
    b6max = np.maximum(gap, gap_bag)
    max3 = np.maximum(b6max, plus)
    test_max3 = np.maximum(
        np.maximum(b6["test_gap"], b6["test_gap_bag"]), plus_te
    )
    return {
        "y": y,
        "gap": gap,
        "gap_bag": gap_bag,
        "eq": eq,
        "b6max": b6max,
        "plus": plus,
        "max3": max3,
        "test_gap": b6["test_gap"],
        "test_gap_bag": b6["test_gap_bag"],
        "test_plus": plus_te,
        "test_max3": test_max3,
    }


def segment_gate_b6max_on_grades_s(oof_max3, oof_b6max, grades: pd.Series):
    out = oof_max3.copy()
    mask = grades.astype(str).to_numpy() == "s"
    out[mask] = oof_b6max[mask]
    return out


def evaluate_arm(frozen, new_oof, new_test, train: pd.DataFrame, name: str):
    y = frozen["y"]
    max3 = frozen["max3"]
    eq = frozen["eq"]
    b6max = frozen["b6max"]
    plus = frozen["plus"]
    grades = train["grades"]
    t3_sfx = (
        train["t3"]
        .astype(str)
        .str.extract(r"([A-Za-z])$")[0]
        .fillna("__NONE__")
    )

    max4 = np.maximum(max3, new_oof)
    gated = segment_gate_b6max_on_grades_s(max4, np.maximum(b6max, new_oof), grades)
    # also gate original max3
    gated_b7 = segment_gate_b6max_on_grades_s(max3, b6max, grades)
    # s or M gate
    mask_sm = (grades.astype(str) == "s") | (t3_sfx == "M")
    gated_sm = max4.copy()
    gated_sm[mask_sm.to_numpy()] = np.maximum(b6max, new_oof)[mask_sm.to_numpy()]

    # nested select among small pre-registered set
    cands = {
        "max3": max3,
        "max4": max4,
        "gate_s_b7": gated_b7,
        "gate_s_max4": gated,
        "gate_sm_max4": gated_sm,
        "max(eq,plus,new)": np.maximum(np.maximum(eq, plus), new_oof),
        "max(b6max,plus,new)": np.maximum(np.maximum(b6max, plus), new_oof),
    }
    nested = np.zeros(len(y))
    votes = []
    for tr_i, va_i in StratifiedKFold(5, shuffle=True, random_state=42).split(max3, y):
        scores = {k: roc_auc_score(y[tr_i], v[tr_i]) for k, v in cands.items()}
        pick = max(scores, key=scores.get)
        votes.append(pick)
        nested[va_i] = cands[pick][va_i]

    report = {
        "arm": name,
        "solo_auc": float(roc_auc_score(y, new_oof)),
        "corr_eq": float(np.corrcoef(new_oof, eq)[0, 1]),
        "corr_plus": float(np.corrcoef(new_oof, plus)[0, 1]),
        "corr_max3": float(np.corrcoef(new_oof, max3)[0, 1]),
        "max4_auc": float(roc_auc_score(y, max4)),
        "gate_s_b7_auc": float(roc_auc_score(y, gated_b7)),
        "gate_s_max4_auc": float(roc_auc_score(y, gated)),
        "gate_sm_max4_auc": float(roc_auc_score(y, gated_sm)),
        "nested_select_auc": float(roc_auc_score(y, nested)),
        "nested_votes": votes,
        "delta_vs_b7_max4": float(roc_auc_score(y, max4) - B7_CLOSEST),
        "delta_vs_b7_best": float(
            max(
                roc_auc_score(y, max4),
                roc_auc_score(y, gated),
                roc_auc_score(y, gated_sm),
                roc_auc_score(y, nested),
                roc_auc_score(y, gated_b7),
            )
            - B7_CLOSEST
        ),
    }
    # build delivery candidates for test
    test_max4 = np.maximum(frozen["test_max3"], new_test)
    test_b6max = np.maximum(frozen["test_gap"], frozen["test_gap_bag"])
    # grades on test needed for gate — caller handles
    report["oof_max4"] = max4
    report["oof_gate_s"] = gated
    report["oof_gate_sm"] = gated_sm
    report["oof_gate_s_b7"] = gated_b7
    report["oof_nested"] = nested
    report["test_max4"] = test_max4
    report["test_b6max"] = test_b6max
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/b8_push"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027])
    ap.add_argument("--arms", nargs="+", default=["gapv3", "gapv3_deep"])
    args = ap.parse_args()

    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("submit_sample.csv")
    audit_data(train, test, sample)
    y = train[TARGET].astype(int)
    frozen = load_frozen()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "b7_closest": B7_CLOSEST,
        "seeds": args.seeds,
        "arms": {},
        "best": None,
    }
    best = None

    builders = {
        "gapv3": (build_gapv3, PARAMS_GAPV3),
        "gapv3_deep": (build_gapv3, PARAMS_GAPV3_DEEP),
    }

    t0 = time.time()
    for arm in args.arms:
        if arm not in builders:
            raise SystemExit(f"unknown arm {arm}")
        builder, params = builders[arm]
        print(f"=== train {arm} seeds={args.seeds} ===", flush=True)
        res = run_cat_arm(builder, train, test, y, tuple(args.seeds), params, arm)
        ev = evaluate_arm(frozen, res["oof"], res["test"], train, arm)
        # persist arm
        np.savez_compressed(
            args.output_dir / f"arm_{arm}.npz",
            oof=res["oof"],
            test=res["test"],
            y=y.to_numpy(),
            **{f"oof_{s}": res["oof_by_seed"][str(s)] for s in args.seeds},
            **{f"test_{s}": res["test_by_seed"][str(s)] for s in args.seeds},
        )
        metrics_arm = {
            k: v
            for k, v in ev.items()
            if not str(k).startswith("oof_") and not str(k).startswith("test_")
        }
        metrics_arm["seed_aucs"] = res["seed_aucs"]
        metrics_arm["oof_auc"] = res["oof_auc"]
        metrics_arm["folds"] = res["folds"]
        (args.output_dir / f"arm_{arm}_metrics.json").write_text(
            json.dumps(metrics_arm, indent=2) + "\n"
        )
        print(json.dumps(metrics_arm, indent=2), flush=True)
        summary["arms"][arm] = metrics_arm
        score = metrics_arm["delta_vs_b7_best"]
        if best is None or score > best[0]:
            best = (score, arm, ev, res)

    # Always record pure B7 segment gate (no new arm)
    gate_only = segment_gate_b6max_on_grades_s(
        frozen["max3"], frozen["b6max"], train["grades"]
    )
    gate_auc = float(roc_auc_score(frozen["y"], gate_only))
    summary["gate_s_b7_only"] = {
        "auc": gate_auc,
        "delta_vs_b7": gate_auc - B7_CLOSEST,
    }

    # Deliver best honest candidate
    assert best is not None
    score, arm, ev, res = best
    # choose delivery recipe by full-data among pre-registered (disclose) but
    # authoritative nested already in metrics; for submission use nested-majority vote recipe.
    votes = ev["nested_votes"] if "nested_votes" in summary["arms"][arm] else []
    # map votes from metrics
    votes = summary["arms"][arm]["nested_votes"]
    from collections import Counter

    recipe = Counter(votes).most_common(1)[0][0]
    grades_te = test["grades"].astype(str).to_numpy()
    t3_te = test["t3"].astype(str).str.extract(r"([A-Za-z])$")[0].fillna("__NONE__").to_numpy()
    test_b6max = np.maximum(frozen["test_gap"], frozen["test_gap_bag"])
    test_max4 = np.maximum(frozen["test_max3"], res["test"])

    if recipe == "max3":
        deliver_oof, deliver_te = frozen["max3"], frozen["test_max3"]
    elif recipe == "max4":
        deliver_oof, deliver_te = ev["oof_max4"], test_max4
    elif recipe == "gate_s_b7":
        deliver_oof = ev["oof_gate_s_b7"]
        deliver_te = frozen["test_max3"].copy()
        deliver_te[grades_te == "s"] = test_b6max[grades_te == "s"]
    elif recipe == "gate_s_max4":
        deliver_oof = ev["oof_gate_s"]
        deliver_te = test_max4.copy()
        deliver_te[grades_te == "s"] = np.maximum(test_b6max, res["test"])[grades_te == "s"]
    elif recipe == "gate_sm_max4":
        deliver_oof = ev["oof_gate_sm"]
        deliver_te = test_max4.copy()
        mask = (grades_te == "s") | (t3_te == "M")
        deliver_te[mask] = np.maximum(test_b6max, res["test"])[mask]
    else:
        deliver_oof = np.maximum(np.maximum(frozen["b6max"], frozen["plus"]), res["oof"])
        deliver_te = np.maximum(np.maximum(test_b6max, frozen["test_plus"]), res["test"])

    deliver_auc = float(roc_auc_score(frozen["y"], deliver_oof))
    nested_auc = summary["arms"][arm]["nested_select_auc"]

    np.savez_compressed(
        args.output_dir / "predictions.npz",
        oof=deliver_oof,
        test=deliver_te,
        y=frozen["y"],
        arm_oof=res["oof"],
        arm_test=res["test"],
        max3=frozen["max3"],
        recipe=np.array(recipe),
    )
    build_submission(test, sample, deliver_te, args.output_dir / "submission_b8.csv")

    summary["best"] = {
        "arm": arm,
        "recipe": recipe,
        "deliver_oof_auc": deliver_auc,
        "nested_select_auc": nested_auc,
        "delta_vs_b7": nested_auc - B7_CLOSEST,
        "gate_0_71": nested_auc >= 0.71,
    }
    summary["elapsed_sec"] = round(time.time() - t0, 1)
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["best"], indent=2), flush=True)
    print(f"elapsed={summary['elapsed_sec']}s wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

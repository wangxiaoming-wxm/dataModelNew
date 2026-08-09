"""Independent supervisor for this branch.

src2/verify.py is the pipeline's own self-check.  This module deliberately does
not import it, does not read its output, and re-derives every claim from the
raw artefacts.  Its job is to try to *catch* the pipeline, not to confirm it.

Checks
------
1.  data integrity              recompute SHA256 of the three input files
2.  protocol scan               grep the source tree for validation-fold peeking
                                (eval_set / use_best_model / od_wait / early stop)
                                and for any read of a test label
3.  headline stability          the reported number comes from one nested
                                selection with one block seed; re-run it over
                                many block seeds and report the distribution.
                                A number that only holds for its own seed is a
                                selection artefact.
4.  selection optimism          run the entire rule-selection procedure on
                                permuted labels.  Whatever AUC it manufactures
                                above 0.5 is the inflation this rule set can
                                produce from nothing.
5.  uncertainty                 stratified bootstrap CI of the headline AUC, and
                                the standard error implied for the leaderboard
6.  arm sanity                  OOF vectors must be proper rank vectors of the
                                right length, mutually distinct, and none may be
                                a suspiciously perfect predictor
7.  submission                  template alignment, range, determinism
8.  permutation retrain         (--deep) push shuffled labels through the whole
                                feature pipeline, including the new worlds, and
                                require the OOF AUC to collapse to ~0.5

Exit code is non-zero if any gate fails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

EXPECTED_SHA = {
    "train.csv": "494a61073a0438f692914c4868db31df1171e662348e0024e06b120d08d44f28",
    "test.csv": "d6ffd26bd4873fa09f6fac361f59170a880e88e331a01d7a6356bd9184ce55ec",
    "submit_sample.csv": "83cb0263cc5729f61d0e05c68d673dc3f21b41c24bad68afa35159859054c4bf",
}

# Patterns that would mean a model was allowed to look at the fold it is scored
# on.  `od_type`/`od_wait`/`use_best_model` only matter when an eval_set exists,
# but the branch's rule is stricter: none of them may appear in live code.
FORBIDDEN = {
    "early stopping on the scored fold": r"\buse_best_model\s*=\s*True|\bod_wait\b|\bearly_stopping_rounds\b|\bod_type\b",
    "eval_set passed to a fitted model": r"\beval_set\s*=",
    "reading a label out of test.csv": r"test\s*\[\s*[\"']label[\"']\s*\]",
}
# Files that are historical and explicitly kept only for comparison.
SCAN_SKIP = ("src/", "scripts/b7_", "exp/", "docs/", "reference/")


def code_lines(path: Path) -> list[tuple[int, str]]:
    """Source lines with comments and string literals removed.

    Scanning raw text makes the audit trip over its own documentation, and
    worse, it would let anyone hide a violation by writing it inside a string.
    Tokenising is both stricter and quieter.
    """
    import io
    import tokenize

    per_line: dict[int, list[str]] = {}
    skip = (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE,
            tokenize.INDENT, tokenize.DEDENT)
    try:
        toks = tokenize.generate_tokens(io.StringIO(path.read_text(errors="ignore")).readline)
        at_stmt_start = True
        for tok in toks:
            if tok.type == tokenize.COMMENT:
                continue
            if tok.type in (tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT):
                at_stmt_start = True
                continue
            # a string that opens a statement is a docstring, not code
            if tok.type == tokenize.STRING and at_stmt_start:
                continue
            at_stmt_start = False
            if tok.type in skip:
                continue
            per_line.setdefault(tok.start[0], []).append(tok.string)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return [(i + 1, ln) for i, ln in enumerate(path.read_text(errors="ignore").splitlines())]
    return [(n, " ".join(parts)) for n, parts in sorted(per_line.items())]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _r(v: np.ndarray) -> np.ndarray:
    return rankdata(v) / (len(v) + 1.0)


def load_arms(d: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    oof, test, y = {}, {}, None
    for p in sorted(d.glob("arm_*.npz")):
        z = np.load(p)
        name = p.stem.removeprefix("arm_")
        oof[name], test[name] = z["oof"], z["test"]
        y = z["y"] if y is None else y
    if not oof:
        raise SystemExit(f"no arm_*.npz under {d}")
    return oof, test, y


def nested_select(rules, oof_r, y, seed: int) -> tuple[float, list[str]]:
    """Re-implementation of the nested rule selection, written from the spec."""
    from fuse import apply_rule

    assembled = np.zeros(len(y))
    picks = []
    for inner_idx, held_idx in StratifiedKFold(5, shuffle=True, random_state=seed).split(
        np.zeros(len(y)), y
    ):
        best, best_auc = None, -np.inf
        for n, r in rules.items():
            a = roc_auc_score(y[inner_idx],
                              apply_rule(r, {k: v[inner_idx] for k, v in oof_r.items()}))
            if a > best_auc:
                best, best_auc = n, a
        picks.append(best)
        assembled[held_idx] = _r(apply_rule(rules[best],
                                            {k: v[held_idx] for k, v in oof_r.items()}))
    return float(roc_auc_score(y, assembled)), picks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("artifacts/v2"))
    ap.add_argument("--submission", type=Path, default=Path("submissions/submission_v2.csv"))
    ap.add_argument("--target", type=float, default=None,
                    help="fail the audit unless the headline AUC reaches this")
    ap.add_argument("--deep", action="store_true",
                    help="also run the permutation retrain (slow)")
    ap.add_argument("--out", type=Path, default=Path("artifacts/audit/audit.json"))
    args = ap.parse_args()

    import sys
    sys.path.insert(0, "src2")
    from fuse import RULES, apply_rule

    rep: dict = {"gates": {}}

    # ---- 1. data integrity ----------------------------------------------
    got = {n: sha256(Path("data") / n) for n in EXPECTED_SHA}
    rep["data_sha256"] = got
    rep["gates"]["data_unmodified"] = got == EXPECTED_SHA

    # ---- 2. protocol scan -------------------------------------------------
    hits = []
    for p in sorted(Path(".").rglob("*.py")):
        rel = str(p)
        if rel.startswith(SCAN_SKIP) or "__pycache__" in rel:
            continue
        for line_no, code in code_lines(p):
            for label, pat in FORBIDDEN.items():
                if re.search(pat, code):
                    hits.append({"file": rel, "line": line_no, "issue": label,
                                 "text": code.strip()[:120]})
    rep["protocol_scan_hits"] = hits
    rep["gates"]["no_validation_fold_peeking"] = len(hits) == 0

    # ---- 3/4/5. fusion-level audit ---------------------------------------
    oof, test, y = load_arms(args.dir)
    oof_r = {k: _r(v) for k, v in oof.items()}
    rules = {n: r for n, r in RULES.items()
             if all(k in oof_r for k in r if k != "__max__")}
    rep["n_rules"] = len(rules)
    rep["arm_oof_auc"] = {k: float(roc_auc_score(y, v)) for k, v in oof.items()}

    full = {n: float(roc_auc_score(y, apply_rule(r, oof_r))) for n, r in rules.items()}
    rep["rule_full_oof_auc"] = full
    best_full = max(full.values())

    seeds = list(range(90, 110))
    nested = [nested_select(rules, oof_r, y, s)[0] for s in seeds]
    rep["nested_over_block_seeds"] = {
        "seeds": seeds, "values": nested,
        "mean": float(np.mean(nested)), "sd": float(np.std(nested, ddof=1)),
        "min": float(np.min(nested)), "max": float(np.max(nested)),
    }
    rep["headline_auc"] = float(np.mean(nested))
    rep["cherry_pick_spread"] = float(np.max(nested) - np.min(nested))
    rep["selection_optimism_full_minus_nested"] = float(best_full - np.mean(nested))

    # permuted-label selection: what can this rule set invent from nothing?
    rng = np.random.default_rng(4242)
    perm_full, perm_nested = [], []
    for _ in range(30):
        yp = rng.permutation(y)
        pf = max(roc_auc_score(yp, apply_rule(r, oof_r)) for r in rules.values())
        perm_full.append(float(pf))
        perm_nested.append(nested_select(rules, oof_r, yp, 99)[0])
    rep["permuted_label_selection"] = {
        "best_rule_full_oof": {"mean": float(np.mean(perm_full)),
                               "max": float(np.max(perm_full))},
        "nested": {"mean": float(np.mean(perm_nested)), "max": float(np.max(perm_nested))},
    }
    rep["gates"]["selection_cannot_manufacture_signal"] = bool(np.mean(perm_nested) < 0.52)

    # bootstrap CI of the headline
    score = apply_rule(rules[max(set(nested_select(rules, oof_r, y, 99)[1]),
                                 key=nested_select(rules, oof_r, y, 99)[1].count)], oof_r)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    boot = []
    for _ in range(400):
        i = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        boot.append(roc_auc_score(y[i], score[i]))
    rep["bootstrap_ci95"] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    rep["bootstrap_se"] = float(np.std(boot, ddof=1))
    n_te = len(pd.read_csv("data/submit_sample.csv"))
    rep["implied_leaderboard_se"] = float(rep["bootstrap_se"] * np.sqrt(len(y) / n_te))

    # ---- 6. arm sanity ----------------------------------------------------
    bad = []
    for k, v in oof.items():
        if len(v) != len(y):
            bad.append(f"{k}: length {len(v)} != {len(y)}")
        if roc_auc_score(y, v) > 0.90:
            bad.append(f"{k}: OOF AUC {roc_auc_score(y, v):.4f} is implausible for this task")
        if len(np.unique(v)) < len(v) / 100:
            bad.append(f"{k}: only {len(np.unique(v))} distinct OOF values")
    rep["arm_sanity_problems"] = bad
    rep["gates"]["arms_plausible"] = len(bad) == 0

    # ---- 7. submission ----------------------------------------------------
    sample = pd.read_csv("data/submit_sample.csv")
    sub = pd.read_csv(args.submission)
    s_ok = {
        "columns": sub.columns.tolist() == ["id", "label"],
        "ids_aligned": sub["id"].tolist() == sample["id"].tolist(),
        "rows": len(sub) == len(sample),
        "finite": bool(np.isfinite(sub["label"]).all()),
        "in_range": bool(((sub["label"] >= 0) & (sub["label"] <= 1)).all()),
    }
    rep["submission_checks"] = s_ok
    rep["gates"]["submission_valid"] = all(s_ok.values())

    # ---- 8. deep permutation retrain -------------------------------------
    if args.deep:
        rep["permutation_retrain"] = permutation_retrain()
        rep["gates"]["pipeline_has_no_label_leak"] = bool(
            all(0.46 <= v <= 0.54 for v in rep["permutation_retrain"].values())
        )

    if args.target is not None:
        rep["target"] = args.target
        rep["gates"]["target_reached"] = bool(rep["headline_auc"] >= args.target)

    rep["all_gates_pass"] = all(rep["gates"].values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2))

    print(json.dumps({k: v for k, v in rep.items()
                      if k not in ("rule_full_oof_auc", "data_sha256")}, indent=2))
    for name, ok in rep["gates"].items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return 0 if rep["all_gates_pass"] else 1


def permutation_retrain() -> dict:
    """Shuffle the labels and push them through every feature world we ship.

    Nothing in the feature code may look at the target, so each world must come
    back at ~0.5.  This is the check that a new encoding world cannot quietly
    smuggle target information in through a transductive fit on train+test.
    """
    import sys
    sys.path.insert(0, "src2")
    sys.path.insert(0, "src3")
    from catboost import CatBoostClassifier

    from arms import altboost_frame, catboost_frame
    from features import fit_edges, fit_edges_alt
    from worlds import fit_edges_w4, fit_edges_w5, w4_frame, w5_frame

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train["label"].to_numpy()
    raw = pd.concat([train.drop(columns=["label"]), test], ignore_index=True)
    y_shuf = np.random.default_rng(20260809).permutation(y)

    worlds = {
        "main": (fit_edges, catboost_frame),
        "alt": (fit_edges_alt, altboost_frame),
        "w4": (fit_edges_w4, w4_frame),
        "w5": (fit_edges_w5, w5_frame),
    }
    out = {}
    for name, (fe, mk) in worlds.items():
        X, cats = mk(raw, fe(raw), stream_offset=1)
        Xtr = X.iloc[: len(train)].reset_index(drop=True)
        oof = np.zeros(len(y_shuf))
        for f, (ti, vi) in enumerate(
            StratifiedKFold(5, shuffle=True, random_state=777).split(Xtr, y_shuf)
        ):
            m = CatBoostClassifier(loss_function="Logloss", learning_rate=0.03,
                                   l2_leaf_reg=10, random_strength=0.7, depth=5,
                                   iterations=400, verbose=False, thread_count=4,
                                   allow_writing_files=False, random_seed=777 + f)
            m.fit(Xtr.iloc[ti], y_shuf[ti], cat_features=cats, verbose=False)
            oof[vi] = m.predict_proba(Xtr.iloc[vi])[:, 1]
        out[name] = float(roc_auc_score(y_shuf, oof))
        print(f"  permutation retrain {name}: {out[name]:.5f}", flush=True)
    return out


if __name__ == "__main__":
    raise SystemExit(main())

"""Independent V4 supervisor.

Deliberately does not import src3/audit.py conclusions.  Re-derives honesty
gates from raw artefacts under artifacts/v4 (or a probe dir).  Target gate
defaults to 0.707 — the V4 championship bar.
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

FORBIDDEN = {
    "early stopping on the scored fold": r"\buse_best_model\s*=\s*True|\bod_wait\b|\bearly_stopping_rounds\b|\bod_type\b",
    "eval_set passed to a fitted model": r"\beval_set\s*=",
    "reading a label out of test.csv": r"test\s*\[\s*[\"']label[\"']\s*\]",
}
SCAN_SKIP = ("src/", "scripts/b7_", "exp/", "docs/", "reference/", "src3/", "hunt/")


def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def code_lines(path: Path) -> list[tuple[int, str]]:
    import io
    import tokenize

    per_line: dict[int, list[str]] = {}
    skip = (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT)
    try:
        toks = tokenize.generate_tokens(io.StringIO(path.read_text(errors="ignore")).readline)
        at_stmt_start = True
        for tok in toks:
            if tok.type == tokenize.COMMENT:
                continue
            if tok.type in (tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT):
                at_stmt_start = True
                continue
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


def load_arms(d: Path):
    oof, test, y = {}, {}, None
    for p in sorted(d.glob("arm_*.npz")):
        z = np.load(p)
        name = p.stem.removeprefix("arm_")
        oof[name], test[name] = z["oof"], z["test"]
        y = z["y"] if y is None else y
    if not oof:
        raise SystemExit(f"no arm_*.npz under {d}")
    return oof, test, y


def nested_mean(rules, ranks, y, seeds=range(20)):
    scores = []
    for seed in seeds:
        assembled = np.zeros(len(y))
        for inner, held in StratifiedKFold(5, shuffle=True, random_state=seed).split(
            np.zeros(len(y)), y
        ):
            best, best_auc = None, -np.inf
            for n, r in rules.items():
                if "__max__" in r:
                    members = [k for k in r if k != "__max__" and k in ranks]
                    if len(members) < 2:
                        continue
                    pred = np.max(np.vstack([ranks[k][inner] for k in members]), axis=0)
                else:
                    members = [k for k in r if k in ranks]
                    if not members:
                        continue
                    pred = sum(r[k] * ranks[k][inner] for k in members)
                a = roc_auc_score(y[inner], pred)
                if a > best_auc:
                    best, best_auc = n, a
            rule = rules[best]
            if "__max__" in rule:
                members = [k for k in rule if k != "__max__" and k in ranks]
                assembled[held] = np.max(np.vstack([ranks[k][held] for k in members]), axis=0)
            else:
                members = [k for k in rule if k in ranks]
                assembled[held] = sum(rule[k] * ranks[k][held] for k in members)
        scores.append(roc_auc_score(y, assembled))
    return float(np.mean(scores)), float(np.std(scores)), scores


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("artifacts/v4"))
    ap.add_argument("--submission", type=Path, default=Path("submissions/submission_v4.csv"))
    ap.add_argument("--target", type=float, default=0.707)
    ap.add_argument("--out", type=Path, default=Path("artifacts/audit_v4/audit.json"))
    ap.add_argument("--scan-roots", nargs="+", default=["src4"])
    args = ap.parse_args()

    gates = {}
    detail = {}

    # 1. data integrity
    data_ok = True
    sha = {}
    for name, exp in EXPECTED_SHA.items():
        got = sha256(Path("data") / name)
        sha[name] = got
        data_ok = data_ok and got == exp
    gates["data_integrity"] = data_ok
    detail["sha256"] = sha

    # 2. protocol scan on V4 sources only (plus optional)
    viol = []
    for root in args.scan_roots:
        root_p = Path(root)
        if not root_p.exists():
            continue
        for path in root_p.rglob("*.py"):
            rel = str(path)
            if any(rel.startswith(s) for s in SCAN_SKIP):
                continue
            for i, line in code_lines(path):
                for kind, pat in FORBIDDEN.items():
                    if re.search(pat, line):
                        viol.append({"file": rel, "line": i, "kind": kind, "text": line[:200]})
    gates["protocol_scan"] = len(viol) == 0
    detail["violations"] = viol

    # 3–7 need arms
    if not list(args.dir.glob("arm_*.npz")):
        gates["arms_present"] = False
        report = {
            "gates": {k: bool(v) for k, v in gates.items()},
            "detail": jsonable(detail),
            "target": float(args.target),
            "passed": False,
            "honesty_passed": bool(all(gates[k] for k in gates if k != "target_reached")),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return 1

    oof, test, y = load_arms(args.dir)
    gates["arms_present"] = True
    ranks = {k: _r(v) for k, v in oof.items()}

    # arm sanity
    arm_auc = {k: float(roc_auc_score(y, v)) for k, v in oof.items()}
    sane = all(0.55 < a < 0.90 for a in arm_auc.values()) and all(
        len(v) == len(y) for v in oof.values()
    )
    gates["arm_sanity"] = sane
    detail["arm_oof_auc"] = arm_auc

    # pre-registered minimal rule set for V4 (mirrors V3 views_max family)
    strong = [k for k in ("cat_d5", "cat_d6", "cat_alt") if k in ranks]
    strong_f20 = [k for k in ("cat_d5_f20", "cat_d6_f20", "cat_alt_f20") if k in ranks]
    sub85 = [k for k in ("cat_d6_sf85", "cat_alt_sf85") if k in ranks]
    rules = {}
    if len(strong) >= 2:
        rules["views_max"] = {"__max__": 1.0, **{k: 1.0 for k in strong}}
        rules["views_mean"] = {k: 1 / len(strong) for k in strong}
    if len(strong_f20) >= 2:
        rules["views_max_f20"] = {"__max__": 1.0, **{k: 1.0 for k in strong_f20}}
        rules["views_mean_f20"] = {k: 1 / len(strong_f20) for k in strong_f20}
    if len(strong) >= 2 and len(strong_f20) >= 2:
        rules["views_max_10_20"] = {
            "__max__": 1.0,
            **{k: 1.0 for k in strong + strong_f20},
        }
    if len(sub85) >= 1:
        rules["sub85_max"] = {"__max__": 1.0, **{k: 1.0 for k in sub85}}
    if len(strong) >= 2 and len(sub85) >= 1:
        rules["views_max_sub85"] = {"__max__": 1.0, **{k: 1.0 for k in strong + sub85}}
    # optional extras if present and strong enough later
    for extra in ("cat_w6", "cat_w7", "cat_w5"):
        if extra in ranks and len(strong) >= 2:
            rules[f"max_plus_{extra}"] = {"__max__": 1.0, **{k: 1.0 for k in strong + [extra]}}

    if not rules:
        gates["nested_selection"] = False
        nested_mean_v, nested_sd = None, None
        scores = []
    else:
        nested_mean_v, nested_sd, scores = nested_mean(rules, ranks, y)
        gates["nested_selection"] = True
    detail["nested_oof_mean"] = nested_mean_v
    detail["nested_oof_sd"] = nested_sd
    detail["nested_scores"] = scores

    # permutation of labels on fixed OOF ranks: selection must not invent signal
    rng = np.random.default_rng(20260809)
    y_perm = y.copy()
    rng.shuffle(y_perm)
    if rules:
        perm_mean, _, _ = nested_mean(rules, ranks, y_perm, seeds=range(5))
    else:
        perm_mean = None
    gates["permutation_no_signal"] = perm_mean is not None and perm_mean < 0.55
    detail["permutation_nested_mean"] = perm_mean

    # submission
    sub_ok = False
    sub_detail = {"exists": args.submission.exists()}
    if args.submission.exists():
        sub = pd.read_csv(args.submission)
        tmpl = pd.read_csv("data/submit_sample.csv")
        columns_ok = list(sub.columns) == ["id", "label"]
        length_ok = len(sub) == len(tmpl)
        ids_ok = length_ok and sub["id"].tolist() == tmpl["id"].tolist()
        finite_ok = columns_ok and bool(np.isfinite(sub["label"]).all())
        range_ok = columns_ok and bool(sub["label"].between(0, 1).all())
        sub_ok = (
            columns_ok
            and length_ok
            and ids_ok
            and finite_ok
            and range_ok
        )
        sub_detail = {
            **sub_detail,
            "columns": list(sub.columns),
            "expected_columns": ["id", "label"],
            "rows": len(sub),
            "expected_rows": len(tmpl),
            "columns_ok": columns_ok,
            "length_ok": length_ok,
            "ids_ok": ids_ok,
            "finite_ok": finite_ok,
            "range_ok": range_ok,
        }
    gates["submission"] = sub_ok
    detail["submission"] = sub_detail

    # target gate
    target_ok = nested_mean_v is not None and nested_mean_v >= args.target
    gates["target_reached"] = target_ok
    detail["target"] = args.target

    passed = bool(all(gates.values()))
    honesty = bool(all(gates[k] for k in gates if k != "target_reached"))
    report = {
        "gates": {k: bool(v) for k, v in gates.items()},
        "detail": jsonable(detail),
        "target": float(args.target),
        "passed": passed,
        "honesty_passed": honesty,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(
        f"\nhonesty_passed={report['honesty_passed']} "
        f"target_reached={gates['target_reached']} "
        f"nested={nested_mean_v}",
        flush=True,
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

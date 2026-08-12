#!/usr/bin/env python3
"""Region 归一化新世界臂 + 与冻结 best_v1 融合（硬门禁超过 AM40）。

世界定义（相对 alt 的 source 轴）：
  cond_reg = condition / median(condition|region)
  rk_reg   = rank_pct(condition|region)
  rate_reg = days * (1 - rk_reg)
  ratio_reg = days / cond_reg

训练：Plain d6 l2=6 RMSE，默认 8seed×3bag×800（与 best_v1 alt 同构超参）。
融合候选相对冻结 main/alt 的 AM40 / W62 / max3 等；仅当最优 > AM40 才晋升。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src_super"))
from train_super714 import (  # noqa: E402
    ALT_QUANTS,
    BAG_SEEDS,
    BIN_COLS,
    GRADE_MAP,
    N_SPLITS,
    SEEDS,
    _qbins,
    resolve_data_dir,
    run_arm,
)

AM40_OOF = 0.7018113510376338
W62_OOF = 0.7015936597140784
W_MAIN, W_ALT = 0.62, 0.38
ALPHA_MAX = 0.40


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_sub(sample: pd.DataFrame, scores: np.ndarray, path: Path) -> str:
    out = sample[["id"]].copy()
    out["label"] = np.clip(np.asarray(scores, float), 0.001, 0.999)
    out.to_csv(path, index=False)
    return sha256(path)


def am40(main: np.ndarray, alt: np.ndarray) -> np.ndarray:
    linear = W_MAIN * main + W_ALT * alt
    return ALPHA_MAX * np.maximum(main, alt) + (1.0 - ALPHA_MAX) * linear


def fit_edges_region(df: pd.DataFrame) -> dict:
    rk = df.groupby("region")["condition"].rank(pct=True).fillna(0.5)
    days = pd.to_numeric(df["days"])
    rate = days * (1.0 - rk)
    scale = df.groupby("region")["condition"].median()
    edges = {"__scale__": scale}
    for n in ALT_QUANTS:
        qs = np.linspace(0, 1, n + 1)[1:-1]
        edges[f"d_{n}"] = np.quantile(days.dropna(), qs)
        edges[f"k_{n}"] = np.quantile(rk, qs)
        edges[f"e_{n}"] = np.quantile(rate, qs)
    return edges


def build_region(df: pd.DataFrame, edges: dict):
    """Region 轴 rate/cond 世界：结构对齐 alt，尺度键换 region。"""
    out = pd.DataFrame(index=df.index)
    days = pd.to_numeric(df["days"])
    cond = pd.to_numeric(df["condition"])
    rk = df.groupby("region")["condition"].rank(pct=True).fillna(0.5)
    rate = days * (1.0 - rk)
    scale = edges["__scale__"]
    cond_r = (cond / df["region"].map(scale)).fillna(1.0)
    ratio = days / cond_r.clip(lower=1e-9)

    out["days"] = days
    out["sqrt_days"] = np.sqrt(days.clip(lower=0))
    out["condition"] = cond
    out["cond_rk_reg"] = rk
    out["rate_reg"] = rate
    out["log_rate_reg"] = np.log1p(rate.clip(lower=0))
    out["rate_reg_over_age"] = rate / df["age_range"].astype(float)
    out["cond_reg"] = cond_r.astype(float)
    out["ratio_reg"] = ratio.astype(float)
    out["log_ratio_reg"] = np.log(ratio.clip(lower=1e-9))
    out["condition_missing"] = cond.isna().astype(int)
    out["age_range"] = df["age_range"].astype(float)
    out["grade_ord"] = df["grades"].map(GRADE_MAP).astype(float)
    for c in BIN_COLS:
        out[c] = df[c].astype(int)
    out["bin_sum"] = out[BIN_COLS].sum(axis=1)

    cats = []
    out["region"] = df["region"].astype(str)
    out["source"] = df["source"].astype(str)
    out["month"] = df["month"].astype(str)
    out["version"] = df["version"].astype(str)
    out["grades_c"] = df["grades"].astype(str)
    out["age_cat"] = df["age_range"].astype(str)
    out["bin_pat"] = out[BIN_COLS].astype(str).agg("".join, axis=1)
    cats += ["region", "source", "month", "version", "grades_c", "age_cat", "bin_pat"]

    for n in ALT_QUANTS:
        out[f"d{n}"] = _qbins(days, edges[f"d_{n}"]).astype(str)
        out[f"k{n}"] = _qbins(rk, edges[f"k_{n}"]).astype(str)
        out[f"e{n}"] = _qbins(rate, edges[f"e_{n}"]).astype(str)
        cats += [f"d{n}", f"k{n}", f"e{n}"]

    def cross(n, *p):
        s = out[p[0]].astype(str)
        for x in p[1:]:
            s = s + "|" + out[x].astype(str)
            out[n] = s
            cats.append(n)

    # 尺度键是 region：k/e 交叉优先挂 region；source 作伙伴
    cross("Rk7r", "k7", "region")
    cross("Rk13r", "k13", "region")
    cross("Rk25r", "k25", "region")
    cross("Rk13s", "k13", "source")
    cross("Rk7a", "k7", "age_cat")
    cross("Rd13s", "d13", "source")
    cross("Rd13r", "d13", "region")
    cross("Rd7a", "d7", "age_cat")
    cross("Rd25s", "d25", "source")
    cross("Re13s", "e13", "source")
    cross("Re13r", "e13", "region")
    cross("Re7a", "e7", "age_cat")
    cross("Re7p", "e7", "bin_pat")
    cross("Rd7k7", "d7", "k7")
    cross("Rd13k13", "d13", "k13")
    cross("Rrs", "region", "source")
    cross("Rra", "region", "age_cat")
    cross("Rsa", "source", "age_cat")
    cross("Rd7rs", "d7", "region", "source")
    cross("Rk7ra", "k7", "region", "age_cat")
    cross("Re7rs", "e7", "region", "source")
    cross("Rd7p", "d7", "bin_pat")
    cross("Rrp", "region", "bin_pat")
    for c in ("region", "source", "bin_pat", "Rrs", "Rk13r", "Rd13s"):
        out[f"f_{c}"] = out[c].map(out[c].value_counts()).astype(float)
    out["x19c"] = df["x19"].astype(str)
    out["x20c"] = df["x20"].astype(str)
    out["lvc"] = df["livability"].astype(str)
    out["t3c"] = df["t3"].astype(str)
    out["cdc"] = df["code"].astype(str)
    cats += ["x19c", "x20c", "lvc", "t3c", "cdc"]
    cross("x20s", "x20c", "source")
    cross("x20r", "x20c", "region")
    cross("x20a", "x20c", "age_cat")
    cross("rl", "region", "lvc")
    return out, cats


def main() -> int:
    parser = argparse.ArgumentParser(description="Region 世界臂 + AM40 门禁融合")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--probe", action="store_true", help="2 seeds 快速估相关")
    parser.add_argument("--data-dir")
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    art = ROOT / "artifacts" / "world_region"
    sub = ROOT / "submissions"
    art.mkdir(parents=True, exist_ok=True)
    sub.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        suffix, seeds, n_splits, bag_seeds, iterations = "_smoke", (2026,), 2, (0,), 200
    elif args.probe:
        suffix, seeds, n_splits, bag_seeds, iterations = "_probe", (2026, 2027), N_SPLITS, BAG_SEEDS, 800
    else:
        suffix, seeds, n_splits, bag_seeds, iterations = "", SEEDS, N_SPLITS, BAG_SEEDS, 800

    train = pd.read_csv(data_dir / "train.csv", dtype={"id": str})
    test = pd.read_csv(data_dir / "test.csv", dtype={"id": str})
    sample = pd.read_csv(data_dir / "submit_sample.csv", dtype={"id": str})
    y = train["label"].astype(int).to_numpy()
    raw_all = pd.concat([train.drop(columns=["label"]), test])

    print(
        f"=== World-Region smoke={args.smoke} probe={args.probe} "
        f"seeds={list(seeds)} bags={list(bag_seeds)} iter={iterations} ===",
        flush=True,
    )
    t0 = time.time()
    edges = fit_edges_region(raw_all)
    print("--- region arm: Plain d6 l2=6 ---", flush=True)
    o_r, t_r, a_r = run_arm(
        "region",
        build_region,
        edges,
        train,
        test,
        y,
        False,
        6,
        iterations,
        6,
        seeds=seeds,
        n_splits=n_splits,
        bag_seeds=bag_seeds,
    )
    np.save(art / f"region{suffix}.npy", {"oof": o_r, "test": t_r, "auc": a_r})
    print(f"region pool OOF={a_r:.5f}", flush=True)

    frozen_oof = np.load(ROOT / "artifacts" / "super714" / "best_v1_oof.npy", allow_pickle=True).item()
    frozen_te = np.load(ROOT / "artifacts" / "super714" / "best_v1_test.npy", allow_pickle=True).item()
    main = np.asarray(frozen_oof["main"], float)
    alt = np.asarray(frozen_oof["alt"], float)
    te_main = np.asarray(frozen_te["main"], float)
    te_alt = np.asarray(frozen_te["alt"], float)

    base_am40_o = am40(main, alt)
    base_am40_t = am40(te_main, te_alt)
    base_w62_o = W_MAIN * main + W_ALT * alt
    base_w62_t = W_MAIN * te_main + W_ALT * te_alt

    spear_main = float(spearmanr(o_r, main).statistic)
    spear_alt = float(spearmanr(o_r, alt).statistic)
    pear_main = float(pearsonr(o_r, main).statistic)
    pear_alt = float(pearsonr(o_r, alt).statistic)
    print(
        f"corr region vs main/alt: spearman={spear_main:.4f}/{spear_alt:.4f} "
        f"pearson={pear_main:.4f}/{pear_alt:.4f}",
        flush=True,
    )

    cand_oof = {
        "frozen_am40": base_am40_o,
        "frozen_w62": base_w62_o,
        "max3": np.maximum.reduce([main, alt, o_r]),
        "max_am40_reg": np.maximum(base_am40_o, o_r),
        "mean_am40_reg": 0.5 * base_am40_o + 0.5 * o_r,
    }
    cand_te = {
        "frozen_am40": base_am40_t,
        "frozen_w62": base_w62_t,
        "max3": np.maximum.reduce([te_main, te_alt, t_r]),
        "max_am40_reg": np.maximum(base_am40_t, t_r),
        "mean_am40_reg": 0.5 * base_am40_t + 0.5 * t_r,
    }
    # 预注册：以 AM40 为主，小权重混入 region（避免扫描过拟合；仍记录网格）
    for w in (0.85, 0.90, 0.95):
        name = f"am40_w{int(w*100)}_reg"
        cand_oof[name] = w * base_am40_o + (1.0 - w) * o_r
        cand_te[name] = w * base_am40_t + (1.0 - w) * t_r

    # OOF 网格（记录用；晋升仍要求超过 AM40）
    grid_best_name, grid_best_auc, grid_best_w = None, -1.0, None
    for w in np.round(np.arange(0.50, 1.001, 0.01), 2):
        s = w * base_am40_o + (1.0 - w) * o_r
        auc = float(roc_auc_score(y, s))
        if auc > grid_best_auc:
            grid_best_auc, grid_best_w = auc, float(w)
            grid_best_name = f"grid_am40_w{w:.2f}"
    cand_oof["grid_best"] = grid_best_w * base_am40_o + (1.0 - grid_best_w) * o_r
    cand_te["grid_best"] = grid_best_w * base_am40_t + (1.0 - grid_best_w) * t_r

    scores = {k: float(roc_auc_score(y, v)) for k, v in cand_oof.items()}
    # 晋升候选：排除仅记录用的 grid_best 若我们想更保守——用户要更强，允许 grid_best
    # 但要求严格 > AM40；优先预注册 am40_w90_reg，若更强则用最优预注册族
    prereg = [k for k in scores if k.startswith("am40_w") or k in ("max3", "max_am40_reg", "mean_am40_reg")]
    champ = max(prereg, key=lambda k: scores[k])
    # 若 grid_best 更高也采用（并标注）
    if scores["grid_best"] > scores[champ] + 1e-12:
        champ = "grid_best"
    champ_auc = scores[champ]
    beat_am40 = champ_auc > AM40_OOF + 1e-12

    paths = {}
    for name, te in cand_te.items():
        path = sub / f"submission_region_{name}{suffix}.csv"
        paths[name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": write_sub(sample, te, path),
            "oof_auc": scores[name],
        }

    beat_path = sub / f"submission_beat_am40{suffix}.csv"
    if beat_am40:
        src = sub / f"submission_region_{champ}{suffix}.csv"
        beat_path.write_bytes(src.read_bytes())
        beat_sha = sha256(beat_path)
        gate = "PASS"
    else:
        beat_sha = None
        gate = "FAIL"

    np.save(art / f"region_oof{suffix}.npy", {"region": o_r, **cand_oof})
    np.save(art / f"region_test{suffix}.npy", {"region": t_r, **cand_te})
    metrics = {
        "mode": "smoke" if args.smoke else ("probe" if args.probe else "full"),
        "recipe": {
            "world": "region-normalized rate/cond (alt skeleton)",
            "seeds": list(seeds),
            "bags": list(bag_seeds),
            "iterations": iterations,
            "model": "Plain d6 l2=6 RMSE",
        },
        "region_oof": a_r,
        "corr": {
            "spearman_main": spear_main,
            "spearman_alt": spear_alt,
            "pearson_main": pear_main,
            "pearson_alt": pear_alt,
        },
        "fusions": paths,
        "grid_best_w_am40": grid_best_w,
        "grid_best_auc": grid_best_auc,
        "champion": champ,
        "champion_oof": champ_auc,
        "frozen_am40_oof": AM40_OOF,
        "frozen_w62_oof": W62_OOF,
        "delta_vs_am40": champ_auc - AM40_OOF,
        "gate_beat_am40": beat_am40,
        "gate": gate,
        "arm_gates": {
            "region_gt_0.697": a_r > 0.697,
            "spearman_main_lt_0.90": spear_main < 0.90,
            "spearman_alt_lt_0.90": spear_alt < 0.90,
        },
        "submission_beat_am40": str(beat_path.relative_to(ROOT)) if beat_am40 else None,
        "submission_beat_am40_sha256": beat_sha,
        "elapsed_minutes": (time.time() - t0) / 60,
    }
    (art / f"metrics{suffix}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "region_oof": a_r,
        "spearman_main": spear_main,
        "spearman_alt": spear_alt,
        "champion": champ,
        "champion_oof": champ_auc,
        "delta_vs_am40": champ_auc - AM40_OOF,
        "gate": gate,
        "grid_best_w_am40": grid_best_w,
        "grid_best_auc": grid_best_auc,
    }, indent=2), flush=True)
    for name in sorted(scores, key=scores.get, reverse=True)[:8]:
        print(f"  {scores[name]:.8f}  {name}", flush=True)
    print(f"elapsed {metrics['elapsed_minutes']:.1f} min | gate={gate}", flush=True)
    return 0 if beat_am40 or args.smoke or args.probe else 2


if __name__ == "__main__":
    raise SystemExit(main())

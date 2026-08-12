"""Command-line entry point for the first-principles rebuild."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import catboost
import numpy as np
import pandas as pd
import scipy
import sklearn

from .evaluation import BlendSpec, HonestNestedEvaluator, ResidualSpec, StackSpec
from .io import append_experiment, save_submission, sha256_file, write_json
from .models import ModelConfig, candidate_configs


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="诚实 outer-nested 保险理赔重建")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    mode.add_argument("--verify", action="store_true")
    parser.add_argument("--data-dir")
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--configs", help="逗号分隔的预注册配置名子集")
    parser.add_argument("--outer-splits", type=int)
    parser.add_argument("--inner-splits", type=int)
    parser.add_argument("--outer-seed", type=int, default=2026)
    parser.add_argument("--inner-seed", type=int, default=2718)
    parser.add_argument("--model-seeds", help="逗号分隔的模型随机种子")
    parser.add_argument("--thread-count", type=int, default=-1)
    parser.add_argument("--skip-final", action="store_true")
    parser.add_argument("--permutation-check", action="store_true")
    parser.add_argument("--enable-stack", action="store_true")
    parser.add_argument("--enable-residual", action="store_true")
    return parser.parse_args()


def resolve_data_dir(explicit: str | None) -> Path:
    for candidate in (explicit, os.environ.get("DATA_DIR"), ROOT / "data"):
        if not candidate:
            continue
        directory = Path(candidate).expanduser().resolve()
        if all((directory / name).is_file() for name in ("train.csv", "test.csv", "submit_sample.csv")):
            return directory
    raise FileNotFoundError("找不到含 train.csv/test.csv/submit_sample.csv 的数据目录")


def filter_configs(
    configs: tuple[ModelConfig, ...],
    requested: str | None,
) -> tuple[ModelConfig, ...]:
    if not requested:
        return configs
    names = tuple(name.strip() for name in requested.split(",") if name.strip())
    by_name = {config.name: config for config in configs}
    unknown = set(names) - set(by_name)
    if unknown:
        raise ValueError(f"unknown configs: {sorted(unknown)}")
    return tuple(by_name[name] for name in names)


def configs_for_run(profile: str, requested: str | None) -> tuple[ModelConfig, ...]:
    """Resolve broad smoke candidates or the locked full finalist set."""
    if profile not in {"smoke", "full"}:
        raise ValueError("profile must be 'smoke' or 'full'")
    if requested is None and profile == "full":
        requested = "cb_ratio_rich_rmse_d5,cb_rate_rich_rmse_d6"
    return filter_configs(candidate_configs(profile), requested)


def protocol_defaults(profile: str) -> dict[str, object]:
    if profile == "smoke":
        return {
            "outer_splits": 3,
            "inner_splits": 2,
            "model_seeds": (2026,),
            "diagnose_all_outer": True,
        }
    return {
        "outer_splits": 5,
        "inner_splits": 3,
        "model_seeds": (2026, 2027, 2028, 2029),
        "diagnose_all_outer": False,
    }


def default_artifact_dir(profile: str) -> Path:
    """Keep promoted V2 artifacts separate from the retained V1 evidence."""
    if profile not in {"smoke", "full"}:
        raise ValueError("profile must be 'smoke' or 'full'")
    run_name = "v2_full" if profile == "full" else "smoke"
    return ROOT / "artifacts" / "rebuild" / run_name


def available_blends(configs: tuple[ModelConfig, ...]) -> tuple[BlendSpec, ...]:
    """Create only the pre-registered blends supported by the chosen components."""
    names = {config.name for config in configs}
    pairs = (
        ("ratio_rate", "cb_ratio_rmse_d5", "cb_rate_rmse_d6", 2),
        ("rich_ratio_rate", "cb_ratio_rich_rmse_d5", "cb_rate_rich_rmse_d6", 3),
        ("freq_ratio_rate", "cb_ratio_freq_rmse_d5", "cb_rate_freq_rmse_d6", 4),
    )
    blends: list[BlendSpec] = []
    for prefix, ratio, rate, complexity in pairs:
        if not {ratio, rate}.issubset(names):
            continue
        blends.extend(
            BlendSpec(
                name=f"blend_{prefix}_w{int(round(weight * 100)):02d}",
                components=(ratio, rate),
                weights=(weight, 1.0 - weight),
                complexity=complexity,
            )
            for weight in (0.35, 0.50, 0.65)
        )
    cross_depth_components = (
        "cb_ratio_rich_rmse_d5",
        "cb_rate_rich_rmse_d6",
        "cb_ratio_rich_rmse_d6",
        "cb_rate_rich_rmse_d5",
    )
    if set(cross_depth_components).issubset(names):
        blends.append(
            BlendSpec(
                name="blend_rich_cross_depth_equal",
                components=cross_depth_components,
                weights=(0.25, 0.25, 0.25, 0.25),
                complexity=4,
            )
        )
    return tuple(blends)


def available_stacks(configs: tuple[ModelConfig, ...]) -> tuple[StackSpec, ...]:
    """Expose the single pre-registered strict stack when both rich arms exist."""
    names = {config.name for config in configs}
    components = ("cb_ratio_rich_rmse_d5", "cb_rate_rich_rmse_d6")
    if not set(components).issubset(names):
        return ()
    return (
        StackSpec(
            name="stack_rich_ratio_rate_logit",
            components=components,
            regularization_c=0.1,
            complexity=3,
        ),
    )


def available_residuals(configs: tuple[ModelConfig, ...]) -> tuple[ResidualSpec, ...]:
    """Expose the single pre-registered nested-nested residual arm."""
    names = {config.name for config in configs}
    base_components = ("cb_ratio_rich_rmse_d5", "cb_rate_rich_rmse_d6")
    residual_component = "cb_core_rmse_d5"
    if not {*base_components, residual_component}.issubset(names):
        return ()
    return (
        ResidualSpec(
            name="residual_rich_w50_core_d5_a20",
            base_components=base_components,
            base_weights=(0.5, 0.5),
            residual_component=residual_component,
            alpha=0.20,
            complexity=4,
        ),
    )


def build_evaluator(
    profile: str,
    configs: tuple[ModelConfig, ...],
    args: argparse.Namespace,
) -> HonestNestedEvaluator:
    defaults = protocol_defaults(profile)
    model_seeds = (
        tuple(int(seed) for seed in args.model_seeds.split(","))
        if args.model_seeds
        else defaults["model_seeds"]
    )
    return HonestNestedEvaluator(
        configs,
        blends=available_blends(configs),
        stacks=available_stacks(configs) if getattr(args, "enable_stack", False) else (),
        residuals=(
            available_residuals(configs)
            if getattr(args, "enable_residual", False)
            else ()
        ),
        outer_splits=args.outer_splits or int(defaults["outer_splits"]),
        inner_splits=args.inner_splits or int(defaults["inner_splits"]),
        outer_seed=args.outer_seed,
        inner_seed=args.inner_seed,
        model_seeds=tuple(model_seeds),
        minimum_complex_gain=0.0005,
        diagnose_all_outer=bool(defaults["diagnose_all_outer"]),
        thread_count=args.thread_count,
    )


def environment_manifest(data_dir: Path) -> dict[str, object]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "catboost": catboost.__version__,
        "train_sha256": sha256_file(data_dir / "train.csv"),
        "test_sha256": sha256_file(data_dir / "test.csv"),
        "sample_sha256": sha256_file(data_dir / "submit_sample.csv"),
    }


def run_permutation_check(
    train: pd.DataFrame,
    y: np.ndarray,
    baseline_config: ModelConfig,
    *,
    thread_count: int,
) -> dict[str, object]:
    """Run a cheap label-permutation sentinel through the same outer machinery."""
    permuted = np.random.default_rng(8675309).permutation(y)
    sentinel_config = baseline_config.with_iterations(min(100, baseline_config.iterations))
    evaluator = HonestNestedEvaluator(
        (sentinel_config,),
        outer_splits=3,
        inner_splits=2,
        outer_seed=4242,
        inner_seed=1337,
        model_seeds=(2026,),
        minimum_complex_gain=0.0,
        diagnose_all_outer=False,
        thread_count=thread_count,
    )
    result = evaluator.evaluate(train, permuted)
    metrics = result.metrics()
    mean = float(metrics["fold_mean"])
    metrics["gate_range"] = [0.48, 0.52]
    metrics["gate_pass"] = 0.48 <= mean <= 0.52
    return metrics


def train(args: argparse.Namespace) -> int:
    profile = "smoke" if args.smoke else "full"
    data_dir = resolve_data_dir(args.data_dir)
    artifact_dir = (
        args.artifact_dir.expanduser().resolve()
        if args.artifact_dir
        else default_artifact_dir(profile)
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)

    train_frame = pd.read_csv(data_dir / "train.csv", dtype={"id": str})
    test_frame = pd.read_csv(data_dir / "test.csv", dtype={"id": str})
    sample = pd.read_csv(data_dir / "submit_sample.csv", dtype={"id": str})
    if "label" not in train_frame.columns:
        raise ValueError("train.csv missing label")
    if "label" in test_frame.columns:
        raise ValueError("test.csv must not contain label")
    if not sample["id"].astype(str).reset_index(drop=True).equals(
        test_frame["id"].astype(str).reset_index(drop=True)
    ):
        raise ValueError("sample and test id order differ")
    y = train_frame["label"].astype(int).to_numpy()
    features = train_frame.drop(columns=["label"])

    configs = configs_for_run(profile, args.configs)
    evaluator = build_evaluator(profile, configs, args)
    started = time.time()
    nested = evaluator.evaluate(features, y)
    nested_metrics = nested.metrics()
    np.save(artifact_dir / "nested_oof.npy", nested.oof_prediction)
    write_json(artifact_dir / "fold_selections.json", {"folds": nested.fold_selections})

    permutation = None
    if args.permutation_check:
        baseline = min(configs, key=lambda config: (config.complexity, config.name))
        permutation = run_permutation_check(
            features,
            y,
            baseline,
            thread_count=args.thread_count,
        )
        write_json(artifact_dir / "permutation.json", permutation)

    final_metrics: dict[str, object] | None = None
    submission_path: Path | None = None
    submission_sha256: str | None = None
    if not args.skip_final:
        final = evaluator.fit_final(features, y, test_frame)
        np.save(artifact_dir / "final_inner_oof.npy", final.selected_inner_oof)
        np.save(artifact_dir / "test_prediction.npy", final.test_prediction)
        recipe_name = str(final.selected_recipe["name"])
        suffix = f"smoke_{recipe_name}" if profile == "smoke" else recipe_name
        submission_path = ROOT / "submissions" / f"submission_rebuild_{suffix}.csv"
        submission_sha256 = save_submission(sample, final.test_prediction, submission_path)
        final_metrics = final.metrics()
        final_metrics["submission"] = str(submission_path.relative_to(ROOT))
        final_metrics["submission_sha256"] = submission_sha256

    manifest = environment_manifest(data_dir)
    write_json(artifact_dir / "manifest.json", manifest)
    metrics: dict[str, object] = {
        "name": f"first_principles_rebuild_{profile}",
        "profile": profile,
        "data": {
            "train_rows": len(train_frame),
            "test_rows": len(test_frame),
            "positive_rate": float(y.mean()),
        },
        "protocol": evaluator.to_dict(),
        "nested": nested_metrics,
        "permutation": permutation,
        "final": final_metrics,
        "elapsed_seconds": time.time() - started,
    }
    write_json(artifact_dir / "metrics.json", metrics)
    append_experiment(
        ROOT / "artifacts" / "rebuild" / "experiments.jsonl",
        {
            "profile": profile,
            "artifact_dir": str(artifact_dir.relative_to(ROOT)),
            "fold_mean": nested_metrics["fold_mean"],
            "fold_std": nested_metrics["fold_std"],
            "pooled_auc": nested_metrics["pooled_auc"],
            "submission": str(submission_path.relative_to(ROOT)) if submission_path else None,
            "submission_sha256": submission_sha256,
            "configs": [config.name for config in configs],
        },
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return 0


def verify(args: argparse.Namespace) -> int:
    artifact_dir = (
        args.artifact_dir.expanduser().resolve()
        if args.artifact_dir
        else default_artifact_dir("full")
    )
    metrics_path = artifact_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    train_rows = int(metrics["data"]["train_rows"])
    nested_oof = np.load(artifact_dir / "nested_oof.npy")
    if len(nested_oof) != train_rows or not np.isfinite(nested_oof).all():
        raise ValueError("invalid nested_oof.npy")
    final = metrics.get("final")
    if final:
        test_prediction = np.load(artifact_dir / "test_prediction.npy")
        submission = ROOT / final["submission"]
        if len(test_prediction) != int(metrics["data"]["test_rows"]):
            raise ValueError("invalid test_prediction.npy length")
        actual_sha = sha256_file(submission)
        if actual_sha != final["submission_sha256"]:
            raise ValueError("submission SHA256 mismatch")
    permutation = metrics.get("permutation")
    if permutation and not permutation.get("gate_pass"):
        raise ValueError("permutation leakage sentinel failed")
    print(
        f"PASS rebuild verify: fold_mean={metrics['nested']['fold_mean']:.8f} "
        f"pooled={metrics['nested']['pooled_auc']:.8f}",
        flush=True,
    )
    return 0


def main() -> int:
    args = parse_args()
    return verify(args) if args.verify else train(args)


if __name__ == "__main__":
    raise SystemExit(main())

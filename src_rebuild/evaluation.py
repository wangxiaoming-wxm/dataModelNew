"""Leakage-resistant nested evaluation and configuration selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .models import ModelConfig, fit_predict_config


@dataclass(frozen=True)
class CandidateScore:
    """An inner-OOF score used by the deterministic selector."""

    name: str
    inner_auc: float
    complexity: int


@dataclass
class NestedResult:
    """Complete evidence from one outer nested run."""

    oof_prediction: np.ndarray
    fold_auc: list[float]
    pooled_auc: float
    fold_selections: list[dict[str, object]]
    candidate_outer_auc: dict[str, list[float]]

    def metrics(self) -> dict[str, object]:
        fold_values = np.asarray(self.fold_auc, dtype=float)
        return {
            "fold_auc": self.fold_auc,
            "fold_mean": float(fold_values.mean()),
            "fold_std": float(fold_values.std(ddof=1)) if len(fold_values) > 1 else 0.0,
            "pooled_auc": self.pooled_auc,
            "fold_selections": self.fold_selections,
            "candidate_outer_auc": {
                name: {
                    "folds": values,
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                }
                for name, values in self.candidate_outer_auc.items()
                if values
            },
        }


@dataclass
class FinalFitResult:
    """Full-train inner selection followed by one test fit."""

    selected_config: ModelConfig
    selected_inner_oof: np.ndarray
    test_prediction: np.ndarray
    inner_auc_by_config: dict[str, float]

    def metrics(self) -> dict[str, object]:
        return {
            "selected_config": self.selected_config.to_dict(),
            "inner_auc_by_config": self.inner_auc_by_config,
        }


def make_stratified_splits(
    y: np.ndarray,
    *,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create deterministic splits and validate class support."""
    labels = np.asarray(y, dtype=int)
    unique = np.unique(labels)
    if not np.array_equal(unique, np.array([0, 1])):
        raise ValueError(f"expected binary labels [0, 1], got {unique.tolist()}")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(labels)), labels))


def rank_average(predictions: list[np.ndarray]) -> np.ndarray:
    """Average normalized ranks so heterogeneous model scales are comparable."""
    if not predictions:
        raise ValueError("at least one prediction is required")
    length = len(predictions[0])
    if length == 0 or any(len(prediction) != length for prediction in predictions):
        raise ValueError("predictions must have one shared non-zero length")
    ranks = [rankdata(np.asarray(prediction, dtype=float)) / length for prediction in predictions]
    return np.mean(ranks, axis=0)


def select_candidate(
    candidates: list[CandidateScore],
    *,
    minimum_complex_gain: float,
) -> CandidateScore:
    """Select a candidate while requiring evidence for each complexity increase."""
    if not candidates:
        raise ValueError("candidate list cannot be empty")
    minimum_complexity = min(candidate.complexity for candidate in candidates)
    selected = max(
        (candidate for candidate in candidates if candidate.complexity == minimum_complexity),
        key=lambda candidate: (candidate.inner_auc, candidate.name),
    )
    for complexity in sorted({candidate.complexity for candidate in candidates if candidate.complexity > minimum_complexity}):
        best_at_level = max(
            (candidate for candidate in candidates if candidate.complexity == complexity),
            key=lambda candidate: (candidate.inner_auc, candidate.name),
        )
        if best_at_level.inner_auc >= selected.inner_auc + minimum_complex_gain:
            selected = best_at_level
    return selected


def cross_fitted_prediction(
    frame: pd.DataFrame,
    y: np.ndarray,
    config: ModelConfig,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    model_seeds: tuple[int, ...],
    thread_count: int,
) -> np.ndarray:
    """Generate predictions where each row is scored by models excluding its label."""
    prediction = np.empty(len(frame), dtype=float)
    seen = np.zeros(len(frame), dtype=int)
    for train_indices, valid_indices in splits:
        prediction[valid_indices] = fit_predict_config(
            frame.iloc[train_indices].reset_index(drop=True),
            y[train_indices],
            frame.iloc[valid_indices].reset_index(drop=True),
            config,
            seeds=model_seeds,
            thread_count=thread_count,
        )
        seen[valid_indices] += 1
    if not np.array_equal(seen, np.ones(len(frame), dtype=int)):
        raise RuntimeError("cross-fit did not score every row exactly once")
    return prediction


class HonestNestedEvaluator:
    """Evaluate the complete inner-selection algorithm on untouched outer folds."""

    def __init__(
        self,
        configs: tuple[ModelConfig, ...],
        *,
        outer_splits: int,
        inner_splits: int,
        outer_seed: int,
        inner_seed: int,
        model_seeds: tuple[int, ...],
        minimum_complex_gain: float = 0.0005,
        diagnose_all_outer: bool = False,
        thread_count: int = -1,
    ) -> None:
        if not configs:
            raise ValueError("at least one model config is required")
        self.configs = configs
        self.outer_splits = outer_splits
        self.inner_splits = inner_splits
        self.outer_seed = outer_seed
        self.inner_seed = inner_seed
        self.model_seeds = model_seeds
        self.minimum_complex_gain = minimum_complex_gain
        self.diagnose_all_outer = diagnose_all_outer
        self.thread_count = thread_count

    def evaluate(self, frame: pd.DataFrame, y: np.ndarray) -> NestedResult:
        """Run outer nested CV; outer labels never influence fold selection."""
        labels = np.asarray(y, dtype=int)
        outer = make_stratified_splits(labels, n_splits=self.outer_splits, seed=self.outer_seed)
        nested_oof = np.empty(len(frame), dtype=float)
        fold_auc: list[float] = []
        selections: list[dict[str, object]] = []
        candidate_outer_auc = {config.name: [] for config in self.configs}

        for fold_index, (outer_train_indices, outer_valid_indices) in enumerate(outer):
            outer_train_frame = frame.iloc[outer_train_indices].reset_index(drop=True)
            outer_valid_frame = frame.iloc[outer_valid_indices].reset_index(drop=True)
            outer_y = labels[outer_train_indices]
            inner = make_stratified_splits(
                outer_y,
                n_splits=self.inner_splits,
                seed=self.inner_seed + fold_index,
            )
            inner_predictions: dict[str, np.ndarray] = {}
            scores: list[CandidateScore] = []
            print(f"[outer {fold_index + 1}/{self.outer_splits}] inner candidate evaluation", flush=True)
            for config in self.configs:
                prediction = cross_fitted_prediction(
                    outer_train_frame,
                    outer_y,
                    config,
                    inner,
                    model_seeds=self.model_seeds,
                    thread_count=self.thread_count,
                )
                inner_predictions[config.name] = prediction
                inner_auc = float(roc_auc_score(outer_y, prediction))
                scores.append(CandidateScore(config.name, inner_auc, config.complexity))
                print(f"  {config.name}: inner={inner_auc:.6f}", flush=True)

            selected_score = select_candidate(scores, minimum_complex_gain=self.minimum_complex_gain)
            selected_config = self._config_by_name(selected_score.name)
            configs_to_fit = self.configs if self.diagnose_all_outer else (selected_config,)
            outer_predictions: dict[str, np.ndarray] = {}
            print(f"  selected={selected_config.name}", flush=True)
            for config in configs_to_fit:
                prediction = fit_predict_config(
                    outer_train_frame,
                    outer_y,
                    outer_valid_frame,
                    config,
                    seeds=self.model_seeds,
                    thread_count=self.thread_count,
                )
                outer_predictions[config.name] = prediction
                score = float(roc_auc_score(labels[outer_valid_indices], prediction))
                candidate_outer_auc[config.name].append(score)
                print(f"  {config.name}: outer={score:.6f}", flush=True)
            selected_prediction = outer_predictions[selected_config.name]
            nested_oof[outer_valid_indices] = selected_prediction
            selected_outer_auc = float(roc_auc_score(labels[outer_valid_indices], selected_prediction))
            fold_auc.append(selected_outer_auc)
            selections.append(
                {
                    "fold": fold_index,
                    "selected": selected_config.name,
                    "selected_inner_auc": selected_score.inner_auc,
                    "outer_auc": selected_outer_auc,
                    "inner_auc_by_config": {score.name: score.inner_auc for score in scores},
                }
            )

        return NestedResult(
            oof_prediction=nested_oof,
            fold_auc=fold_auc,
            pooled_auc=float(roc_auc_score(labels, nested_oof)),
            fold_selections=selections,
            candidate_outer_auc=candidate_outer_auc,
        )

    def fit_final(
        self,
        train_frame: pd.DataFrame,
        y: np.ndarray,
        test_frame: pd.DataFrame,
    ) -> FinalFitResult:
        """Repeat inner selection on full train, then fit one locked test model."""
        labels = np.asarray(y, dtype=int)
        inner = make_stratified_splits(labels, n_splits=self.inner_splits, seed=self.inner_seed)
        inner_predictions: dict[str, np.ndarray] = {}
        scores: list[CandidateScore] = []
        print("[final] full-train inner candidate evaluation", flush=True)
        for config in self.configs:
            prediction = cross_fitted_prediction(
                train_frame.reset_index(drop=True),
                labels,
                config,
                inner,
                model_seeds=self.model_seeds,
                thread_count=self.thread_count,
            )
            inner_predictions[config.name] = prediction
            inner_auc = float(roc_auc_score(labels, prediction))
            scores.append(CandidateScore(config.name, inner_auc, config.complexity))
            print(f"  {config.name}: inner={inner_auc:.6f}", flush=True)
        selected_score = select_candidate(scores, minimum_complex_gain=self.minimum_complex_gain)
        selected_config = self._config_by_name(selected_score.name)
        print(f"  selected={selected_config.name}", flush=True)
        test_prediction = fit_predict_config(
            train_frame.reset_index(drop=True),
            labels,
            test_frame.reset_index(drop=True),
            selected_config,
            seeds=self.model_seeds,
            thread_count=self.thread_count,
        )
        return FinalFitResult(
            selected_config=selected_config,
            selected_inner_oof=inner_predictions[selected_config.name],
            test_prediction=test_prediction,
            inner_auc_by_config={score.name: score.inner_auc for score in scores},
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "outer_splits": self.outer_splits,
            "inner_splits": self.inner_splits,
            "outer_seed": self.outer_seed,
            "inner_seed": self.inner_seed,
            "model_seeds": list(self.model_seeds),
            "minimum_complex_gain": self.minimum_complex_gain,
            "diagnose_all_outer": self.diagnose_all_outer,
            "configs": [asdict(config) for config in self.configs],
        }

    def _config_by_name(self, name: str) -> ModelConfig:
        for config in self.configs:
            if config.name == name:
                return config
        raise KeyError(name)

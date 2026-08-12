"""Leakage-resistant nested evaluation and configuration selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .models import ModelConfig, fit_predict_config


@dataclass(frozen=True)
class CandidateScore:
    """An inner-OOF score used by the deterministic selector."""

    name: str
    inner_auc: float
    complexity: int


@dataclass(frozen=True)
class BlendSpec:
    """A fixed rank blend whose components are fitted independently."""

    name: str
    components: tuple[str, ...]
    weights: tuple[float, ...]
    complexity: int

    def __post_init__(self) -> None:
        if not self.components or len(self.components) != len(self.weights):
            raise ValueError("blend components and weights must have one shared non-zero length")
        if any(weight < 0 for weight in self.weights):
            raise ValueError("blend weights must be non-negative")
        if not np.isclose(sum(self.weights), 1.0):
            raise ValueError("blend weights must sum to one")

    def combine(self, predictions: dict[str, np.ndarray]) -> np.ndarray:
        """Combine pre-registered component predictions without fitting weights."""
        arrays = [np.asarray(predictions[name], dtype=float) for name in self.components]
        if any(len(array) != len(arrays[0]) for array in arrays):
            raise ValueError("blend component lengths differ")
        return sum(weight * array for weight, array in zip(self.weights, arrays))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ResidualSpec:
    """A fixed sequential arm trained only on cross-fitted base residuals."""

    name: str
    base_components: tuple[str, ...]
    base_weights: tuple[float, ...]
    residual_component: str
    alpha: float
    complexity: int
    subinner_splits: int = 2
    subinner_seed: int = 424243

    def __post_init__(self) -> None:
        if not self.base_components or len(self.base_components) != len(self.base_weights):
            raise ValueError("residual base components and weights must share a non-zero length")
        if any(weight < 0 for weight in self.base_weights):
            raise ValueError("residual base weights must be non-negative")
        if not np.isclose(sum(self.base_weights), 1.0):
            raise ValueError("residual base weights must sum to one")
        if self.alpha <= 0:
            raise ValueError("residual alpha must be positive")
        if self.subinner_splits < 2:
            raise ValueError("residual sub-inner splits must be at least two")

    @property
    def components(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.base_components, self.residual_component)))

    def base_prediction(self, predictions: dict[str, np.ndarray]) -> np.ndarray:
        arrays = [np.asarray(predictions[name], dtype=float) for name in self.base_components]
        if any(len(array) != len(arrays[0]) for array in arrays):
            raise ValueError("residual base component lengths differ")
        return sum(weight * array for weight, array in zip(self.base_weights, arrays))

    def combine(self, base_prediction: np.ndarray, residual_prediction: np.ndarray) -> np.ndarray:
        base = np.asarray(base_prediction, dtype=float)
        residual = np.asarray(residual_prediction, dtype=float)
        if len(base) != len(residual):
            raise ValueError("base and residual prediction lengths differ")
        return base + self.alpha * residual

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class StackSpec:
    """A fixed, cross-fitted logistic meta-model over OOF component ranks."""

    name: str
    components: tuple[str, ...]
    regularization_c: float
    complexity: int

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("stack must have at least one component")
        if self.regularization_c <= 0:
            raise ValueError("stack regularization C must be positive")

    def cross_fit(
        self,
        predictions: dict[str, np.ndarray],
        y: np.ndarray,
        splits: list[tuple[np.ndarray, np.ndarray]],
    ) -> tuple[np.ndarray, list[list[float]]]:
        """Cross-fit the meta-model so its own inner score is label-honest."""
        matrix = self._matrix(predictions)
        labels = np.asarray(y, dtype=int)
        oof = np.empty(len(labels), dtype=float)
        coefficients: list[list[float]] = []
        seen = np.zeros(len(labels), dtype=int)
        for train_indices, valid_indices in splits:
            model = self._new_model()
            model.fit(matrix[train_indices], labels[train_indices])
            oof[valid_indices] = model.predict_proba(matrix[valid_indices])[:, 1]
            coefficients.append(model.coef_[0].astype(float).tolist())
            seen[valid_indices] += 1
        if not np.array_equal(seen, np.ones(len(labels), dtype=int)):
            raise RuntimeError("stack cross-fit did not score every row exactly once")
        return oof, coefficients

    def fit_predict(
        self,
        train_predictions: dict[str, np.ndarray],
        y: np.ndarray,
        prediction_predictions: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, list[float]]:
        """Fit on full cross-fitted train ranks and predict an untouched partition."""
        model = self._new_model()
        model.fit(self._matrix(train_predictions), np.asarray(y, dtype=int))
        prediction = model.predict_proba(self._matrix(prediction_predictions))[:, 1]
        return prediction, model.coef_[0].astype(float).tolist()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def _matrix(self, predictions: dict[str, np.ndarray]) -> np.ndarray:
        arrays = [np.asarray(predictions[name], dtype=float) for name in self.components]
        if any(len(array) != len(arrays[0]) for array in arrays):
            raise ValueError("stack component lengths differ")
        return np.column_stack(arrays)

    def _new_model(self) -> LogisticRegression:
        return LogisticRegression(
            C=self.regularization_c,
            l1_ratio=0.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=2026,
        )


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

    selected_recipe: dict[str, object]
    selected_inner_oof: np.ndarray
    test_prediction: np.ndarray
    inner_auc_by_config: dict[str, float]
    recipe_metadata: dict[str, object] | None = None

    def metrics(self) -> dict[str, object]:
        return {
            "selected_recipe": self.selected_recipe,
            "inner_auc_by_config": self.inner_auc_by_config,
            "recipe_metadata": self.recipe_metadata,
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


def cross_fitted_residual_prediction(
    frame: pd.DataFrame,
    y: np.ndarray,
    spec: ResidualSpec,
    config_by_name: dict[str, ModelConfig],
    base_oof_predictions: dict[str, np.ndarray],
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    model_seeds: tuple[int, ...],
    thread_count: int,
) -> np.ndarray:
    """Cross-fit residuals with an additional base-OOF layer in every train fold."""
    prediction = np.empty(len(frame), dtype=float)
    seen = np.zeros(len(frame), dtype=int)
    for fold_index, (train_indices, valid_indices) in enumerate(splits):
        train_frame = frame.iloc[train_indices].reset_index(drop=True)
        valid_frame = frame.iloc[valid_indices].reset_index(drop=True)
        train_y = y[train_indices]
        subinner = make_stratified_splits(
            train_y,
            n_splits=spec.subinner_splits,
            seed=spec.subinner_seed + fold_index,
        )
        subinner_base_predictions: dict[str, np.ndarray] = {}
        for component_name in spec.base_components:
            subinner_base_predictions[component_name] = cross_fitted_prediction(
                train_frame,
                train_y,
                config_by_name[component_name],
                subinner,
                model_seeds=model_seeds,
                thread_count=thread_count,
            )
        train_base = spec.base_prediction(subinner_base_predictions)
        residual_target = train_y.astype(float) - train_base
        residual_valid = fit_predict_config(
            train_frame,
            residual_target,
            valid_frame,
            config_by_name[spec.residual_component],
            seeds=model_seeds,
            thread_count=thread_count,
            rank_output=False,
        )
        valid_base = spec.base_prediction(
            {
                component_name: base_oof_predictions[component_name][valid_indices]
                for component_name in spec.base_components
            }
        )
        prediction[valid_indices] = spec.combine(valid_base, residual_valid)
        seen[valid_indices] += 1
    if not np.array_equal(seen, np.ones(len(frame), dtype=int)):
        raise RuntimeError("residual cross-fit did not score every row exactly once")
    return prediction


class HonestNestedEvaluator:
    """Evaluate the complete inner-selection algorithm on untouched outer folds."""

    def __init__(
        self,
        configs: tuple[ModelConfig, ...],
        *,
        blends: tuple[BlendSpec, ...] = (),
        stacks: tuple[StackSpec, ...] = (),
        residuals: tuple[ResidualSpec, ...] = (),
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
        config_names = {config.name for config in configs}
        for recipe in (*blends, *stacks, *residuals):
            missing = set(recipe.components) - config_names
            if missing:
                raise ValueError(f"recipe {recipe.name!r} has unknown components {sorted(missing)}")
        recipe_names = (
            [config.name for config in configs]
            + [blend.name for blend in blends]
            + [stack.name for stack in stacks]
            + [residual.name for residual in residuals]
        )
        if len(recipe_names) != len(set(recipe_names)):
            raise ValueError("config and blend names must be unique")
        self.configs = configs
        self.blends = blends
        self.stacks = stacks
        self.residuals = residuals
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
        candidate_outer_auc = {name: [] for name in self._recipe_names()}

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
            for blend in self.blends:
                prediction = blend.combine(inner_predictions)
                inner_predictions[blend.name] = prediction
                inner_auc = float(roc_auc_score(outer_y, prediction))
                scores.append(CandidateScore(blend.name, inner_auc, blend.complexity))
                print(f"  {blend.name}: inner={inner_auc:.6f}", flush=True)
            inner_stack_coefficients: dict[str, list[list[float]]] = {}
            for stack in self.stacks:
                prediction, coefficients = stack.cross_fit(
                    inner_predictions,
                    outer_y,
                    inner,
                )
                inner_predictions[stack.name] = prediction
                inner_stack_coefficients[stack.name] = coefficients
                inner_auc = float(roc_auc_score(outer_y, prediction))
                scores.append(CandidateScore(stack.name, inner_auc, stack.complexity))
                print(f"  {stack.name}: inner={inner_auc:.6f}", flush=True)
            config_by_name = {config.name: config for config in self.configs}
            for residual in self.residuals:
                prediction = cross_fitted_residual_prediction(
                    outer_train_frame,
                    outer_y,
                    residual,
                    config_by_name,
                    inner_predictions,
                    inner,
                    model_seeds=self.model_seeds,
                    thread_count=self.thread_count,
                )
                inner_predictions[residual.name] = prediction
                inner_auc = float(roc_auc_score(outer_y, prediction))
                scores.append(CandidateScore(residual.name, inner_auc, residual.complexity))
                print(f"  {residual.name}: inner={inner_auc:.6f}", flush=True)

            selected_score = select_candidate(scores, minimum_complex_gain=self.minimum_complex_gain)
            required_names = (
                {config.name for config in self.configs}
                if self.diagnose_all_outer
                else set(self._components_for_recipe(selected_score.name))
            )
            configs_to_fit = tuple(
                config for config in self.configs if config.name in required_names
            )
            outer_predictions: dict[str, np.ndarray] = {}
            print(f"  selected={selected_score.name}", flush=True)
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
            for blend in self.blends:
                if all(component in outer_predictions for component in blend.components):
                    outer_predictions[blend.name] = blend.combine(outer_predictions)
            outer_stack_coefficients: dict[str, list[float]] = {}
            for stack in self.stacks:
                if all(component in outer_predictions for component in stack.components):
                    prediction, coefficients = stack.fit_predict(
                        inner_predictions,
                        outer_y,
                        outer_predictions,
                    )
                    outer_predictions[stack.name] = prediction
                    outer_stack_coefficients[stack.name] = coefficients
            for residual in self.residuals:
                if not all(component in outer_predictions for component in residual.base_components):
                    continue
                train_base = residual.base_prediction(inner_predictions)
                valid_base = residual.base_prediction(outer_predictions)
                residual_prediction = fit_predict_config(
                    outer_train_frame,
                    outer_y.astype(float) - train_base,
                    outer_valid_frame,
                    self._config_by_name(residual.residual_component),
                    seeds=self.model_seeds,
                    thread_count=self.thread_count,
                    rank_output=False,
                )
                outer_predictions[residual.name] = residual.combine(
                    valid_base,
                    residual_prediction,
                )
            recipes_to_score = (
                self._recipe_names() if self.diagnose_all_outer else (selected_score.name,)
            )
            for recipe_name in recipes_to_score:
                prediction = outer_predictions[recipe_name]
                score = float(roc_auc_score(labels[outer_valid_indices], prediction))
                candidate_outer_auc[recipe_name].append(score)
                print(f"  {recipe_name}: outer={score:.6f}", flush=True)
            selected_prediction = outer_predictions[selected_score.name]
            nested_oof[outer_valid_indices] = selected_prediction
            selected_outer_auc = float(roc_auc_score(labels[outer_valid_indices], selected_prediction))
            fold_auc.append(selected_outer_auc)
            selections.append(
                {
                    "fold": fold_index,
                    "selected": selected_score.name,
                    "selected_inner_auc": selected_score.inner_auc,
                    "outer_auc": selected_outer_auc,
                    "inner_auc_by_config": {score.name: score.inner_auc for score in scores},
                    "stack_coefficients": {
                        name: {
                            "inner_cross_fit": inner_stack_coefficients[name],
                            "outer_fit": outer_stack_coefficients.get(name),
                        }
                        for name in inner_stack_coefficients
                    },
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
        for blend in self.blends:
            prediction = blend.combine(inner_predictions)
            inner_predictions[blend.name] = prediction
            inner_auc = float(roc_auc_score(labels, prediction))
            scores.append(CandidateScore(blend.name, inner_auc, blend.complexity))
            print(f"  {blend.name}: inner={inner_auc:.6f}", flush=True)
        final_stack_cross_fit_coefficients: dict[str, list[list[float]]] = {}
        for stack in self.stacks:
            prediction, coefficients = stack.cross_fit(inner_predictions, labels, inner)
            inner_predictions[stack.name] = prediction
            final_stack_cross_fit_coefficients[stack.name] = coefficients
            inner_auc = float(roc_auc_score(labels, prediction))
            scores.append(CandidateScore(stack.name, inner_auc, stack.complexity))
            print(f"  {stack.name}: inner={inner_auc:.6f}", flush=True)
        config_by_name = {config.name: config for config in self.configs}
        for residual in self.residuals:
            prediction = cross_fitted_residual_prediction(
                train_frame.reset_index(drop=True),
                labels,
                residual,
                config_by_name,
                inner_predictions,
                inner,
                model_seeds=self.model_seeds,
                thread_count=self.thread_count,
            )
            inner_predictions[residual.name] = prediction
            inner_auc = float(roc_auc_score(labels, prediction))
            scores.append(CandidateScore(residual.name, inner_auc, residual.complexity))
            print(f"  {residual.name}: inner={inner_auc:.6f}", flush=True)
        selected_score = select_candidate(scores, minimum_complex_gain=self.minimum_complex_gain)
        print(f"  selected={selected_score.name}", flush=True)
        component_predictions: dict[str, np.ndarray] = {}
        for component_name in self._components_for_recipe(selected_score.name):
            config = self._config_by_name(component_name)
            component_predictions[component_name] = fit_predict_config(
                train_frame.reset_index(drop=True),
                labels,
                test_frame.reset_index(drop=True),
                config,
                seeds=self.model_seeds,
                thread_count=self.thread_count,
            )
        recipe_metadata: dict[str, object] | None = None
        if any(stack.name == selected_score.name for stack in self.stacks):
            stack = self._stack_by_name(selected_score.name)
            test_prediction, coefficients = stack.fit_predict(
                inner_predictions,
                labels,
                component_predictions,
            )
            recipe_metadata = {
                "inner_cross_fit_coefficients": final_stack_cross_fit_coefficients[stack.name],
                "full_fit_coefficients": coefficients,
            }
        elif any(residual.name == selected_score.name for residual in self.residuals):
            residual = self._residual_by_name(selected_score.name)
            train_base = residual.base_prediction(inner_predictions)
            test_base = residual.base_prediction(component_predictions)
            residual_prediction = fit_predict_config(
                train_frame.reset_index(drop=True),
                labels.astype(float) - train_base,
                test_frame.reset_index(drop=True),
                self._config_by_name(residual.residual_component),
                seeds=self.model_seeds,
                thread_count=self.thread_count,
                rank_output=False,
            )
            test_prediction = residual.combine(test_base, residual_prediction)
            recipe_metadata = {
                "residual_target_mean": float(np.mean(labels - train_base)),
                "residual_target_std": float(np.std(labels - train_base)),
            }
        else:
            test_prediction = self._prediction_for_recipe(
                selected_score.name,
                component_predictions,
            )
        return FinalFitResult(
            selected_recipe=self._recipe_descriptor(selected_score.name),
            selected_inner_oof=inner_predictions[selected_score.name],
            test_prediction=test_prediction,
            inner_auc_by_config={score.name: score.inner_auc for score in scores},
            recipe_metadata=recipe_metadata,
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
            "blends": [blend.to_dict() for blend in self.blends],
            "stacks": [stack.to_dict() for stack in self.stacks],
            "residuals": [residual.to_dict() for residual in self.residuals],
        }

    def _config_by_name(self, name: str) -> ModelConfig:
        for config in self.configs:
            if config.name == name:
                return config
        raise KeyError(name)

    def _blend_by_name(self, name: str) -> BlendSpec:
        for blend in self.blends:
            if blend.name == name:
                return blend
        raise KeyError(name)

    def _stack_by_name(self, name: str) -> StackSpec:
        for stack in self.stacks:
            if stack.name == name:
                return stack
        raise KeyError(name)

    def _residual_by_name(self, name: str) -> ResidualSpec:
        for residual in self.residuals:
            if residual.name == name:
                return residual
        raise KeyError(name)

    def _recipe_names(self) -> tuple[str, ...]:
        return tuple(config.name for config in self.configs) + tuple(
            blend.name for blend in self.blends
        ) + tuple(stack.name for stack in self.stacks) + tuple(
            residual.name for residual in self.residuals
        )

    def _components_for_recipe(self, name: str) -> tuple[str, ...]:
        if any(config.name == name for config in self.configs):
            return (name,)
        for blend in self.blends:
            if blend.name == name:
                return blend.components
        for stack in self.stacks:
            if stack.name == name:
                return stack.components
        return self._residual_by_name(name).components

    def _prediction_for_recipe(
        self,
        name: str,
        predictions: dict[str, np.ndarray],
    ) -> np.ndarray:
        if name in predictions:
            return predictions[name]
        return self._blend_by_name(name).combine(predictions)

    def _recipe_descriptor(self, name: str) -> dict[str, object]:
        if any(config.name == name for config in self.configs):
            return {"type": "model", **self._config_by_name(name).to_dict()}
        for blend in self.blends:
            if blend.name == name:
                return {"type": "blend", **blend.to_dict()}
        for stack in self.stacks:
            if stack.name == name:
                return {"type": "stack", **stack.to_dict()}
        return {"type": "residual", **self._residual_by_name(name).to_dict()}

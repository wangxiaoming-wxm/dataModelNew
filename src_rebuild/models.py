"""Pre-registered CatBoost model family for honest nested evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from scipy.stats import rankdata

from .features import RebuildFeatureBuilder


@dataclass(frozen=True)
class ModelConfig:
    """One immutable model and feature-world configuration."""

    name: str
    feature_mode: str
    objective: str
    depth: int
    iterations: int
    learning_rate: float
    l2_leaf_reg: float
    random_strength: float
    complexity: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def with_iterations(self, iterations: int) -> "ModelConfig":
        return replace(self, iterations=iterations)


def candidate_configs(profile: str) -> tuple[ModelConfig, ...]:
    """Return the finite candidate set; no runtime grid expansion is allowed."""
    if profile not in {"smoke", "full"}:
        raise ValueError("profile must be 'smoke' or 'full'")
    logloss_iterations = 250 if profile == "smoke" else 700
    rmse_iterations = 300 if profile == "smoke" else 800
    return (
        ModelConfig(
            name="cb_core_logloss_d5",
            feature_mode="core",
            objective="logloss",
            depth=5,
            iterations=logloss_iterations,
            learning_rate=0.03,
            l2_leaf_reg=10.0,
            random_strength=0.7,
            complexity=0,
        ),
        ModelConfig(
            name="cb_all_logloss_d5",
            feature_mode="all",
            objective="logloss",
            depth=5,
            iterations=logloss_iterations,
            learning_rate=0.03,
            l2_leaf_reg=10.0,
            random_strength=0.7,
            complexity=1,
        ),
        ModelConfig(
            name="cb_core_rmse_d5",
            feature_mode="core",
            objective="rmse",
            depth=5,
            iterations=rmse_iterations,
            learning_rate=0.03,
            l2_leaf_reg=10.0,
            random_strength=0.7,
            complexity=1,
        ),
        ModelConfig(
            name="cb_core_logloss_d6",
            feature_mode="core",
            objective="logloss",
            depth=6,
            iterations=logloss_iterations,
            learning_rate=0.03,
            l2_leaf_reg=12.0,
            random_strength=0.7,
            complexity=1,
        ),
        ModelConfig(
            name="cb_ratio_logloss_d5",
            feature_mode="ratio",
            objective="logloss",
            depth=5,
            iterations=logloss_iterations,
            learning_rate=0.03,
            l2_leaf_reg=10.0,
            random_strength=0.7,
            complexity=1,
        ),
        ModelConfig(
            name="cb_ratio_rmse_d5",
            feature_mode="ratio",
            objective="rmse",
            depth=5,
            iterations=rmse_iterations,
            learning_rate=0.03,
            l2_leaf_reg=10.0,
            random_strength=0.7,
            complexity=1,
        ),
        ModelConfig(
            name="cb_rate_rmse_d6",
            feature_mode="rate",
            objective="rmse",
            depth=6,
            iterations=rmse_iterations,
            learning_rate=0.03,
            l2_leaf_reg=6.0,
            random_strength=0.7,
            complexity=1,
        ),
        ModelConfig(
            name="cb_all_id_logloss_d5",
            feature_mode="all_id",
            objective="logloss",
            depth=5,
            iterations=logloss_iterations,
            learning_rate=0.03,
            l2_leaf_reg=12.0,
            random_strength=0.7,
            complexity=2,
        ),
    )


def fit_predict_config(
    train_frame: pd.DataFrame,
    y_train: np.ndarray,
    prediction_frame: pd.DataFrame,
    config: ModelConfig,
    *,
    seeds: tuple[int, ...],
    thread_count: int = -1,
) -> np.ndarray:
    """Fit one configuration on one partition and rank-average predictions."""
    builder = RebuildFeatureBuilder(config.feature_mode)
    train_matrix = builder.fit_transform(train_frame)
    prediction_matrix = builder.transform(prediction_frame)
    train_pool = Pool(
        train_matrix.frame,
        y_train,
        cat_features=list(train_matrix.cat_columns),
    )
    prediction_pool = Pool(
        prediction_matrix.frame,
        cat_features=list(prediction_matrix.cat_columns),
    )

    predictions: list[np.ndarray] = []
    for seed in seeds:
        common = {
            "iterations": config.iterations,
            "learning_rate": config.learning_rate,
            "depth": config.depth,
            "l2_leaf_reg": config.l2_leaf_reg,
            "random_strength": config.random_strength,
            "random_seed": seed,
            "verbose": False,
            "allow_writing_files": False,
            "thread_count": thread_count,
        }
        if config.objective == "logloss":
            model = CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="AUC",
                **common,
            )
            model.fit(train_pool, verbose=False)
            raw_prediction = model.predict_proba(prediction_pool)[:, 1]
        elif config.objective == "rmse":
            model = CatBoostRegressor(
                loss_function="RMSE",
                eval_metric="RMSE",
                **common,
            )
            model.fit(train_pool, verbose=False)
            raw_prediction = model.predict(prediction_pool)
        else:
            raise ValueError(f"unsupported objective {config.objective!r}")
        predictions.append(rankdata(np.asarray(raw_prediction, dtype=float)) / len(raw_prediction))
    return np.mean(predictions, axis=0)

"""SUPER714 的折内诚实 ``TE(source × days_bin10)`` 特征。

调用方应在每个外层 CV fold 内调用 :func:`build_source_days_te`：

* ``fit_frame`` 只传外层训练行；
* ``valid_frame`` 传外层验证行；
* ``other_frames`` 通常传测试集；
* 训练行的编码由内层交叉拟合生成，任何行都不会看到自己的标签；
* 验证集和测试集只使用完整外层训练折的统计量。

本模块只负责特征构造，不包含 CatBoost 训练或提交流水线。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

FEATURE_NAME = "te_source_days_bin10"
_MISSING_SOURCE = "__MISSING_SOURCE__"


@dataclass(frozen=True)
class FoldTargetEncoding:
    """一个外层 fold 的 TE 结果及可审计状态。"""

    fit: pd.Series
    valid: pd.Series
    others: tuple[pd.Series, ...]
    days_edges: np.ndarray
    prior: float
    mapping: Mapping[str, float]


def fit_days_bin_edges(days: pd.Series, n_bins: int = 10) -> np.ndarray:
    """仅用给定训练折拟合等频分箱边界。

    重复分位点会被合并；缺失值不会参与边界拟合，之后进入独立的 ``-1``
    分箱。
    """

    if n_bins < 2:
        raise ValueError("n_bins 必须至少为 2")

    numeric = pd.to_numeric(days, errors="coerce").to_numpy(dtype=float)
    finite = numeric[np.isfinite(numeric)]
    if finite.size == 0:
        return np.empty(0, dtype=float)

    quantiles = np.arange(1, n_bins, dtype=float) / n_bins
    return np.unique(np.quantile(finite, quantiles)).astype(float)


def make_source_days_key(
    frame: pd.DataFrame,
    days_edges: np.ndarray,
    *,
    source_col: str = "source",
    days_col: str = "days",
) -> pd.Series:
    """按已经拟合的边界构造稳定的 ``source|days_bin`` 键。"""

    missing = [col for col in (source_col, days_col) if col not in frame.columns]
    if missing:
        raise KeyError(f"缺少 TE 输入列: {missing}")

    source = frame[source_col].astype("string").fillna(_MISSING_SOURCE)
    days = pd.to_numeric(frame[days_col], errors="coerce").to_numpy(dtype=float)
    bins = np.full(len(frame), -1, dtype=np.int16)
    finite = np.isfinite(days)
    bins[finite] = np.searchsorted(days_edges, days[finite], side="right")

    values = source.astype(str).to_numpy() + "|d" + bins.astype(str)
    return pd.Series(values, index=frame.index, name="source_days_bin10")


def _validate_target(y: Sequence[int] | np.ndarray, expected_rows: int) -> np.ndarray:
    target = np.asarray(y)
    if target.ndim != 1 or len(target) != expected_rows:
        raise ValueError(
            f"y_fit 必须是一维且长度等于 fit_frame: {target.shape} vs {expected_rows}"
        )
    if pd.isna(target).any():
        raise ValueError("y_fit 不能包含缺失值")

    unique = np.unique(target)
    if not np.all(np.isin(unique, (0, 1))):
        raise ValueError(f"y_fit 必须是二元 0/1 标签，实际取值: {unique.tolist()}")
    return target.astype(float, copy=False)


def _fit_mapping(
    keys: pd.Series,
    y: np.ndarray,
    *,
    smoothing: float,
) -> tuple[dict[str, float], float]:
    prior = float(np.mean(y))
    stats = (
        pd.DataFrame({"key": keys.to_numpy(), "target": y})
        .groupby("key", sort=False, observed=True)["target"]
        .agg(["sum", "count"])
    )
    encoded = (stats["sum"] + smoothing * prior) / (
        stats["count"] + smoothing
    )
    return encoded.astype(float).to_dict(), prior


def _apply_mapping(
    keys: pd.Series,
    mapping: Mapping[str, float],
    prior: float,
) -> pd.Series:
    encoded = keys.map(mapping).fillna(prior).astype(float)
    encoded.name = FEATURE_NAME
    return encoded


def build_source_days_te(
    fit_frame: pd.DataFrame,
    y_fit: Sequence[int] | np.ndarray,
    valid_frame: pd.DataFrame,
    other_frames: Sequence[pd.DataFrame] = (),
    *,
    n_bins: int = 10,
    smoothing: float = 20.0,
    inner_splits: int = 4,
    inner_seed: int = 2026,
    source_col: str = "source",
    days_col: str = "days",
) -> FoldTargetEncoding:
    """构造一个外层 fold 内可直接加入模型帧的诚实 TE。

    ``fit`` 使用内层 OOF 编码；``valid`` 和 ``others`` 使用完整外层训练折
    的映射。未知组合统一回退到相应训练统计量的正例先验。
    """

    if len(fit_frame) == 0:
        raise ValueError("fit_frame 不能为空")
    if smoothing < 0:
        raise ValueError("smoothing 不能为负数")
    if inner_splits < 2:
        raise ValueError("inner_splits 必须至少为 2")

    y = _validate_target(y_fit, len(fit_frame))
    class_counts = np.bincount(y.astype(int), minlength=2)
    if np.min(class_counts) < inner_splits:
        raise ValueError(
            "每个类别的样本数必须不少于 inner_splits；"
            f"类别计数={class_counts.tolist()}, inner_splits={inner_splits}"
        )

    days_edges = fit_days_bin_edges(fit_frame[days_col], n_bins=n_bins)
    fit_keys = make_source_days_key(
        fit_frame,
        days_edges,
        source_col=source_col,
        days_col=days_col,
    )
    valid_keys = make_source_days_key(
        valid_frame,
        days_edges,
        source_col=source_col,
        days_col=days_col,
    )
    other_keys = tuple(
        make_source_days_key(
            frame,
            days_edges,
            source_col=source_col,
            days_col=days_col,
        )
        for frame in other_frames
    )

    fit_encoded = pd.Series(
        np.nan,
        index=fit_frame.index,
        dtype=float,
        name=FEATURE_NAME,
    )
    splitter = StratifiedKFold(
        n_splits=inner_splits,
        shuffle=True,
        random_state=inner_seed,
    )
    positions = np.arange(len(fit_frame))
    for inner_train, inner_valid in splitter.split(positions, y):
        mapping, prior = _fit_mapping(
            fit_keys.iloc[inner_train],
            y[inner_train],
            smoothing=smoothing,
        )
        encoded = _apply_mapping(
            fit_keys.iloc[inner_valid],
            mapping,
            prior,
        )
        fit_encoded.iloc[inner_valid] = encoded.to_numpy()

    if fit_encoded.isna().any():
        raise RuntimeError("内层 OOF TE 未覆盖全部训练行")

    full_mapping, full_prior = _fit_mapping(
        fit_keys,
        y,
        smoothing=smoothing,
    )
    valid_encoded = _apply_mapping(valid_keys, full_mapping, full_prior)
    other_encoded = tuple(
        _apply_mapping(keys, full_mapping, full_prior) for keys in other_keys
    )

    return FoldTargetEncoding(
        fit=fit_encoded,
        valid=valid_encoded,
        others=other_encoded,
        days_edges=days_edges.copy(),
        prior=full_prior,
        mapping=full_mapping,
    )

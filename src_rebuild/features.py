"""Label-free, fold-fitted feature construction for the rebuild."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


FEATURE_MODES = ("core", "all", "all_id")
RAW_X_COLUMNS = tuple(f"x{index}" for index in range(19))
CATEGORICAL_COLUMNS = (
    "month",
    "region",
    "t1",
    "t2",
    "t3",
    "r1",
    "r2",
    "code",
    "x19",
    "x20",
    "c1",
    "c2",
    "age_range",
    "livability",
    "source",
    "grades",
    "w1",
    "w2",
    "version",
)
GROUP_COLUMNS = ("month", "region", "source", "version", "grades", "code", "age_range")
GROUP_NUMERIC_COLUMNS = ("days", "condition")
LOG_COLUMNS = ("days", "condition", "cc", "max_g")


@dataclass(frozen=True)
class FeatureMatrix:
    """A model-ready frame and its categorical feature names."""

    frame: pd.DataFrame
    cat_columns: tuple[str, ...]


class RebuildFeatureBuilder:
    """Fit label-free feature statistics on one training partition."""

    def __init__(self, mode: str) -> None:
        if mode not in FEATURE_MODES:
            raise ValueError(f"unknown feature mode {mode!r}; expected one of {FEATURE_MODES}")
        self.mode = mode
        self._frequency_maps: dict[str, pd.Series] = {}
        self._group_mean_maps: dict[tuple[str, str], pd.Series] = {}
        self._global_means: dict[str, float] = {}
        self._numeric_medians: dict[str, float] = {}
        self._columns: tuple[str, ...] | None = None
        self._cat_columns: tuple[str, ...] | None = None

    def fit(self, frame: pd.DataFrame) -> "RebuildFeatureBuilder":
        """Fit frequencies, group means, schema and numeric imputers."""
        self._validate_input(frame)
        for group in GROUP_COLUMNS:
            normalized = self._category_values(frame[group])
            self._frequency_maps[group] = normalized.value_counts(normalize=True, dropna=False)
            for numeric in GROUP_NUMERIC_COLUMNS:
                values = pd.to_numeric(frame[numeric], errors="coerce")
                stats = pd.DataFrame({"group": normalized, "value": values}).groupby("group")["value"].mean()
                self._group_mean_maps[(group, numeric)] = stats
                self._global_means[numeric] = float(values.mean())

        raw = self._build(frame)
        cat_columns = tuple(column for column in CATEGORICAL_COLUMNS if column in raw.columns)
        for column in raw.columns:
            if column in cat_columns:
                continue
            values = pd.to_numeric(raw[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            median = float(values.median())
            self._numeric_medians[column] = median if np.isfinite(median) else 0.0
        self._columns = tuple(raw.columns)
        self._cat_columns = cat_columns
        return self

    def transform(self, frame: pd.DataFrame) -> FeatureMatrix:
        """Transform a frame using statistics fitted on a training partition."""
        if self._columns is None or self._cat_columns is None:
            raise RuntimeError("feature builder must be fitted before transform")
        self._validate_input(frame, require_label=False)
        result = self._build(frame)
        missing = set(self._columns) - set(result.columns)
        extra = set(result.columns) - set(self._columns)
        if missing or extra:
            raise ValueError(f"feature schema changed; missing={sorted(missing)}, extra={sorted(extra)}")
        result = result.loc[:, self._columns].copy()
        for column in self._cat_columns:
            result[column] = self._category_values(result[column])
        for column, median in self._numeric_medians.items():
            values = pd.to_numeric(result[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            result[column] = values.fillna(median).astype(float)
        return FeatureMatrix(result.reset_index(drop=True), self._cat_columns)

    def fit_transform(self, frame: pd.DataFrame) -> FeatureMatrix:
        """Fit and transform the same training partition."""
        return self.fit(frame).transform(frame)

    @staticmethod
    def _category_values(series: pd.Series) -> pd.Series:
        return series.fillna("__MISSING__").astype(str)

    def _validate_input(self, frame: pd.DataFrame, *, require_label: bool = False) -> None:
        required = {
            "id",
            "days",
            "condition",
            "cc",
            "max_g",
            *CATEGORICAL_COLUMNS,
            *RAW_X_COLUMNS,
        }
        if require_label:
            required.add("label")
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")

    def _base_columns(self, frame: pd.DataFrame) -> list[str]:
        excluded = {"id", "label"}
        if self.mode == "core":
            excluded.update(RAW_X_COLUMNS)
        return [column for column in frame.columns if column not in excluded]

    def _build(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.loc[:, self._base_columns(frame)].copy()

        for column in CATEGORICAL_COLUMNS:
            if column in out.columns:
                out[column] = self._category_values(out[column])

        for column in LOG_COLUMNS:
            values = pd.to_numeric(frame[column], errors="coerce")
            out[f"{column}_logabs"] = np.log1p(np.abs(values))
            out[f"{column}_sign"] = np.sign(values)

        days = pd.to_numeric(frame["days"], errors="coerce")
        condition = pd.to_numeric(frame["condition"], errors="coerce")
        out["days_condition_ratio"] = days / (1.0 + np.abs(condition))
        out["days_condition_product"] = days * condition
        out["days_sqrt"] = np.sqrt(np.clip(days, 0.0, None))
        out["condition_sq"] = condition * condition

        for group in GROUP_COLUMNS:
            normalized = self._category_values(frame[group])
            frequency_map = self._frequency_maps.get(group)
            if frequency_map is None:
                raise RuntimeError("feature builder statistics are not fitted")
            out[f"{group}_freq"] = normalized.map(frequency_map).fillna(0.0).astype(float)
            for numeric in GROUP_NUMERIC_COLUMNS:
                mapping = self._group_mean_maps[(group, numeric)]
                fallback = self._global_means[numeric]
                group_mean = normalized.map(mapping).fillna(fallback).astype(float)
                values = pd.to_numeric(frame[numeric], errors="coerce")
                out[f"{numeric}_by_{group}_dev"] = values - group_mean

        if self.mode in ("all", "all_id"):
            raw_x = frame.loc[:, RAW_X_COLUMNS].apply(pd.to_numeric, errors="coerce")
            out["x_row_mean"] = raw_x.mean(axis=1)
            out["x_row_std"] = raw_x.std(axis=1)
            out["x_row_min"] = raw_x.min(axis=1)
            out["x_row_max"] = raw_x.max(axis=1)
            out["x_row_q25"] = raw_x.quantile(0.25, axis=1)
            out["x_row_q75"] = raw_x.quantile(0.75, axis=1)

        if self.mode == "all_id":
            self._add_identifier_features(out, frame["id"])

        return out

    @staticmethod
    def _add_identifier_features(out: pd.DataFrame, identifiers: pd.Series) -> None:
        normalized = identifiers.fillna("").astype(str).str.lower()
        valid = normalized.str.fullmatch(r"[0-9a-f]{16}")
        if not bool(valid.all()):
            bad_count = int((~valid).sum())
            raise ValueError(f"id must be 16 hexadecimal characters; invalid rows={bad_count}")
        bytes_array = np.array(
            [[int(value[pos : pos + 2], 16) for pos in range(0, 16, 2)] for value in normalized],
            dtype=np.uint8,
        )
        for index in range(8):
            out[f"id_byte_{index}"] = bytes_array[:, index].astype(float)
        for index in range(16):
            byte = bytes_array[:, index // 2]
            nibble = byte >> 4 if index % 2 == 0 else byte & 15
            out[f"id_nibble_{index}"] = nibble.astype(float)
        out["id_byte_mean"] = bytes_array.mean(axis=1)
        out["id_byte_std"] = bytes_array.std(axis=1)
        out["id_popcount"] = np.unpackbits(bytes_array, axis=1).sum(axis=1).astype(float)

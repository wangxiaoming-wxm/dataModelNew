"""Label-free, fold-fitted feature construction for the rebuild."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


FEATURE_MODES = ("core", "all", "all_id", "ratio", "rate", "ratio_rich", "rate_rich")
RAW_X_COLUMNS = tuple(f"x{index}" for index in range(19))
BIN_COLUMNS = ("t1", "t2", "r1", "r2", "c1", "c2", "w1", "w2")
GRADE_MAP = {"s": 1.0, "ss": 2.0, "sss": 3.0}
RATIO_QUANTILES = (5, 10, 20, 40)
RATE_QUANTILES = (7, 13, 25)
DAYS_FIXED_EDGES = np.array([700, 2500, 5000, 7000, 9000, 10000], dtype=float)
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
        self._source_condition_medians: pd.Series | None = None
        self._source_condition_values: dict[str, np.ndarray] = {}
        self._global_condition_median = 1.0
        self._quantile_edges: dict[str, np.ndarray] = {}

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

        self._fit_specialized_statistics(frame)
        raw = self._build(frame)
        cat_columns = tuple(
            column for column in raw.columns if not pd.api.types.is_numeric_dtype(raw[column])
        )
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
        if self.mode in ("ratio", "ratio_rich"):
            return self._build_ratio_world(frame, rich=self.mode == "ratio_rich")
        if self.mode in ("rate", "rate_rich"):
            return self._build_rate_world(frame, rich=self.mode == "rate_rich")
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

    def _fit_specialized_statistics(self, frame: pd.DataFrame) -> None:
        condition = pd.to_numeric(frame["condition"], errors="coerce")
        source = self._category_values(frame["source"])
        finite_condition = condition[np.isfinite(condition)]
        if len(finite_condition):
            self._global_condition_median = float(finite_condition.median())

        if self.mode in ("ratio", "ratio_rich"):
            self._source_condition_medians = pd.DataFrame(
                {"source": source, "condition": condition}
            ).groupby("source")["condition"].median()
            source_scale = source.map(self._source_condition_medians).fillna(
                self._global_condition_median
            )
            source_scale = source_scale.replace(0.0, self._global_condition_median)
            condition_ratio = condition / source_scale
            exposure_ratio = pd.to_numeric(frame["days"], errors="coerce") / condition_ratio.clip(
                lower=1e-9
            )
            for count in RATIO_QUANTILES:
                self._set_quantile_edges(f"days_{count}", frame["days"], count)
                self._set_quantile_edges(f"condition_{count}", condition, count)
                self._set_quantile_edges(f"condition_ratio_{count}", condition_ratio, count)
                self._set_quantile_edges(f"exposure_ratio_{count}", exposure_ratio, count)

        if self.mode in ("rate", "rate_rich"):
            grouped = pd.DataFrame({"source": source, "condition": condition}).groupby("source")
            self._source_condition_values = {
                str(name): np.sort(group["condition"].dropna().to_numpy(float))
                for name, group in grouped
            }
            percentile = self._condition_source_percentile(frame)
            exposure_rate = pd.to_numeric(frame["days"], errors="coerce") * (1.0 - percentile)
            for count in RATE_QUANTILES:
                self._set_quantile_edges(f"days_{count}", frame["days"], count)
                self._set_quantile_edges(f"condition_pct_{count}", percentile, count)
                self._set_quantile_edges(f"exposure_rate_{count}", exposure_rate, count)

    def _set_quantile_edges(self, name: str, values: pd.Series, count: int) -> None:
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
        finite = numeric[np.isfinite(numeric)]
        if not len(finite):
            self._quantile_edges[name] = np.array([], dtype=float)
            return
        self._quantile_edges[name] = np.quantile(
            finite,
            np.linspace(0.0, 1.0, count + 1)[1:-1],
        )

    def _quantile_bin(self, values: pd.Series, name: str) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce").fillna(-1.0).to_numpy(float)
        return pd.Series(np.digitize(numeric, self._quantile_edges[name]), index=values.index).astype(
            str
        )

    def _condition_source_percentile(self, frame: pd.DataFrame) -> pd.Series:
        source = self._category_values(frame["source"])
        condition = pd.to_numeric(frame["condition"], errors="coerce")
        result = np.full(len(frame), 0.5, dtype=float)
        for position, (group, value) in enumerate(zip(source, condition)):
            reference = self._source_condition_values.get(str(group))
            if reference is None or not len(reference) or not np.isfinite(value):
                continue
            result[position] = np.searchsorted(reference, value, side="right") / len(reference)
        return pd.Series(result, index=frame.index)

    @staticmethod
    def _add_cross(out: pd.DataFrame, name: str, columns: tuple[str, ...]) -> None:
        value = out[columns[0]].astype(str)
        for column in columns[1:]:
            value = value + "|" + out[column].astype(str)
        out[name] = value

    def _common_world_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=frame.index)
        for column in ("days", "condition", "cc", "max_g", "V", "age_range"):
            out[column] = pd.to_numeric(frame[column], errors="coerce")
        out["log_days"] = np.log1p(np.clip(out["days"], 0.0, None))
        out["log_condition"] = np.log1p(np.clip(out["condition"], 0.0, None))
        out["condition_missing"] = out["condition"].isna().astype(float)
        out["grade_ordinal"] = self._category_values(frame["grades"]).str.lower().map(
            GRADE_MAP
        )
        for column in BIN_COLUMNS:
            out[column] = pd.to_numeric(frame[column], errors="coerce")
        out["binary_sum"] = out.loc[:, BIN_COLUMNS].sum(axis=1)
        out["binary_pattern"] = frame.loc[:, BIN_COLUMNS].astype(str).agg("".join, axis=1)
        for column in ("region", "source", "month", "version", "grades", "age_range"):
            out[f"{column}_cat"] = self._category_values(frame[column])
        for column in ("x19", "x20", "livability", "t3", "code"):
            out[f"{column}_cat"] = self._category_values(frame[column])
        for group in GROUP_COLUMNS:
            normalized = self._category_values(frame[group])
            out[f"{group}_freq"] = normalized.map(self._frequency_maps[group]).fillna(0.0)
        self._add_cross(out, "region|source", ("region_cat", "source_cat"))
        self._add_cross(out, "region|age", ("region_cat", "age_range_cat"))
        self._add_cross(out, "source|age", ("source_cat", "age_range_cat"))
        self._add_cross(out, "x20|source", ("x20_cat", "source_cat"))
        self._add_cross(out, "x20|region", ("x20_cat", "region_cat"))
        return out

    def _build_ratio_world(self, frame: pd.DataFrame, *, rich: bool = False) -> pd.DataFrame:
        if self._source_condition_medians is None:
            raise RuntimeError("ratio statistics are not fitted")
        out = self._common_world_features(frame)
        source = self._category_values(frame["source"])
        condition = pd.to_numeric(frame["condition"], errors="coerce")
        source_scale = source.map(self._source_condition_medians).fillna(
            self._global_condition_median
        )
        source_scale = source_scale.replace(0.0, self._global_condition_median)
        condition_ratio = condition / source_scale
        days = pd.to_numeric(frame["days"], errors="coerce")
        exposure_ratio = days / condition_ratio.clip(lower=1e-9)
        out["condition_source_ratio"] = condition_ratio
        out["log_condition_source_ratio"] = np.log(condition_ratio.clip(lower=1e-9))
        out["exposure_ratio"] = exposure_ratio
        out["log_exposure_ratio"] = np.log(exposure_ratio.clip(lower=1e-9))
        out["exposure_ratio_p75"] = days / condition_ratio.clip(lower=1e-9).pow(0.75)
        out["condition_days_product"] = condition * days
        out["condition_over_days"] = condition / (days.abs() + 1.0)
        for count in RATIO_QUANTILES:
            out[f"days_q{count}"] = self._quantile_bin(days, f"days_{count}")
            out[f"condition_q{count}"] = self._quantile_bin(
                condition.fillna(-1.0), f"condition_{count}"
            )
            out[f"condition_ratio_q{count}"] = self._quantile_bin(
                condition_ratio, f"condition_ratio_{count}"
            )
            out[f"ratio_q{count}"] = self._quantile_bin(
                exposure_ratio, f"exposure_ratio_{count}"
            )
        for name, columns in (
            ("days_q10|region", ("days_q10", "region_cat")),
            ("days_q10|source", ("days_q10", "source_cat")),
            ("condition_q10|region", ("condition_q10", "region_cat")),
            ("condition_q10|source", ("condition_q10", "source_cat")),
            ("ratio_q10|region", ("ratio_q10", "region_cat")),
            ("ratio_q10|source", ("ratio_q10", "source_cat")),
            ("ratio_q10|age", ("ratio_q10", "age_range_cat")),
            ("ratio_q20|region", ("ratio_q20", "region_cat")),
            ("condition_ratio_q10|source", ("condition_ratio_q10", "source_cat")),
            ("days_q10|condition_q10", ("days_q10", "condition_q10")),
            ("days_q10|binary", ("days_q10", "binary_pattern")),
            ("region|binary", ("region_cat", "binary_pattern")),
        ):
            self._add_cross(out, name, columns)
        if rich:
            self._add_ratio_rich_features(out, frame)
        return out

    def _build_rate_world(self, frame: pd.DataFrame, *, rich: bool = False) -> pd.DataFrame:
        out = self._common_world_features(frame)
        days = pd.to_numeric(frame["days"], errors="coerce")
        percentile = self._condition_source_percentile(frame)
        exposure_rate = days * (1.0 - percentile)
        out["condition_source_pct"] = percentile
        out["exposure_rate"] = exposure_rate
        out["log_exposure_rate"] = np.log1p(exposure_rate.clip(lower=0.0))
        out["rate_over_age"] = exposure_rate / pd.to_numeric(
            frame["age_range"], errors="coerce"
        ).clip(lower=1.0)
        for count in RATE_QUANTILES:
            out[f"days_q{count}"] = self._quantile_bin(days, f"days_{count}")
            out[f"condition_pct_q{count}"] = self._quantile_bin(
                percentile, f"condition_pct_{count}"
            )
            out[f"rate_q{count}"] = self._quantile_bin(
                exposure_rate, f"exposure_rate_{count}"
            )
        for name, columns in (
            ("condition_pct_q13|source", ("condition_pct_q13", "source_cat")),
            ("condition_pct_q13|region", ("condition_pct_q13", "region_cat")),
            ("days_q13|source", ("days_q13", "source_cat")),
            ("days_q13|region", ("days_q13", "region_cat")),
            ("rate_q13|source", ("rate_q13", "source_cat")),
            ("rate_q13|region", ("rate_q13", "region_cat")),
            ("rate_q7|age", ("rate_q7", "age_range_cat")),
            ("days_q13|condition_pct_q13", ("days_q13", "condition_pct_q13")),
            ("rate_q7|binary", ("rate_q7", "binary_pattern")),
        ):
            self._add_cross(out, name, columns)
        if rich:
            self._add_rate_rich_features(out)
        return out

    def _add_ratio_rich_features(self, out: pd.DataFrame, frame: pd.DataFrame) -> None:
        days = pd.to_numeric(frame["days"], errors="coerce")
        age = pd.to_numeric(frame["age_range"], errors="coerce").clip(lower=1.0)
        out["days_over_age"] = days / age
        out["days_fixed"] = pd.Series(
            np.digitize(days.fillna(-1.0).to_numpy(float), DAYS_FIXED_EDGES),
            index=out.index,
        ).astype(str)
        interactions = (
            ("days_q5|region", ("days_q5", "region_cat")),
            ("days_q5|source", ("days_q5", "source_cat")),
            ("days_q20|region", ("days_q20", "region_cat")),
            ("days_q20|source", ("days_q20", "source_cat")),
            ("days_q10|age", ("days_q10", "age_range_cat")),
            ("condition_q5|region", ("condition_q5", "region_cat")),
            ("condition_q5|source", ("condition_q5", "source_cat")),
            ("condition_q20|region", ("condition_q20", "region_cat")),
            ("condition_q20|source", ("condition_q20", "source_cat")),
            ("condition_ratio_q5|region", ("condition_ratio_q5", "region_cat")),
            ("condition_ratio_q5|source", ("condition_ratio_q5", "source_cat")),
            ("condition_ratio_q10|region", ("condition_ratio_q10", "region_cat")),
            ("condition_ratio_q10|age", ("condition_ratio_q10", "age_range_cat")),
            ("condition_ratio_q20|region", ("condition_ratio_q20", "region_cat")),
            ("condition_ratio_q20|source", ("condition_ratio_q20", "source_cat")),
            ("ratio_q5|region", ("ratio_q5", "region_cat")),
            ("ratio_q5|source", ("ratio_q5", "source_cat")),
            ("ratio_q20|source", ("ratio_q20", "source_cat")),
            ("ratio_q10|binary", ("ratio_q10", "binary_pattern")),
            ("days_q5|condition_q5", ("days_q5", "condition_q5")),
            ("days_q20|condition_q20", ("days_q20", "condition_q20")),
            ("days_q5|condition_ratio_q5", ("days_q5", "condition_ratio_q5")),
            ("days_q10|condition_ratio_q10", ("days_q10", "condition_ratio_q10")),
            ("days_q10|condition_q10|region", ("days_q10", "condition_q10", "region_cat")),
            ("days_q10|condition_q10|source", ("days_q10", "condition_q10", "source_cat")),
            ("days_q10|condition_q10|age", ("days_q10", "condition_q10", "age_range_cat")),
            ("source|condition_q10|age", ("source_cat", "condition_q10", "age_range_cat")),
            ("region|condition_q10|age", ("region_cat", "condition_q10", "age_range_cat")),
            ("region|source|age", ("region_cat", "source_cat", "age_range_cat")),
            ("ratio_q5|region|source", ("ratio_q5", "region_cat", "source_cat")),
            ("ratio_q20|region|source", ("ratio_q20", "region_cat", "source_cat")),
            ("days_fixed|source", ("days_fixed", "source_cat")),
            ("days_fixed|condition_q10", ("days_fixed", "condition_q10")),
            ("days_fixed|condition_ratio_q10", ("days_fixed", "condition_ratio_q10")),
            ("days_fixed|region", ("days_fixed", "region_cat")),
            ("x20|age", ("x20_cat", "age_range_cat")),
            ("x19|livability", ("x19_cat", "livability_cat")),
            ("livability|age", ("livability_cat", "age_range_cat")),
            ("region|livability", ("region_cat", "livability_cat")),
            ("t3|days_q5", ("t3_cat", "days_q5")),
            ("source|x20|age", ("source_cat", "x20_cat", "age_range_cat")),
            ("region|x20|age", ("region_cat", "x20_cat", "age_range_cat")),
            ("region|source|x19", ("region_cat", "source_cat", "x19_cat")),
        )
        for name, columns in interactions:
            self._add_cross(out, name, columns)

    def _add_rate_rich_features(self, out: pd.DataFrame) -> None:
        interactions = (
            ("condition_pct_q7|source", ("condition_pct_q7", "source_cat")),
            ("condition_pct_q25|source", ("condition_pct_q25", "source_cat")),
            ("condition_pct_q7|region", ("condition_pct_q7", "region_cat")),
            ("condition_pct_q7|age", ("condition_pct_q7", "age_range_cat")),
            ("days_q7|region", ("days_q7", "region_cat")),
            ("days_q7|source", ("days_q7", "source_cat")),
            ("days_q7|age", ("days_q7", "age_range_cat")),
            ("days_q25|region", ("days_q25", "region_cat")),
            ("rate_q7|region", ("rate_q7", "region_cat")),
            ("rate_q7|source", ("rate_q7", "source_cat")),
            ("rate_q13|age", ("rate_q13", "age_range_cat")),
            ("days_q7|condition_pct_q7", ("days_q7", "condition_pct_q7")),
            ("days_q13|condition_pct_q13", ("days_q13", "condition_pct_q13")),
            ("days_q7|region|source", ("days_q7", "region_cat", "source_cat")),
            ("condition_pct_q7|region|age", ("condition_pct_q7", "region_cat", "age_range_cat")),
            ("rate_q7|region|source", ("rate_q7", "region_cat", "source_cat")),
            ("days_q7|binary", ("days_q7", "binary_pattern")),
            ("region|binary", ("region_cat", "binary_pattern")),
            ("x20|age", ("x20_cat", "age_range_cat")),
            ("region|livability", ("region_cat", "livability_cat")),
        )
        for name, columns in interactions:
            self._add_cross(out, name, columns)

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

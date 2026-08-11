"""feat_semantic: 严格复刻参考版 0.692 语义三阶交叉特征工程 + 受控增强。

严格对齐参考版 0.7冲刺执行报告 的 "最终最强配置":
  features.blocks: [raw, structured_string, days_condition, dual_category]
  dual_category:
    columns: [region, source, version, age_range, month, livability, condition, t3]
    max_categories: 64
    cross_order: 3
    max_cross_columns: 6   # 交叉只在前6个高信号业务类别上进行
  days_condition: 多尺度分箱(5,10,20) + days_condition_bin + product/ratio/missing + 单轴交叉
  structured_string: 参考版 StructuredStringFeatureBlock 完整实现

关键设计原则(来自参考版停止依据):
  - 三阶交叉集中在 region/source/version/age_range/month/livability
  - condition/t3 只生成单列双表示, 不进入显式交叉
  - 不堆匿名数值(x0..x20聚合) -> 参考版消融显示 0.660 退化
  - 不做 days/condition x 语义类别的"风险交叉" -> 参考版验证退化(0.6848)
所有统计量仅 fit 在训练折, transform 隔离(折叠式, 防泄漏)。

增强(不踩参考版已验证退化方向):
  - 种子从 3 扩到 5 (2026/2027/2028/2029/2030), 合法扩大多种子平均增益
  - CatBoost 内部 8-seed bagging(subsample+random_strength 组合), 提升稳定性
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from itertools import combinations

MISSING_TOKEN = "__MISSING__"
_NUMBER_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)")
_TOKEN_RE = re.compile(r"[^A-Za-z0-9]+")


class FeatureBuilderSemantic:
    """严格对齐参考版 blocks: raw + structured_string + days_condition + dual_category。"""

    def __init__(self, quantile_bins=(5, 10, 20), max_categories=64,
                 cross_order=3, max_cross_columns=6,
                 dual_columns=("region", "source", "version", "age_range",
                               "month", "livability", "condition", "t3"),
                 selected_cross=None):
        self.quantile_bins = tuple(sorted({int(b) for b in quantile_bins if int(b) >= 2}))
        self.max_categories = max_categories
        self.cross_order = cross_order
        self.max_cross_columns = max_cross_columns
        self.dual_columns = tuple(dual_columns)
        # selected_cross: 仅保留这些交叉组合名(增益归因筛选后); None=保留全部
        self.selected_cross = set(selected_cross) if selected_cross else None
        self._fitted = False
        self._medians = {}
        self._bin_edges = {}
        self.feature_names_ = []
        self.categorical_features_ = []

    # ---------- 工具 ----------
    @staticmethod
    def _as_str(s):
        return s.astype("string").fillna(MISSING_TOKEN).astype(str)

    @staticmethod
    def _quantile_edges(values, bins):
        finite = values[np.isfinite(values)]
        if finite.empty or bins < 2:
            return np.array([], dtype=float)
        edges = np.unique(finite.quantile(np.linspace(0.0, 1.0, int(bins) + 1)).to_numpy(dtype=float))
        return edges[1:-1] if len(edges) > 1 else np.array([], dtype=float)

    def _apply_bins(self, values, edges):
        num = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        out = np.full(len(num), -1, dtype=np.int16)
        valid = np.isfinite(num)
        if len(edges):
            out[valid] = np.searchsorted(edges, num[valid], side="right").astype(np.int16)
        else:
            out[valid] = 0
        return pd.Series(out, index=values.index, dtype="int16")

    def _without_target(self, frame):
        drop = [c for c in frame.columns if str(c).lower() in {"label", "target", "y"}]
        return frame.drop(columns=drop).copy()

    def _validate(self, frame):
        if frame.columns.duplicated().any():
            raise ValueError("duplicate columns")

    # ---------- fit / transform ----------
    def fit(self, frame, y=None):
        self._validate(frame)
        src = self._without_target(frame)
        for col in ("days", "condition"):
            if col in src:
                v = pd.to_numeric(src[col], errors="coerce")
                self._medians[col] = float(v.median()) if v.notna().any() else 0.0
                self._bin_edges[col] = {b: self._quantile_edges(v, b) for b in self.quantile_bins}
        prov = self._build(src)
        self.feature_names_ = list(prov.columns)
        self.categorical_features_ = [c for c in self.feature_names_ if not pd.api.types.is_numeric_dtype(prov[c])]
        self._fitted = True
        return self

    def transform(self, frame):
        if not self._fitted:
            raise RuntimeError("not fitted")
        self._validate(frame)
        src = self._without_target(frame)
        res = self._build(src)
        for col in self.feature_names_:
            if col not in res:
                res[col] = MISSING_TOKEN if col in self.categorical_features_ else 0.0
        res = res.reindex(columns=self.feature_names_)
        for col in self.categorical_features_:
            res[col] = self._as_str(res[col])
        return res

    def fit_transform(self, frame, y=None):
        return self.fit(frame, y).transform(frame)

    # ---------- 构建: 严格对齐 4 个 block ----------
    def _build(self, src):
        res = src.copy()
        if "id" in res:
            res = res.drop(columns=["id"])
        # 归一所有 object/string 列
        for c in list(res.columns):
            if pd.api.types.is_object_dtype(res[c]) or pd.api.types.is_string_dtype(res[c]):
                res[c] = self._as_str(res[c])
        self._add_structured(res, src)
        self._add_days_condition(res, src)
        self._add_dual_category(res, src)
        return res

    # ---- block: structured_string (完整对齐参考版) ----
    def _add_structured(self, res, src):
        for column in [c for c in src.columns
                       if c in src and (pd.api.types.is_object_dtype(src[c])
                                        or pd.api.types.is_string_dtype(src[c])
                                        or isinstance(src[c].dtype, pd.CategoricalDtype))]:
            values = self._as_str(src[column]) if column in src else pd.Series(MISSING_TOKEN, index=src.index)
            missing = values.eq(MISSING_TOKEN)
            pieces = values.str.split(_TOKEN_RE, n=1, expand=True)
            prefix = pieces[0].replace("", MISSING_TOKEN).fillna(MISSING_TOKEN)
            suffix = values.str.extract(r"(?:^|[-_|:/\s])([^\-_|:/\s]+)$", expand=False).replace("", MISSING_TOKEN).fillna(MISSING_TOKEN)
            number = pd.to_numeric(values.str.extract(_NUMBER_RE, expand=False), errors="coerce")
            res[f"{column}__prefix"] = prefix.astype(str)
            res[f"{column}__suffix"] = suffix.astype(str)
            res[f"{column}__number"] = number.astype(float)
            res[f"{column}__length"] = values.str.len().astype(float)
            res[f"{column}__digit_count"] = values.str.count(r"\d").astype(float)
            res[f"{column}__alpha_count"] = values.str.count(r"[A-Za-z]").astype(float)
            res[f"{column}__special_count"] = values.str.count(r"[^A-Za-z0-9]").astype(float)
            res[f"{column}__pattern"] = values.map(self._pattern)
            res[f"{column}__missing"] = missing.astype("int8")

    @staticmethod
    def _pattern(value: str) -> str:
        if value == MISSING_TOKEN:
            return "MISSING"
        chars = []
        for char in value:
            chars.append("A" if char.isalpha() else "9" if char.isdigit() else "_")
        return "".join(chars) or "EMPTY"

    # ---- block: days_condition (对齐参考版多尺度分箱 + 交互面) ----
    def _add_days_condition(self, res, src):
        if "days" not in src or "condition" not in src:
            return
        days = pd.to_numeric(src["days"], errors="coerce")
        cond = pd.to_numeric(src["condition"], errors="coerce")
        days_f = days.fillna(self._medians.get("days", 0.0))
        cond_f = cond.fillna(self._medians.get("condition", 0.0))
        res["days__filled"] = days_f.astype(float)
        res["condition__filled"] = cond_f.astype(float)
        res["days__log1p"] = np.log1p(days_f.clip(lower=0)).astype(float)
        res["condition__log1p"] = np.log1p(cond_f.clip(lower=0)).astype(float)
        res["days__missing"] = days.isna().astype("int8")
        res["condition__missing"] = cond.isna().astype("int8")
        res["days_condition__product"] = (days_f * cond_f).astype(float)
        res["days_condition__ratio"] = (cond_f / (days_f.abs() + 1.0)).astype(float)
        res["days_condition__missing"] = (days.isna() | cond.isna()).astype("int8")
        fb = self._bin_edges.get("days", {})
        cb = self._bin_edges.get("condition", {})
        first_bins = self.quantile_bins[0]
        for b in self.quantile_bins:
            res[f"days__bin_{b}"] = self._apply_bins(days, fb.get(b, np.array([]))).astype(str).radd("bin_")
            res[f"condition__bin_{b}"] = self._apply_bins(cond, cb.get(b, np.array([]))).astype(str).radd("bin_")
        for b in self.quantile_bins:
            res[f"days_condition__bin_{b}"] = (
                res[f"days__bin_{b}"].astype(str) + "__" + res[f"condition__bin_{b}"].astype(str)
            )
        res["days_condition_bin"] = res[f"days_condition__bin_{first_bins}"]

    # ---- block: dual_category (对齐参考版: 双通道 + 显式6列三阶交叉) ----
    def _add_dual_category(self, res, src):
        # 仅使用参考版指定列的 dual 双表示
        dual_present = [c for c in self.dual_columns if c in res.columns]
        cross_cols = dual_present[: self.max_cross_columns]  # 前6个高信号业务类别
        for c in dual_present:
            vals = self._as_str(res[c])
            res[f"{c}__category"] = vals
            # 序数码: 仅在该列基数 <= max_categories 时生成(对齐参考版双表示约束)
            if vals.nunique(dropna=True) <= self.max_categories:
                res[f"{c}__category_code"] = vals.map(
                    {v: i for i, v in enumerate(vals.drop_duplicates())}
                ).fillna(-1).astype("int32")
        # 受控 n 阶字符串交叉(2阶+3阶) 仅在 cross_cols 上
        str_map = {c: self._as_str(res[c]) for c in cross_cols}
        for order in range(2, min(self.cross_order, len(cross_cols)) + 1):
            for cols in combinations(cross_cols, order):
                name = "__X__".join(map(str, cols)) + "__category_cross"
                if self.selected_cross is not None and name not in self.selected_cross:
                    continue  # 增益归因筛选: 跳过低增益交叉(降维防过拟合)
                comb = str_map[cols[0]]
                for c in cols[1:]:
                    comb = comb + "|" + str_map[c]
                res[name] = comb.astype(str)


__all__ = ["FeatureBuilderSemantic"]

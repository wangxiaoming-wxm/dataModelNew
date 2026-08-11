"""折内 TE 的防泄漏与确定性测试。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src_super.features_te import FEATURE_NAME, build_source_days_te


class SourceDaysTargetEncodingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fit = pd.DataFrame(
            {
                "source": [f"source_{index}" for index in range(40)],
                "days": np.linspace(1.0, 400.0, 40),
            }
        )
        self.y = np.asarray([0, 1] * 20)
        self.valid = pd.DataFrame(
            {"source": ["never_seen"], "days": [10_000.0]}
        )
        self.test = pd.DataFrame(
            {"source": ["also_unseen"], "days": [20_000.0]}
        )

    def test_training_rows_receive_inner_oof_values(self) -> None:
        encoded = build_source_days_te(
            self.fit,
            self.y,
            self.valid,
            (self.test,),
            inner_splits=4,
            inner_seed=2026,
        )

        self.assertEqual(encoded.fit.name, FEATURE_NAME)
        self.assertEqual(len(encoded.fit), len(self.fit))
        self.assertTrue(np.isfinite(encoded.fit).all())
        # 每行 source 唯一，内层映射必然 unseen，只能回退到排除该行后的 prior。
        self.assertTrue(((encoded.fit > 0.0) & (encoded.fit < 1.0)).all())

    def test_unseen_validation_and_test_keys_fall_back_to_outer_prior(self) -> None:
        encoded = build_source_days_te(
            self.fit,
            self.y,
            self.valid,
            (self.test,),
            inner_splits=4,
            inner_seed=2026,
        )

        self.assertAlmostEqual(encoded.prior, 0.5)
        self.assertAlmostEqual(float(encoded.valid.iloc[0]), encoded.prior)
        self.assertAlmostEqual(float(encoded.others[0].iloc[0]), encoded.prior)

    def test_encoding_is_deterministic(self) -> None:
        first = build_source_days_te(
            self.fit,
            self.y,
            self.valid,
            inner_splits=4,
            inner_seed=2030,
        )
        second = build_source_days_te(
            self.fit,
            self.y,
            self.valid,
            inner_splits=4,
            inner_seed=2030,
        )

        np.testing.assert_array_equal(first.days_edges, second.days_edges)
        np.testing.assert_allclose(first.fit, second.fit)
        np.testing.assert_allclose(first.valid, second.valid)


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src_rebuild.cli import available_blends, available_stacks, configs_for_run
from src_rebuild.evaluation import (
    BlendSpec,
    CandidateScore,
    StackSpec,
    make_stratified_splits,
    rank_average,
    select_candidate,
)
from src_rebuild.features import RebuildFeatureBuilder
from src_rebuild.io import save_submission
from src_rebuild.models import candidate_configs


def tiny_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["0001020304050607", "1011121314151617", "2021222324252627", "3031323334353637"],
            "month": ["m1", "m1", "m2", "m3"],
            "days": [1.0, 4.0, 9.0, 16.0],
            "region": ["r1", "r1", "r2", "r3"],
            "cc": [1.0, 2.0, 3.0, 4.0],
            "condition": [0.2, 0.4, 0.8, 1.6],
            "t1": [0, 1, 0, 1],
            "t2": [1, 0, 1, 0],
            "t3": ["a", "a", "b", "c"],
            "r1": [0, 1, 0, 1],
            "r2": [1, 0, 1, 0],
            "V": [1.0, 2.0, 3.0, 4.0],
            "code": ["c1", "c1", "c2", "c3"],
            **{f"x{i}": np.arange(4, dtype=float) + i for i in range(21)},
            "c1": [0, 1, 0, 1],
            "c2": [1, 0, 1, 0],
            "max_g": [2.0, 3.0, 4.0, 5.0],
            "age_range": [1, 2, 3, 4],
            "livability": [1.0, 2.0, 3.0, 4.0],
            "source": ["s1", "s1", "s2", "s3"],
            "grades": ["g1", "g1", "g2", "g3"],
            "w1": [0, 1, 0, 1],
            "w2": [1, 0, 1, 0],
            "version": ["v1", "v1", "v2", "v3"],
            "label": [0, 1, 0, 1],
        }
    )


class FeatureBuilderTests(unittest.TestCase):
    def test_core_is_label_free_and_excludes_identifier_worlds(self):
        frame = tiny_frame()
        matrix = RebuildFeatureBuilder("core").fit_transform(frame.iloc[:3])

        self.assertNotIn("label", matrix.frame.columns)
        self.assertNotIn("id", matrix.frame.columns)
        self.assertNotIn("x0", matrix.frame.columns)
        self.assertFalse(any(column.startswith("id_") for column in matrix.frame.columns))
        self.assertIn("days_by_source_dev", matrix.frame.columns)

    def test_all_id_adds_label_free_bytes_and_handles_unseen_groups(self):
        frame = tiny_frame()
        builder = RebuildFeatureBuilder("all_id").fit(frame.iloc[:3])
        transformed = builder.transform(frame.iloc[3:])

        self.assertIn("x0", transformed.frame.columns)
        self.assertIn("id_byte_0", transformed.frame.columns)
        self.assertIn("id_nibble_15", transformed.frame.columns)
        self.assertTrue(np.isfinite(transformed.frame["source_freq"]).all())
        self.assertEqual(float(transformed.frame["source_freq"].iloc[0]), 0.0)

    def test_transform_schema_matches_fit_schema(self):
        frame = tiny_frame()
        builder = RebuildFeatureBuilder("all").fit(frame.iloc[:3])
        train_matrix = builder.transform(frame.iloc[:3])
        valid_matrix = builder.transform(frame.iloc[3:])

        self.assertEqual(list(train_matrix.frame.columns), list(valid_matrix.frame.columns))
        self.assertEqual(train_matrix.cat_columns, valid_matrix.cat_columns)

    def test_ratio_world_fits_bins_and_source_scale_on_training_partition(self):
        frame = tiny_frame()
        builder = RebuildFeatureBuilder("ratio").fit(frame.iloc[:3])
        transformed = builder.transform(frame.iloc[3:])

        self.assertIn("condition_source_ratio", transformed.frame.columns)
        self.assertIn("ratio_q10", transformed.frame.columns)
        self.assertIn("ratio_q10|region", transformed.frame.columns)
        self.assertIn("ratio_q10", transformed.cat_columns)

    def test_rate_world_uses_training_source_distribution_with_unseen_fallback(self):
        frame = tiny_frame()
        builder = RebuildFeatureBuilder("rate").fit(frame.iloc[:3])
        transformed = builder.transform(frame.iloc[3:])

        self.assertIn("condition_source_pct", transformed.frame.columns)
        self.assertIn("exposure_rate", transformed.frame.columns)
        self.assertAlmostEqual(float(transformed.frame["condition_source_pct"].iloc[0]), 0.5)

    def test_rich_worlds_add_only_label_free_pre_registered_interactions(self):
        frame = tiny_frame()
        ratio = RebuildFeatureBuilder("ratio_rich").fit_transform(frame.iloc[:3])
        rate = RebuildFeatureBuilder("rate_rich").fit_transform(frame.iloc[:3])

        self.assertIn("days_fixed", ratio.frame.columns)
        self.assertIn("ratio_q20|region|source", ratio.frame.columns)
        self.assertIn("condition_ratio_q20|source", ratio.frame.columns)
        self.assertIn("rate_q7|region|source", rate.frame.columns)
        self.assertIn("condition_pct_q25|source", rate.frame.columns)
        self.assertNotIn("label", ratio.frame.columns)
        self.assertNotIn("label", rate.frame.columns)


class EvaluationTests(unittest.TestCase):
    def test_full_entry_defaults_to_locked_finalists(self):
        names = [config.name for config in configs_for_run("full", None)]
        self.assertEqual(names, ["cb_ratio_rmse_d5", "cb_rate_rmse_d6"])

    def test_rich_configs_lock_boosting_types_and_blend_grid(self):
        configs = candidate_configs("smoke")
        by_name = {config.name: config for config in configs}

        self.assertEqual(by_name["cb_ratio_rich_rmse_d5"].boosting_type, "Ordered")
        self.assertEqual(by_name["cb_rate_rich_rmse_d6"].boosting_type, "Plain")
        blend_names = {blend.name for blend in available_blends(configs)}
        self.assertIn("blend_rich_ratio_rate_w35", blend_names)
        self.assertIn("blend_rich_ratio_rate_w50", blend_names)
        self.assertIn("blend_rich_ratio_rate_w65", blend_names)

    def test_selector_requires_margin_for_more_complex_candidate(self):
        candidates = [
            CandidateScore("simple", inner_auc=0.7000, complexity=0),
            CandidateScore("complex", inner_auc=0.7004, complexity=1),
        ]
        selected = select_candidate(candidates, minimum_complex_gain=0.0005)
        self.assertEqual(selected.name, "simple")

        candidates[1] = CandidateScore("complex", inner_auc=0.7006, complexity=1)
        selected = select_candidate(candidates, minimum_complex_gain=0.0005)
        self.assertEqual(selected.name, "complex")

    def test_stratified_splits_cover_each_row_once_without_overlap(self):
        y = np.array([0] * 15 + [1] * 5)
        splits = make_stratified_splits(y, n_splits=5, seed=2026)
        seen = np.zeros(len(y), dtype=int)
        for train_indices, valid_indices in splits:
            self.assertEqual(len(np.intersect1d(train_indices, valid_indices)), 0)
            seen[valid_indices] += 1
        np.testing.assert_array_equal(seen, np.ones(len(y), dtype=int))

    def test_rank_average_is_monotonic_and_deterministic(self):
        first = np.array([0.1, 0.3, 0.2])
        second = np.array([10.0, 30.0, 20.0])
        result = rank_average([first, second])
        np.testing.assert_allclose(result, np.array([1 / 3, 1.0, 2 / 3]))

    def test_blend_spec_combines_only_pre_registered_components(self):
        blend = BlendSpec(
            name="ratio_rate_mean",
            components=("ratio", "rate"),
            weights=(0.5, 0.5),
            complexity=2,
        )
        prediction = blend.combine(
            {
                "ratio": np.array([0.2, 0.8]),
                "rate": np.array([0.4, 0.6]),
            }
        )
        np.testing.assert_allclose(prediction, np.array([0.3, 0.7]))
        with self.assertRaises(KeyError):
            blend.combine({"ratio": np.array([0.2, 0.8])})

    def test_stack_spec_cross_fits_meta_model_and_records_coefficients(self):
        y = np.array([0, 1] * 10)
        predictions = {
            "ratio": np.linspace(0.05, 0.95, len(y)),
            "rate": np.roll(np.linspace(0.05, 0.95, len(y)), 2),
        }
        splits = make_stratified_splits(y, n_splits=2, seed=2026)
        stack = StackSpec(
            name="stack",
            components=("ratio", "rate"),
            regularization_c=0.1,
            complexity=3,
        )

        oof, coefficients = stack.cross_fit(predictions, y, splits)
        final_prediction, final_coefficients = stack.fit_predict(
            predictions,
            y,
            {"ratio": predictions["ratio"][:3], "rate": predictions["rate"][:3]},
        )

        self.assertEqual(len(oof), len(y))
        self.assertTrue(np.isfinite(oof).all())
        self.assertEqual(len(coefficients), 2)
        self.assertEqual(len(final_coefficients), 2)
        self.assertEqual(len(final_prediction), 3)

    def test_rich_stack_is_available_only_with_both_components(self):
        configs = candidate_configs("smoke")
        stacks = available_stacks(configs)
        self.assertEqual([stack.name for stack in stacks], ["stack_rich_ratio_rate_logit"])


class SubmissionTests(unittest.TestCase):
    def test_save_submission_preserves_ids_and_bounds_values(self):
        sample = pd.DataFrame({"id": ["a", "b"], "label": [0.0, 0.0]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.csv"
            digest = save_submission(sample, np.array([-1.0, 2.0]), path)
            saved = pd.read_csv(path, dtype={"id": str})

        self.assertEqual(saved["id"].tolist(), ["a", "b"])
        self.assertEqual(saved["label"].tolist(), [0.001, 0.999])
        self.assertEqual(len(digest), 64)


if __name__ == "__main__":
    unittest.main()

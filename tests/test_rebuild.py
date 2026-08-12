import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src_rebuild.evaluation import CandidateScore, make_stratified_splits, rank_average, select_candidate
from src_rebuild.features import RebuildFeatureBuilder
from src_rebuild.io import save_submission


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


class EvaluationTests(unittest.TestCase):
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

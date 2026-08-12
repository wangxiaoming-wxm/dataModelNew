import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


class Vz20FrozenFusionTests(unittest.TestCase):
    def test_am40_and_w62_anchors(self):
        y = pd.read_csv(ROOT / "data" / "train.csv")["label"].astype(int).to_numpy()
        oof = np.load(ROOT / "artifacts" / "super714" / "best_v1_oof.npy", allow_pickle=True).item()
        main = np.asarray(oof["main"], float)
        alt = np.asarray(oof["alt"], float)
        w62 = 0.62 * main + 0.38 * alt
        am40 = 0.40 * np.maximum(main, alt) + 0.60 * w62
        self.assertAlmostEqual(roc_auc_score(y, w62), 0.7015936597140784, places=12)
        self.assertAlmostEqual(roc_auc_score(y, am40), 0.7018113510376338, places=12)

    def test_metrics_json_after_build(self):
        path = ROOT / "vz20" / "artifacts" / "metrics.json"
        if not path.is_file():
            self.skipTest("run build_vz20.py first")
        metrics = json.loads(path.read_text())
        self.assertEqual(metrics["path_to_0.72"], False)
        self.assertEqual(metrics["path_to_0.749"], False)
        self.assertGreater(metrics["oof_auc"], metrics["w62_oof_auc"])


if __name__ == "__main__":
    unittest.main()

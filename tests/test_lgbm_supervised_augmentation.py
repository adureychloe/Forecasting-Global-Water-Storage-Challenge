import numpy as np
import pandas as pd
from unittest import TestCase
from unittest.mock import patch

import gpu_lgbm_horizon as lgbm_runner


class FullFeatureAugmentationTest(TestCase):
    def test_full_features_use_supervised_augmentation(self):
        calls = []

        def augment(x_fit, y_fit, x_query, fit_raw, query_raw):
            calls.append((len(y_fit), len(fit_raw), len(query_raw)))
            x_fit = x_fit.copy()
            x_query = x_query.copy()
            x_fit["memory"] = 1.0
            x_query["memory"] = 2.0
            return x_fit, x_query

        x_fit = pd.DataFrame({"base": [0.0, 1.0]})
        x_query = pd.DataFrame({"base": [2.0]})
        fit_raw = pd.DataFrame({"time": ["2010-01-01", "2010-02-01"]})
        query_raw = pd.DataFrame({"time": ["2010-03-01"]})

        with patch.object(lgbm_runner.runner, "augment_supervised_features", augment):
            fit, query = lgbm_runner.augment_full_features(
                x_fit, np.array([0.1, 0.2], dtype=np.float32), x_query, fit_raw, query_raw
            )

        self.assertEqual(calls, [(2, 2, 1)])
        self.assertEqual(fit["memory"].tolist(), [1.0, 1.0])
        self.assertEqual(query["memory"].tolist(), [2.0])

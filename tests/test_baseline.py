import unittest
import numpy as np
import pandas as pd

from baseline import build_features, make_submission_frame, make_time_split, make_regressor


class BaselineBehaviorTests(unittest.TestCase):
    def test_features_use_only_current_or_past_rows(self):
        frame = pd.DataFrame(
            {
                "LAT": [0.0, 0.0, 0.0],
                "LON": [10.0, 10.0, 10.0],
                "DATE": ["2020-01-01", "2020-02-01", "2020-03-01"],
                "TWS_t": [1.0, 2.0, np.nan],
                "SPEI_1": [0.1, 0.2, 0.3],
            }
        )
        features = build_features(frame)
        self.assertEqual(features.loc[1, "TWS_t_lag1"], 1.0)
        self.assertTrue(np.isnan(features.loc[0, "TWS_t_lag1"]))
        self.assertEqual(features.loc[2, "TWS_t_lag1"], 2.0)
        self.assertNotIn("TWS_t_lead1", features.columns)

    def test_time_split_keeps_validation_dates_after_training_dates(self):
        frame = pd.DataFrame({"DATE": pd.date_range("2020-01-01", periods=10)})
        train_idx, valid_idx = make_time_split(frame, valid_fraction=0.2)
        self.assertLess(frame.loc[train_idx, "DATE"].max(), frame.loc[valid_idx, "DATE"].min())

    def test_long_lags_are_past_only(self):
        frame = pd.DataFrame(
            {
                "lat": [1.0] * 5,
                "lon": [2.0] * 5,
                "time": pd.date_range("2020-01-01", periods=5, freq="MS"),
                "TWS_t": [1.0, 2.0, 3.0, 4.0, 5.0],
                "SPEI_03_t": [0.0, 1.0, 2.0, 3.0, 4.0],
            }
        )
        features = build_features(frame)
        self.assertTrue(np.isnan(features.loc[2, "TWS_t_lag3"]))
        self.assertEqual(features.loc[3, "TWS_t_lag3"], 1.0)
        self.assertEqual(features.loc[4, "TWS_t_lag3"], 2.0)
        self.assertEqual(features.loc[4, "SPEI_03_t_roll3"], 2.0)

    def test_model_factory_exposes_tree_baselines(self):
        self.assertEqual(make_regressor("hgb").__class__.__name__, "HistGradientBoostingRegressor")
        self.assertEqual(make_regressor("extra_trees").__class__.__name__, "ExtraTreesRegressor")

    def test_submission_preserves_sample_id_order(self):
        test = pd.DataFrame({"ID": ["b", "a"], "lat": [0, 0], "lon": [0, 1]})
        sample = pd.DataFrame({"ID": ["a", "b"], "Target": [0.0, 0.0]})
        submission = make_submission_frame(test, np.array([2.0, 1.0]), sample)
        self.assertEqual(submission["ID"].tolist(), ["a", "b"])
        self.assertEqual(submission["Target"].tolist(), [1.0, 2.0])

    def test_masked_tws_lags_forward_fill_last_observation_only(self):
        frame = pd.DataFrame(
            {
                "lat": [1.0] * 4,
                "lon": [2.0] * 4,
                "time": pd.date_range("2020-01-01", periods=4, freq="MS"),
                "TWS_t": [1.0, np.nan, np.nan, 4.0],
            }
        )
        features = build_features(frame)
        self.assertEqual(features.loc[2, "TWS_t_ffill"], 1.0)
        self.assertEqual(features.loc[2, "TWS_t_lag1"], 1.0)
        self.assertEqual(features.loc[3, "TWS_t_lag1"], 1.0)
        self.assertEqual(features.loc[1, "TWS_t_missing"], 1.0)
        self.assertEqual(features.loc[2, "TWS_t_age"], 2.0)
        self.assertEqual(features.loc[3, "TWS_t_age"], 0.0)



if __name__ == "__main__":
    unittest.main()

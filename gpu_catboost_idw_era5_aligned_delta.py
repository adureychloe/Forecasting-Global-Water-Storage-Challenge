"""2010+ CatBoost aligned walk trained on last-obs delta, not cell-mean center.

Centering (#256) failed online. This subtracts the current persist/ffill only.
"""

import numpy as np

import gpu_catboost_idw_era5_aligned as base


def persist_col(features):
    for column in ("TWS_t_ffill", "tws_neighbor_anchor_mean"):
        if column in features.columns:
            return features[column].to_numpy(dtype=np.float32)
    return np.zeros(len(features), dtype=np.float32)


def transform_targets(target, features):
    return np.asarray(target, dtype=np.float32) - persist_col(features)


def inverse_predictions(prediction, features):
    return np.asarray(prediction, dtype=np.float32) + persist_col(features)


base.runner.transform_targets = transform_targets
base.runner.inverse_predictions = inverse_predictions
base.runner.PREFIX = "catboost_horizon_idw_era5_aligned_delta"
base.runner.LOG = base.runner.RESULTS / f"{base.runner.PREFIX}.jsonl"
if __name__ == "__main__":
    raise SystemExit(base.custom_main())

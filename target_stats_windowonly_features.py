"""Recent target windows without the train/query-mismatched last-target feature."""

import numpy as np

import target_stats_features
import target_stats_recent_features


def augment_features(x_fit, y_fit, x_query, fit_raw, query_raw):
    x_fit, x_query = target_stats_features.augment_features(
        x_fit, y_fit, x_query, fit_raw, query_raw
    )
    fit_extra = target_stats_recent_features._training_recent(fit_raw, y_fit)
    query_extra = target_stats_recent_features._query_recent(fit_raw, y_fit, query_raw)
    columns = [f"target_loc_roll{window}_{stat}" for window in (12, 24) for stat in ("mean", "std")]
    for column in columns:
        x_fit[column] = fit_extra[column].fillna(0).to_numpy(dtype=np.float32)
        x_query[column] = query_extra[column].fillna(0).to_numpy(dtype=np.float32)
    return x_fit, x_query

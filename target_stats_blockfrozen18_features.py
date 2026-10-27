"""Query-length 18-month block-frozen target history."""

import numpy as np

import target_stats_blockfrozen_features as block_features
import target_stats_features


BLOCK_MONTHS = 18


def augment_features(x_fit, y_fit, x_query, fit_raw, query_raw):
    x_fit, x_query = target_stats_features.augment_features(
        x_fit, y_fit, x_query, fit_raw, query_raw
    )
    fit_extra = block_features._training_block_features(
        fit_raw, y_fit, query_raw, block_months=BLOCK_MONTHS
    )
    query_extra = block_features._query_block_features(fit_raw, y_fit, query_raw)
    for column in block_features.FEATURES:
        x_fit[column] = fit_extra[column].fillna(0).to_numpy(dtype=np.float32)
        x_query[column] = query_extra[column].fillna(0).to_numpy(dtype=np.float32)
    return x_fit, x_query

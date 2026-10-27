"""Block-frozen target history aligned to each query's calendar span."""

import numpy as np
import pandas as pd

import target_stats_blockfrozen_features as block_features
import target_stats_features


def query_span_months(query_raw) -> int:
    time = pd.to_datetime(query_raw["time"])
    ordinal = time.dt.year.to_numpy(dtype=int) * 12 + time.dt.month.to_numpy(dtype=int)
    return int(ordinal.max() - ordinal.min() + 1)


def augment_features(x_fit, y_fit, x_query, fit_raw, query_raw):
    x_fit, x_query = target_stats_features.augment_features(
        x_fit, y_fit, x_query, fit_raw, query_raw
    )
    fit_extra = block_features._training_block_features(
        fit_raw,
        y_fit,
        query_raw,
        block_months=query_span_months(query_raw),
    )
    query_extra = block_features._query_block_features(fit_raw, y_fit, query_raw)
    for column in block_features.FEATURES:
        x_fit[column] = fit_extra[column].fillna(0).to_numpy(dtype=np.float32)
        x_query[column] = query_extra[column].fillna(0).to_numpy(dtype=np.float32)
    return x_fit, x_query

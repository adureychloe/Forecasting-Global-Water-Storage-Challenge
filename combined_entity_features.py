"""Compose block-frozen target memory with label-free categorical locations."""

import categorical_location_features
import target_stats_blockfrozen_features


def augment_features(x_fit, y_fit, x_query, fit_raw, query_raw):
    x_fit, x_query = target_stats_blockfrozen_features.augment_features(
        x_fit, y_fit, x_query, fit_raw, query_raw
    )
    return categorical_location_features.augment_features(
        x_fit, y_fit, x_query, fit_raw, query_raw
    )

"""Label-free categorical location identifiers for CatBoost Ordered CTRs."""

from __future__ import annotations

import numpy as np


def augment_features(x_fit, y_fit, x_query, fit_raw, query_raw):
    del y_fit
    for features, raw in ((x_fit, fit_raw), (x_query, query_raw)):
        lat = raw["lat"].to_numpy(dtype=float)
        lon = raw["lon"].to_numpy(dtype=float)
        features["loc_id"] = [f"{a:.5f}_{b:.5f}" for a, b in zip(lat, lon)]
        features["lat_band_cat"] = np.floor((lat + 90.0) / 10.0).astype(int).astype(str)
        features["lon_band_cat"] = np.floor((lon + 180.0) / 15.0).astype(int).astype(str)
    return x_fit, x_query

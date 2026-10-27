"""Fit-only ERA5 climate clusters. lat/lon are lookup keys, not model features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from era5_cell_features import load_or_build_table

N_CLUSTERS = 8
COLS = ("t2m", "wb", "soil_water", "sd")


def augment_features(x_fit, x_query, fit_raw, query_raw):
    table = load_or_build_table(pd.concat([fit_raw, query_raw], ignore_index=True))
    fit = fit_raw.copy()
    fit["_month_key"] = fit.time.dt.year.to_numpy(np.int32) * 12 + fit.time.dt.month.to_numpy(np.int32)
    joined = fit[["lat", "lon", "_month_key"]].merge(
        table[["lat", "lon", "_month_key", *COLS]],
        on=["lat", "lon", "_month_key"],
        how="left",
    )
    cell = joined.groupby(["lat", "lon"], sort=False)[list(COLS)].mean()
    matrix = cell.to_numpy(dtype=np.float64)
    matrix = np.where(np.isfinite(matrix), matrix, 0.0)
    scale = matrix.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    normed = (matrix - matrix.mean(axis=0)) / scale
    labels = KMeans(n_clusters=N_CLUSTERS, random_state=2027, n_init=10).fit_predict(normed)
    lookup = cell.index.to_frame(index=False)
    lookup["era5_clim_cluster"] = labels.astype(np.float32)
    x_fit = x_fit.copy()
    x_query = x_query.copy()
    x_fit["era5_clim_cluster"] = (
        fit_raw[["lat", "lon"]].merge(lookup, on=["lat", "lon"], how="left")["era5_clim_cluster"].to_numpy(np.float32)
    )
    x_query["era5_clim_cluster"] = (
        query_raw[["lat", "lon"]].merge(lookup, on=["lat", "lon"], how="left")["era5_clim_cluster"].to_numpy(np.float32)
    )
    x_fit["era5_clim_cluster"] = x_fit["era5_clim_cluster"].fillna(0)
    x_query["era5_clim_cluster"] = x_query["era5_clim_cluster"].fillna(0)
    return x_fit, x_query

"""Fit-only calendar-month ERA5 climatology anomalies. lat/lon are grouping keys only."""

from __future__ import annotations

import numpy as np
import pandas as pd

BASE_COLS = (
    "era5_tp",
    "era5_e",
    "era5_ro",
    "era5_pev",
    "era5_sf",
    "era5_sd",
    "era5_t2m",
    "era5_wb",
    "era5_soil_water",
    "era5_tp_month",
)


def augment_features(x_fit, x_query, fit_raw, query_raw, lags=()):
    cols = [column for column in BASE_COLS if column in x_fit.columns]
    if not cols:
        return x_fit, x_query
    month_fit = pd.to_datetime(fit_raw["time"]).dt.month.to_numpy(dtype=np.int16)
    month_query = pd.to_datetime(query_raw["time"]).dt.month.to_numpy(dtype=np.int16)
    work = pd.DataFrame(
        {
            "lat": fit_raw["lat"].to_numpy(dtype=np.float64),
            "lon": fit_raw["lon"].to_numpy(dtype=np.float64),
            "month": month_fit,
        }
    )
    for column in cols:
        work[column] = x_fit[column].to_numpy(dtype=np.float32)
    clim = work.groupby(["lat", "lon", "month"], sort=False)[list(cols)].mean().reset_index()

    def apply(features, raw, month):
        key = pd.DataFrame(
            {
                "lat": raw["lat"].to_numpy(dtype=np.float64),
                "lon": raw["lon"].to_numpy(dtype=np.float64),
                "month": month,
            }
        )
        mapped = key.merge(clim, on=["lat", "lon", "month"], how="left")
        out = features.copy()
        for column in cols:
            out[f"{column}_anom"] = (
                features[column].to_numpy(dtype=np.float32) - mapped[column].to_numpy(dtype=np.float32)
            )
        for lag in lags:
            lag_month = ((month.astype(np.int16) - 1 - int(lag)) % 12) + 1
            lag_key = pd.DataFrame(
                {
                    "lat": raw["lat"].to_numpy(dtype=np.float64),
                    "lon": raw["lon"].to_numpy(dtype=np.float64),
                    "month": lag_month,
                }
            )
            lag_mapped = lag_key.merge(clim, on=["lat", "lon", "month"], how="left")
            for column in cols:
                source = f"{column}_lag{int(lag)}"
                if source not in features.columns:
                    continue
                out[f"{column}_lag{int(lag)}_anom"] = (
                    features[source].to_numpy(dtype=np.float32)
                    - lag_mapped[column].to_numpy(dtype=np.float32)
                )
        return out

    return apply(x_fit, fit_raw, month_fit), apply(x_query, query_raw, month_query)

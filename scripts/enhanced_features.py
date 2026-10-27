"""Leakage-safe exact-month features inspired by the public reference pipeline."""

import numpy as np
import pandas as pd

from baseline import build_features as base_build_features


def _exact_merge(frame: pd.DataFrame, observed: pd.DataFrame, value_col: str, lag: int) -> np.ndarray:
    key = frame[["lat", "lon", "time"]].copy()
    key["lookup_time"] = pd.to_datetime(key["time"]) - pd.DateOffset(months=lag)
    right = observed[["lat", "lon", "time", value_col]].rename(
        columns={"time": "lookup_time", value_col: "_value"}
    )
    merged = key.merge(right, on=["lat", "lon", "lookup_time"], how="left", sort=False)
    return merged["_value"].to_numpy()


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = base_build_features(frame)
    if not {"lat", "lon", "time"}.issubset(frame.columns):
        return result

    work = frame.copy()
    work["time"] = pd.to_datetime(work["time"], errors="coerce")
    observed_tws = work[["lat", "lon", "time", "TWS_t"]].dropna(subset=["TWS_t"])
    exact = {}
    for lag in (1, 2, 3):
        exact[lag] = _exact_merge(work, observed_tws, "TWS_t", lag)
        result[f"TWS_exact_lag{lag}"] = exact[lag]

    lag_frame = pd.DataFrame(exact)
    result["TWS_exact_roll_mean3"] = lag_frame.mean(axis=1, skipna=True).to_numpy()
    result["TWS_exact_roll_std3"] = lag_frame.std(axis=1, skipna=True).to_numpy()
    result["TWS_exact_momentum13"] = (lag_frame[1] - lag_frame[3]).to_numpy()

    spei_cols = [c for c in ("SPEI_01_t", "SPEI_03_t", "SPEI_06_t", "SPEI_12_t") if c in work.columns]
    if spei_cols:
        vals = work[spei_cols]
        if {"SPEI_01_t", "SPEI_12_t"}.issubset(vals.columns):
            result["SPEI_div_1_12"] = (vals["SPEI_01_t"] - vals["SPEI_12_t"]).to_numpy()
        if {"SPEI_03_t", "SPEI_12_t"}.issubset(vals.columns):
            result["SPEI_div_3_12"] = (vals["SPEI_03_t"] - vals["SPEI_12_t"]).to_numpy()
        result["SPEI_mean_all"] = vals.mean(axis=1).to_numpy()
        result["SPEI_all_negative"] = (vals.lt(0).all(axis=1)).astype(np.int8).to_numpy()
        if "SPEI_03_t" in vals and "SPEI_12_t" in vals:
            prev = _exact_merge(work, work[["lat", "lon", "time", "SPEI_03_t"]].dropna(), "SPEI_03_t", 1)
            result["SPEI_03_change"] = (vals["SPEI_03_t"].to_numpy() - prev)

    if "SOIL_MOISTURE_t" in work.columns:
        soil_prev = _exact_merge(
            work,
            work[["lat", "lon", "time", "SOIL_MOISTURE_t"]].dropna(),
            "SOIL_MOISTURE_t",
            1,
        )
        result["SOIL_MOISTURE_change"] = work["SOIL_MOISTURE_t"].to_numpy() - soil_prev
        if "SPEI_03_t" in work.columns:
            result["SOIL_SPEI03_interaction"] = work["SOIL_MOISTURE_t"].to_numpy() * work["SPEI_03_t"].to_numpy()

    return result

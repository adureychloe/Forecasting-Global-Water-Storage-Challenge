"""Recent-window extensions to leakage-safe coordinate target statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd

import target_stats_features


KEYS = ["lat", "lon"]


def _training_recent(raw: pd.DataFrame, target: np.ndarray) -> pd.DataFrame:
    table = raw[["lat", "lon", "time"]].reset_index(drop=True).copy()
    table["time"] = pd.to_datetime(table["time"])
    table["_target"] = np.asarray(target, dtype=np.float32)
    table["_position"] = np.arange(len(table))
    ordered = table.sort_values(KEYS + ["time"], kind="mergesort")
    ordered["target_loc_last"] = ordered.groupby(KEYS, sort=False, observed=True)["_target"].shift(1)
    for window in (12, 24):
        mean = (
            ordered.groupby(KEYS, sort=False, observed=True)["target_loc_last"]
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=KEYS, drop=True)
        )
        std = (
            ordered.groupby(KEYS, sort=False, observed=True)["target_loc_last"]
            .rolling(window, min_periods=2)
            .std(ddof=0)
            .reset_index(level=KEYS, drop=True)
        )
        ordered[f"target_loc_roll{window}_mean"] = mean
        ordered[f"target_loc_roll{window}_std"] = std
    columns = ["target_loc_last"] + [f"target_loc_roll{window}_{stat}" for window in (12, 24) for stat in ("mean", "std")]
    return ordered.sort_values("_position")[columns].reset_index(drop=True)


def _query_recent(fit_raw: pd.DataFrame, target: np.ndarray, query_raw: pd.DataFrame) -> pd.DataFrame:
    fit = fit_raw[["lat", "lon", "time"]].reset_index(drop=True).copy()
    fit["time"] = pd.to_datetime(fit["time"])
    fit["_target"] = np.asarray(target, dtype=np.float32)
    fit = fit.sort_values(KEYS + ["time"], kind="mergesort")
    stats = fit.groupby(KEYS, sort=False, observed=True)["_target"].last().rename("target_loc_last").reset_index()
    for window in (12, 24):
        recent = fit.groupby(KEYS, sort=False, observed=True).tail(window).copy()
        recent["_target_sq"] = recent["_target"].to_numpy(dtype=np.float64) ** 2
        grouped = recent.groupby(KEYS, sort=False, observed=True)
        aggregate = grouped["_target"].agg(["sum", "count"])
        aggregate["sum_sq"] = grouped["_target_sq"].sum()
        aggregate[f"target_loc_roll{window}_mean"] = aggregate["sum"] / aggregate["count"]
        variance = aggregate["sum_sq"] / aggregate["count"] - aggregate[f"target_loc_roll{window}_mean"] ** 2
        aggregate[f"target_loc_roll{window}_std"] = np.sqrt(np.maximum(variance, 0.0))
        aggregate = aggregate[[f"target_loc_roll{window}_mean", f"target_loc_roll{window}_std"]].reset_index()
        stats = stats.merge(aggregate, on=KEYS, how="left", sort=False)
    query = query_raw[["lat", "lon"]].reset_index(drop=True).copy()
    query["_position"] = np.arange(len(query))
    result = query.merge(stats, on=KEYS, how="left", sort=False).sort_values("_position")
    columns = ["target_loc_last"] + [f"target_loc_roll{window}_{stat}" for window in (12, 24) for stat in ("mean", "std")]
    return result[columns].reset_index(drop=True)


def augment_features(x_fit, y_fit, x_query, fit_raw, query_raw):
    x_fit, x_query = target_stats_features.augment_features(
        x_fit, y_fit, x_query, fit_raw, query_raw
    )
    fit_extra = _training_recent(fit_raw, y_fit)
    query_extra = _query_recent(fit_raw, y_fit, query_raw)
    for column in fit_extra:
        x_fit[column] = fit_extra[column].fillna(0).to_numpy(dtype=np.float32)
        x_query[column] = query_extra[column].fillna(0).to_numpy(dtype=np.float32)
    return x_fit, x_query

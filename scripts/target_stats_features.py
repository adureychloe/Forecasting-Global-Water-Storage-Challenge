"""Leakage-safe past target statistics for fixed coordinate panels."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _past_group_stats(raw: pd.DataFrame, target: np.ndarray, keys: list[str], prefix: str) -> pd.DataFrame:
    table = raw[["lat", "lon", "time"]].reset_index(drop=True).copy()
    table["time"] = pd.to_datetime(table["time"])
    table["month"] = table["time"].dt.month.astype(np.int8)
    table["_target"] = np.asarray(target, dtype=np.float32)
    table["_position"] = np.arange(len(table))
    ordered = table.sort_values(keys + ["time"], kind="mergesort")
    grouped = ordered.groupby(keys, sort=False, observed=True)
    count = grouped.cumcount().to_numpy(dtype=np.float64)
    target64 = ordered["_target"].to_numpy(dtype=np.float64)
    cumulative_sum = grouped["_target"].cumsum().to_numpy(dtype=np.float64) - target64
    ordered["_target_sq"] = target64 * target64
    cumulative_sq = grouped["_target_sq"].cumsum().to_numpy(dtype=np.float64) - target64 * target64
    mean = np.divide(cumulative_sum, count, out=np.full(len(ordered), np.nan), where=count > 0)
    variance = np.divide(cumulative_sq, count, out=np.full(len(ordered), np.nan), where=count > 0) - mean * mean
    ordered[f"{prefix}_mean"] = mean
    ordered[f"{prefix}_std"] = np.sqrt(np.maximum(variance, 0.0))
    ordered.loc[count < 2, f"{prefix}_std"] = np.nan
    ordered[f"{prefix}_log_count"] = np.log1p(count)
    return ordered.sort_values("_position")[[f"{prefix}_mean", f"{prefix}_std", f"{prefix}_log_count"]].reset_index(drop=True)


def _query_group_stats(
    fit_raw: pd.DataFrame,
    target: np.ndarray,
    query_raw: pd.DataFrame,
    keys: list[str],
    prefix: str,
) -> pd.DataFrame:
    fit = fit_raw[["lat", "lon", "time"]].reset_index(drop=True).copy()
    query = query_raw[["lat", "lon", "time"]].reset_index(drop=True).copy()
    fit["time"] = pd.to_datetime(fit["time"])
    query["time"] = pd.to_datetime(query["time"])
    fit["month"] = fit["time"].dt.month.astype(np.int8)
    query["month"] = query["time"].dt.month.astype(np.int8)
    fit["_target"] = np.asarray(target, dtype=np.float32)
    stats = fit.groupby(keys, sort=False, observed=True)["_target"].agg(["mean", "std", "count"]).reset_index()
    stats = stats.rename(
        columns={
            "mean": f"{prefix}_mean",
            "std": f"{prefix}_std",
            "count": f"{prefix}_log_count",
        }
    )
    stats[f"{prefix}_log_count"] = np.log1p(stats[f"{prefix}_log_count"].astype(float))
    query["_position"] = np.arange(len(query))
    merged = query.merge(stats, on=keys, how="left", sort=False).sort_values("_position")
    return merged[[f"{prefix}_mean", f"{prefix}_std", f"{prefix}_log_count"]].reset_index(drop=True)


def augment_features(x_fit, y_fit, x_query, fit_raw, query_raw):
    specs = [
        (["lat", "lon"], "target_loc_past"),
        (["lat", "lon", "month"], "target_loc_month_past"),
    ]
    for keys, prefix in specs:
        fit_extra = _past_group_stats(fit_raw, y_fit, keys, prefix)
        query_extra = _query_group_stats(fit_raw, y_fit, query_raw, keys, prefix)
        for column in fit_extra:
            x_fit[column] = fit_extra[column].fillna(0).to_numpy(dtype=np.float32)
            x_query[column] = query_extra[column].fillna(0).to_numpy(dtype=np.float32)
    return x_fit, x_query

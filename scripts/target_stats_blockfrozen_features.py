"""Forecast-block-frozen target history to align training and query availability."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

import target_stats_features


KEYS = ["lat", "lon"]
FEATURES = ["target_block_last"] + [f"target_block_roll{window}_{stat}" for window in (12, 24) for stat in ("mean", "std")]


def _aggregate(source: pd.DataFrame) -> pd.DataFrame:
    ordered = source.sort_values(KEYS + ["_ordinal"], kind="mergesort")
    stats = ordered.groupby(KEYS, sort=False, observed=True)["_target"].last().rename("target_block_last").reset_index()
    for window in (12, 24):
        recent = ordered.groupby(KEYS, sort=False, observed=True).tail(window).copy()
        recent["_target_sq"] = recent["_target"].to_numpy(dtype=np.float64) ** 2
        grouped = recent.groupby(KEYS, sort=False, observed=True)
        aggregate = grouped["_target"].agg(["sum", "count"])
        aggregate["sum_sq"] = grouped["_target_sq"].sum()
        mean_column = f"target_block_roll{window}_mean"
        std_column = f"target_block_roll{window}_std"
        aggregate[mean_column] = aggregate["sum"] / aggregate["count"]
        variance = aggregate["sum_sq"] / aggregate["count"] - aggregate[mean_column] ** 2
        aggregate[std_column] = np.sqrt(np.maximum(variance, 0.0))
        stats = stats.merge(aggregate[[mean_column, std_column]].reset_index(), on=KEYS, how="left", sort=False)
    return stats


def _table(raw: pd.DataFrame, target=None) -> pd.DataFrame:
    result = raw[["lat", "lon", "time"]].reset_index(drop=True).copy()
    result["time"] = pd.to_datetime(result["time"])
    result["_ordinal"] = result["time"].dt.year.to_numpy(dtype=int) * 12 + result["time"].dt.month.to_numpy(dtype=int)
    result["_position"] = np.arange(len(result))
    if target is not None:
        result["_target"] = np.asarray(target, dtype=np.float32)
    return result


def _training_block_features(
    fit_raw: pd.DataFrame,
    target: np.ndarray,
    query_raw: pd.DataFrame,
    block_months: int = 12,
) -> pd.DataFrame:
    fit = _table(fit_raw, target)
    query_start = pd.to_datetime(query_raw["time"]).min()
    anchor = int(query_start.year * 12 + query_start.month)
    distance = anchor - fit["_ordinal"].to_numpy(dtype=int)
    fit["_block_start"] = anchor - block_months * np.ceil(distance / float(block_months)).astype(int)
    values = np.full((len(fit), len(FEATURES)), np.nan, dtype=np.float32)
    for block_start in np.sort(fit["_block_start"].unique()):
        source = fit.loc[fit["_ordinal"] < block_start]
        if source.empty:
            continue
        stats = _aggregate(source)
        positions = np.flatnonzero(fit["_block_start"].to_numpy() == block_start)
        mapped = fit.iloc[positions][KEYS].merge(stats, on=KEYS, how="left", sort=False)
        values[positions] = mapped[FEATURES].to_numpy(dtype=np.float32)
    return pd.DataFrame(values, columns=FEATURES)


def _query_block_features(fit_raw: pd.DataFrame, target: np.ndarray, query_raw: pd.DataFrame) -> pd.DataFrame:
    fit = _table(fit_raw, target)
    query = _table(query_raw)
    query_start = int(query["_ordinal"].min())
    stats = _aggregate(fit.loc[fit["_ordinal"] < query_start])
    return query[KEYS].merge(stats, on=KEYS, how="left", sort=False)[FEATURES]


def augment_features(x_fit, y_fit, x_query, fit_raw, query_raw):
    x_fit, x_query = target_stats_features.augment_features(
        x_fit, y_fit, x_query, fit_raw, query_raw
    )
    fit_extra = _training_block_features(fit_raw, y_fit, query_raw)
    query_extra = _query_block_features(fit_raw, y_fit, query_raw)
    for column in FEATURES:
        x_fit[column] = fit_extra[column].fillna(0).to_numpy(dtype=np.float32)
        x_query[column] = query_extra[column].fillna(0).to_numpy(dtype=np.float32)
    return x_fit, x_query

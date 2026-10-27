"""Temporal CNN with a fixed spherical 5-NN spatial aggregation layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

import gpu_torch_tcn_sequence as runner


BASE_COLS = tuple(runner.SEQ_COLS)
NEIGHBOR_COLS = tuple(f"KNN5_{column}" for column in BASE_COLS)
original_make_arrays = runner.make_arrays
runner.PREFIX = "torch_graph_tcn_sequence"
runner.LOG = runner.RESULTS / f"{runner.PREFIX}.jsonl"
runner.SEQ_COLS = BASE_COLS + NEIGHBOR_COLS


def spherical(coordinates: pd.DataFrame):
    lat = np.deg2rad(coordinates["lat"].to_numpy(dtype=float))
    lon = np.deg2rad(coordinates["lon"].to_numpy(dtype=float))
    return np.column_stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)))


def graph_make_arrays(frame: pd.DataFrame):
    work = frame.reset_index(drop=True).copy()
    work["time"] = pd.to_datetime(work["time"])
    coordinates = work[["lat", "lon"]].drop_duplicates().sort_values(["lat", "lon"]).reset_index(drop=True)
    coordinate_lookup = pd.Series(
        np.arange(len(coordinates)),
        index=pd.MultiIndex.from_frame(coordinates[["lat", "lon"]]),
    )
    coordinate_index = coordinate_lookup.loc[pd.MultiIndex.from_frame(work[["lat", "lon"]])].to_numpy(dtype=int)
    dates = np.sort(work["time"].unique())
    date_lookup = pd.Series(np.arange(len(dates)), index=dates)
    date_index = date_lookup.loc[work["time"]].to_numpy(dtype=int)
    spherical_coordinates = spherical(coordinates)
    neighbors = NearestNeighbors(n_neighbors=6, metric="euclidean", n_jobs=-1).fit(spherical_coordinates)
    neighbor_index = neighbors.kneighbors(spherical_coordinates, return_distance=False)[:, 1:]
    for source, output in zip(BASE_COLS, NEIGHBOR_COLS):
        matrix = np.full((len(dates), len(coordinates)), np.nan, dtype=np.float32)
        matrix[date_index, coordinate_index] = work[source].to_numpy(dtype=np.float32)
        with np.errstate(invalid="ignore"):
            aggregated = np.nanmean(matrix[:, neighbor_index], axis=2)
        work[output] = aggregated[date_index, coordinate_index]
    return original_make_arrays(work)


runner.make_arrays = graph_make_arrays


if __name__ == "__main__":
    raise SystemExit(runner.main())

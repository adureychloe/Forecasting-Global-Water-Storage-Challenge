"""Leakage-safe neighbourhood TWS aggregates from last observed anchors.

Uses lat/lon only to identify cells and neighbours. Neighbouring TWS is taken
at month t or earlier (forward-filled observed TWS). The cell itself is excluded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parent / "results" / "neighborhood_500km.npz"
DIST_CACHE = Path(__file__).resolve().parent / "results" / "neighborhood_500km_dist.npz"
RADIUS_KM = 500.0
RADIUS_250_KM = 250.0
IDW_MIN_KM = 50.0
IDW_POWER = 2.0
EARTH_KM = 6371.0


def _haversine_km(lat0, lon0, lat, lon):
    lat0 = np.deg2rad(lat0)
    lon0 = np.deg2rad(lon0)
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    dlat = lat - lat0
    dlon = lon - lon0
    chord = np.sin(dlat / 2.0) ** 2 + np.cos(lat0) * np.cos(lat) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_KM * np.arcsin(np.sqrt(np.clip(chord, 0.0, 1.0)))


def unique_cells(frame: pd.DataFrame) -> pd.DataFrame:
    cells = frame[["lat", "lon"]].drop_duplicates().sort_values(["lat", "lon"], kind="mergesort")
    return cells.reset_index(drop=True)


def build_neighbor_index(cells: pd.DataFrame, radius_km: float = RADIUS_KM) -> list[np.ndarray]:
    lat = cells["lat"].to_numpy(dtype=np.float64)
    lon = cells["lon"].to_numpy(dtype=np.float64)
    n = len(cells)
    max_dlat = radius_km / 111.32
    neighbors: list[np.ndarray] = []
    for i in range(n):
        cos_lat = max(np.cos(np.deg2rad(lat[i])), 0.05)
        max_dlon = radius_km / (111.32 * cos_lat)
        candidates = np.flatnonzero(
            (np.abs(lat - lat[i]) <= max_dlat)
            & (np.abs(lon - lon[i]) <= max_dlon)
            & (np.arange(n) != i)
        )
        if candidates.size == 0:
            neighbors.append(candidates)
            continue
        dist = _haversine_km(lat[i], lon[i], lat[candidates], lon[candidates])
        neighbors.append(candidates[dist <= radius_km].astype(np.int32, copy=False))
    return neighbors


def load_or_build_neighbors(frame: pd.DataFrame, radius_km: float = RADIUS_KM) -> tuple[pd.DataFrame, list[np.ndarray]]:
    cells = unique_cells(frame)
    cache = CACHE
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        payload = np.load(cache, allow_pickle=True)
        cached_lat = payload["lat"]
        cached_lon = payload["lon"]
        if (
            len(cached_lat) == len(cells)
            and np.allclose(cached_lat, cells["lat"].to_numpy())
            and np.allclose(cached_lon, cells["lon"].to_numpy())
        ):
            return cells, list(payload["neighbors"])
    neighbors = build_neighbor_index(cells, radius_km=radius_km)
    np.savez(
        cache,
        lat=cells["lat"].to_numpy(dtype=np.float64),
        lon=cells["lon"].to_numpy(dtype=np.float64),
        neighbors=np.array(neighbors, dtype=object),
    )
    return cells, neighbors


def _group_stats(values: np.ndarray, neighbor_ids: np.ndarray) -> tuple[float, float, float]:
    if neighbor_ids.size == 0:
        return np.nan, np.nan, 0.0
    sample = values[neighbor_ids]
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return np.nan, np.nan, 0.0
    return float(sample.mean()), float(sample.std(ddof=0)), float(sample.size)


def cell_lookup_ids(raw: pd.DataFrame, cells: pd.DataFrame) -> np.ndarray:
    lookup = pd.Series(
        np.arange(len(cells), dtype=np.int32),
        index=pd.MultiIndex.from_frame(cells[["lat", "lon"]]),
    )
    ids = lookup.reindex(pd.MultiIndex.from_arrays([raw["lat"].to_numpy(), raw["lon"].to_numpy()])).to_numpy()
    if np.isnan(ids).any():
        raise ValueError("row is missing from the neighbourhood cell index")
    return ids.astype(np.int32)


def neighbor_means(anchors: np.ndarray, neighbors: list[np.ndarray]) -> np.ndarray:
    out = np.full(len(anchors), np.nan, dtype=np.float64)
    for i, neighbor_ids in enumerate(neighbors):
        if neighbor_ids.size == 0:
            continue
        sample = anchors[neighbor_ids]
        sample = sample[np.isfinite(sample)]
        if sample.size:
            out[i] = sample.mean()
    return out


def last_observed_anchors(history: pd.DataFrame, cells: pd.DataFrame) -> np.ndarray:
    work = history[["lat", "lon", "time", "TWS_t"]].copy()
    work["time"] = pd.to_datetime(work["time"])
    work["TWS_t"] = pd.to_numeric(work["TWS_t"], errors="coerce")
    work = work.dropna(subset=["TWS_t"]).sort_values("time", kind="mergesort")
    last = work.groupby(["lat", "lon"], sort=False)["TWS_t"].last()
    keys = pd.MultiIndex.from_frame(cells[["lat", "lon"]])
    return last.reindex(keys).to_numpy(dtype=np.float64)


def route_masked_predictions(
    query: pd.DataFrame,
    pred: np.ndarray,
    cells: pd.DataFrame,
    neighbors: list[np.ndarray],
    anchors: np.ndarray,
    *,
    weight: float,
    recursive: bool,
    min_age: float = 0.0,
) -> np.ndarray:
    """Blend model predictions toward neighbourhood persist on masked rows only."""
    routed = np.asarray(pred, dtype=np.float64).copy()
    state = np.asarray(anchors, dtype=np.float64).copy()
    row_ids = cell_lookup_ids(query, cells)
    times = pd.to_datetime(query["time"])
    masked = query["TWS_t_masked"].to_numpy(dtype=bool)
    tws = pd.to_numeric(query["TWS_t"], errors="coerce").to_numpy(dtype=np.float64)
    age = query["TWS_t_age"].to_numpy(dtype=np.float64) if "TWS_t_age" in query.columns else np.where(masked, 1.0, 0.0)
    for _, group in pd.DataFrame({"time": times, "pos": np.arange(len(query))}).groupby("time", sort=True):
        pos = group["pos"].to_numpy()
        ids = row_ids[pos]
        spatial = neighbor_means(state, neighbors)[ids]
        fallback = np.where(np.isfinite(spatial), spatial, state[ids])
        fallback = np.where(np.isfinite(fallback), fallback, routed[pos])
        apply = masked[pos] & (age[pos] >= min_age) & np.isfinite(fallback)
        routed[pos[apply]] = (1.0 - weight) * routed[pos[apply]] + weight * fallback[apply]
        updated = state[ids].copy()
        observed = np.isfinite(tws[pos])
        updated[observed] = tws[pos][observed]
        if recursive:
            updated[~observed] = routed[pos][~observed]
        state[ids] = updated
    return routed.astype(np.float32)


def add_neighborhood_features(raw: pd.DataFrame, features: pd.DataFrame, radius_km: float = RADIUS_KM) -> pd.DataFrame:
    if not {"lat", "lon", "time"}.issubset(raw.columns):
        return features
    out = features.copy()
    cells, neighbors = load_or_build_neighbors(raw, radius_km=radius_km)
    cell_id = pd.MultiIndex.from_frame(cells[["lat", "lon"]])
    cell_lookup = pd.Series(np.arange(len(cells), dtype=np.int32), index=cell_id)
    row_ids = cell_lookup.reindex(pd.MultiIndex.from_arrays([raw["lat"].to_numpy(), raw["lon"].to_numpy()])).to_numpy()
    if np.isnan(row_ids).any():
        raise ValueError("row is missing from the neighbourhood cell index")
    row_ids = row_ids.astype(np.int32)

    if "TWS_t_ffill" in out.columns:
        anchors = out["TWS_t_ffill"].to_numpy(dtype=np.float64)
    else:
        work = raw[["lat", "lon", "time", "TWS_t"]].copy()
        work["time"] = pd.to_datetime(work["time"])
        work = work.sort_values(["lat", "lon", "time"], kind="mergesort")
        work["anchor"] = work.groupby(["lat", "lon"], sort=False)["TWS_t"].ffill()
        anchors = work["anchor"].reindex(raw.index).to_numpy(dtype=np.float64)

    observed = pd.to_numeric(raw["TWS_t"], errors="coerce").to_numpy(dtype=np.float64)
    times = pd.to_datetime(raw["time"]).to_numpy()
    n = len(raw)
    mean = np.full(n, np.nan, dtype=np.float32)
    std = np.full(n, np.nan, dtype=np.float32)
    count = np.zeros(n, dtype=np.float32)
    obs_mean = np.full(n, np.nan, dtype=np.float32)
    for time_value in np.unique(times):
        idx = np.flatnonzero(times == time_value)
        ids = row_ids[idx]
        month_anchor = np.full(len(cells), np.nan, dtype=np.float64)
        month_obs = np.full(len(cells), np.nan, dtype=np.float64)
        month_anchor[ids] = anchors[idx]
        month_obs[ids] = observed[idx]
        for position, cell in zip(idx, ids):
            mean[position], std[position], count[position] = _group_stats(month_anchor, neighbors[cell])
            obs_mean[position], _, _ = _group_stats(month_obs, neighbors[cell])

    out["tws_neighbor_anchor_mean"] = mean
    out["tws_neighbor_anchor_std"] = std
    out["tws_neighbor_anchor_count"] = count
    out["tws_neighbor_obs_mean"] = obs_mean
    out["tws_local_dev"] = (anchors - mean).astype(np.float32)
    return out


def neighbor_distances(cells: pd.DataFrame, neighbors: list[np.ndarray]) -> list[np.ndarray]:
    lat = cells["lat"].to_numpy(dtype=np.float64)
    lon = cells["lon"].to_numpy(dtype=np.float64)
    distances: list[np.ndarray] = []
    for i, neighbor_ids in enumerate(neighbors):
        if neighbor_ids.size == 0:
            distances.append(np.zeros(0, dtype=np.float64))
            continue
        distances.append(_haversine_km(lat[i], lon[i], lat[neighbor_ids], lon[neighbor_ids]))
    return distances


def load_or_build_neighbors_with_dist(
    frame: pd.DataFrame, radius_km: float = RADIUS_KM
) -> tuple[pd.DataFrame, list[np.ndarray], list[np.ndarray]]:
    cells, neighbors = load_or_build_neighbors(frame, radius_km=radius_km)
    cache = DIST_CACHE
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        payload = np.load(cache, allow_pickle=True)
        cached_lat = payload["lat"]
        cached_lon = payload["lon"]
        if (
            len(cached_lat) == len(cells)
            and np.allclose(cached_lat, cells["lat"].to_numpy())
            and np.allclose(cached_lon, cells["lon"].to_numpy())
        ):
            return cells, list(payload["neighbors"]), list(payload["distances"])
    distances = neighbor_distances(cells, neighbors)
    np.savez(
        cache,
        lat=cells["lat"].to_numpy(dtype=np.float64),
        lon=cells["lon"].to_numpy(dtype=np.float64),
        neighbors=np.array(neighbors, dtype=object),
        distances=np.array(distances, dtype=object),
    )
    return cells, neighbors, distances


def _idw_mean(values: np.ndarray, neighbor_ids: np.ndarray, dists: np.ndarray) -> float:
    if neighbor_ids.size == 0:
        return np.nan
    sample = values[neighbor_ids]
    ok = np.isfinite(sample)
    if not ok.any():
        return np.nan
    weights = 1.0 / np.power(np.maximum(dists[ok], IDW_MIN_KM), IDW_POWER)
    return float(np.sum(weights * sample[ok]) / np.sum(weights))


def overlay_pred_neighborhood(
    query_raw: pd.DataFrame,
    x_query: pd.DataFrame,
    pred: np.ndarray,
    radius_km: float = RADIUS_KM,
    update_self: bool = True,
) -> pd.DataFrame:
    """Refresh same-month neighbourhood columns using current predictions on masked rows.

    update_self=False keeps own TWS_t_ffill / local_dev (last-obs) and only
    refreshes neighbour stats. Use that for train-time teacher features so
    in-sample target predictions do not leak into the cell's own columns.
    """
    if not {"lat", "lon"}.issubset(query_raw.columns):
        return x_query
    cells, neighbors, distances = load_or_build_neighbors_with_dist(query_raw, radius_km=radius_km)
    row_ids = cell_lookup_ids(query_raw, cells)
    if "TWS_t_masked" in query_raw.columns:
        masked = query_raw["TWS_t_masked"].to_numpy(dtype=bool)
    else:
        masked = pd.to_numeric(query_raw["TWS_t"], errors="coerce").isna().to_numpy()
    tws = pd.to_numeric(query_raw["TWS_t"], errors="coerce").to_numpy(dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    tws = np.where(masked, pred, tws)
    field = np.full(len(cells), np.nan, dtype=np.float64)
    field[row_ids] = tws
    n = len(query_raw)
    mean = np.full(n, np.nan, dtype=np.float32)
    std = np.full(n, np.nan, dtype=np.float32)
    count = np.zeros(n, dtype=np.float32)
    idw = np.full(n, np.nan, dtype=np.float32)
    mean250 = np.full(n, np.nan, dtype=np.float32)
    count250 = np.zeros(n, dtype=np.float32)
    for i, cell in enumerate(row_ids):
        nb = neighbors[cell]
        dist = distances[cell]
        mean[i], std[i], count[i] = _group_stats(field, nb)
        idw[i] = _idw_mean(field, nb, dist)
        near = nb[dist <= RADIUS_250_KM]
        mean250[i], _, count250[i] = _group_stats(field, near)
    out = x_query.copy()
    out["tws_neighbor_obs_mean"] = mean
    out["tws_neighbor_anchor_mean"] = mean
    out["tws_neighbor_anchor_std"] = std
    out["tws_neighbor_anchor_count"] = count
    out["tws_neighbor_anchor_idw"] = idw
    out["tws_neighbor250_anchor_mean"] = mean250
    out["tws_neighbor250_anchor_count"] = count250
    if update_self:
        out["tws_local_dev"] = (tws - mean).astype(np.float32)
        out["tws_local_dev_idw"] = (tws - idw).astype(np.float32)
        if "TWS_t_ffill" in out.columns:
            ffill = np.array(out["TWS_t_ffill"].to_numpy(dtype=np.float32), copy=True)
            ffill[masked] = pred[masked].astype(np.float32)
            out["TWS_t_ffill"] = ffill
    else:
        if "TWS_t_ffill" in out.columns:
            own = out["TWS_t_ffill"].to_numpy(dtype=np.float32)
            out["tws_local_dev"] = (own - mean).astype(np.float32)
            out["tws_local_dev_idw"] = (own - idw).astype(np.float32)
    return out


def add_rich_neighborhood_features(
    raw: pd.DataFrame,
    features: pd.DataFrame,
    radius_km: float = RADIUS_KM,
    fill_missing_with_anchor: bool = False,
) -> pd.DataFrame:
    """500 km mean/std plus IDW, 250 km ring, and same-month neighbour SPEI/soil."""
    if not {"lat", "lon", "time"}.issubset(raw.columns):
        return features
    out = features.copy()
    cells, neighbors, distances = load_or_build_neighbors_with_dist(raw, radius_km=radius_km)
    row_ids = cell_lookup_ids(raw, cells)

    if "TWS_t_ffill" in out.columns:
        anchors = out["TWS_t_ffill"].to_numpy(dtype=np.float64)
    else:
        work = raw[["lat", "lon", "time", "TWS_t"]].copy()
        work["time"] = pd.to_datetime(work["time"])
        work = work.sort_values(["lat", "lon", "time"], kind="mergesort")
        work["anchor"] = work.groupby(["lat", "lon"], sort=False)["TWS_t"].ffill()
        anchors = work["anchor"].reindex(raw.index).to_numpy(dtype=np.float64)

    observed = pd.to_numeric(raw["TWS_t"], errors="coerce").to_numpy(dtype=np.float64)
    spei = (
        pd.to_numeric(raw["SPEI_03_t"], errors="coerce").to_numpy(dtype=np.float64)
        if "SPEI_03_t" in raw.columns
        else None
    )
    soil = (
        pd.to_numeric(raw["SOIL_MOISTURE_t"], errors="coerce").to_numpy(dtype=np.float64)
        if "SOIL_MOISTURE_t" in raw.columns
        else None
    )
    times = pd.to_datetime(raw["time"]).to_numpy()
    n = len(raw)
    n_cells = len(cells)
    mean = np.full(n, np.nan, dtype=np.float32)
    std = np.full(n, np.nan, dtype=np.float32)
    count = np.zeros(n, dtype=np.float32)
    obs_mean = np.full(n, np.nan, dtype=np.float32)
    idw = np.full(n, np.nan, dtype=np.float32)
    mean250 = np.full(n, np.nan, dtype=np.float32)
    count250 = np.zeros(n, dtype=np.float32)
    spei_mean = np.full(n, np.nan, dtype=np.float32)
    soil_mean = np.full(n, np.nan, dtype=np.float32)

    for time_value in np.unique(times):
        idx = np.flatnonzero(times == time_value)
        ids = row_ids[idx]
        month_anchor = np.full(n_cells, np.nan, dtype=np.float64)
        month_obs = np.full(n_cells, np.nan, dtype=np.float64)
        month_anchor[ids] = anchors[idx]
        month_obs[ids] = observed[idx]
        if fill_missing_with_anchor:
            missing = ~np.isfinite(month_obs[ids])
            month_obs[ids[missing]] = month_anchor[ids[missing]]
        month_spei = None
        month_soil = None
        if spei is not None:
            month_spei = np.full(n_cells, np.nan, dtype=np.float64)
            month_spei[ids] = spei[idx]
        if soil is not None:
            month_soil = np.full(n_cells, np.nan, dtype=np.float64)
            month_soil[ids] = soil[idx]
        for position, cell in zip(idx, ids):
            nb = neighbors[cell]
            dist = distances[cell]
            mean[position], std[position], count[position] = _group_stats(month_anchor, nb)
            obs_mean[position], _, _ = _group_stats(month_obs, nb)
            idw[position] = _idw_mean(month_anchor, nb, dist)
            near = nb[dist <= RADIUS_250_KM]
            mean250[position], _, count250[position] = _group_stats(month_anchor, near)
            if month_spei is not None:
                spei_mean[position], _, _ = _group_stats(month_spei, nb)
            if month_soil is not None:
                soil_mean[position], _, _ = _group_stats(month_soil, nb)

    out["tws_neighbor_anchor_mean"] = mean
    out["tws_neighbor_anchor_std"] = std
    out["tws_neighbor_anchor_count"] = count
    out["tws_neighbor_obs_mean"] = obs_mean
    out["tws_local_dev"] = (anchors - mean).astype(np.float32)
    out["tws_neighbor_anchor_idw"] = idw
    out["tws_local_dev_idw"] = (anchors - idw).astype(np.float32)
    out["tws_neighbor250_anchor_mean"] = mean250
    out["tws_neighbor250_anchor_count"] = count250
    out["tws_neighbor_spei03_mean"] = spei_mean
    out["tws_neighbor_soil_mean"] = soil_mean
    return out

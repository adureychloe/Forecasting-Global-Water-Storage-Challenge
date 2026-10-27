"""Join ERA5 monthly fields onto competition cells. lat/lon are lookup keys only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "results" / "era5_cell_month.parquet"
ERA5_DIR = ROOT / "data" / "era5" / "extracted"

SOIL_THICKNESS_M = (0.07, 0.21, 0.72, 1.89)
ERA5_VALUE_COLS = (
    "tp",
    "e",
    "ro",
    "pev",
    "sf",
    "sd",
    "t2m",
    "swvl1",
    "swvl2",
    "swvl3",
    "swvl4",
    "tp_month",
    "e_month",
    "ro_month",
    "wb",
    "soil_water",
)
LAGS = (1, 2, 3)
ROLLS = (3, 6, 12)


def _month_key(times) -> np.ndarray:
    times = pd.to_datetime(times)
    if isinstance(times, pd.Series):
        return times.dt.year.to_numpy(dtype=np.int32) * 12 + times.dt.month.to_numpy(dtype=np.int32)
    index = pd.DatetimeIndex(times)
    return index.year.to_numpy(dtype=np.int32) * 12 + index.month.to_numpy(dtype=np.int32)


def _unique_cells(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[["lat", "lon"]].drop_duplicates().sort_values(["lat", "lon"], kind="mergesort").reset_index(drop=True)


def build_cell_month_table(cells: pd.DataFrame) -> pd.DataFrame:
    import xarray as xr

    cells = cells[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
    lat_da = xr.DataArray(cells["lat"].to_numpy(dtype=np.float64), dims="cell")
    lon_da = xr.DataArray(cells["lon"].to_numpy(dtype=np.float64), dims="cell")
    parts: list[pd.DataFrame] = []
    for year in range(2002, 2019):
        avgua = xr.open_dataset(ERA5_DIR / f"era5_{year}_avgua.nc")
        avgad = xr.open_dataset(ERA5_DIR / f"era5_{year}_avgad.nc")
        sampled_a = avgua.sel(latitude=lat_da, longitude=lon_da, method="nearest")
        sampled_d = avgad.sel(latitude=lat_da, longitude=lon_da, method="nearest")
        times = pd.to_datetime(sampled_a["valid_time"].to_numpy()).to_period("M").to_timestamp()
        n_time = len(times)
        n_cell = len(cells)
        row = {
            "lat": np.repeat(cells["lat"].to_numpy(dtype=np.float64), n_time),
            "lon": np.repeat(cells["lon"].to_numpy(dtype=np.float64), n_time),
            "time": np.tile(times.to_numpy(), n_cell),
        }
        for name in ("sd", "t2m", "swvl1", "swvl2", "swvl3", "swvl4"):
            row[name] = sampled_a[name].transpose("cell", "valid_time").values.reshape(-1)
        for name in ("tp", "e", "ro", "pev", "sf"):
            row[name] = sampled_d[name].transpose("cell", "valid_time").values.reshape(-1)
        avgua.close()
        avgad.close()
        parts.append(pd.DataFrame(row))
        print(f"sampled {year}", flush=True)
    table = pd.concat(parts, ignore_index=True)
    table["time"] = pd.to_datetime(table["time"])
    days = table["time"].dt.days_in_month.to_numpy(dtype=np.float32)
    table["tp_month"] = table["tp"].to_numpy(dtype=np.float32) * days
    table["e_month"] = table["e"].to_numpy(dtype=np.float32) * days
    table["ro_month"] = table["ro"].to_numpy(dtype=np.float32) * days
    table["wb"] = table["tp_month"] + table["e_month"] - table["ro_month"]
    table["soil_water"] = (
        table["swvl1"] * SOIL_THICKNESS_M[0]
        + table["swvl2"] * SOIL_THICKNESS_M[1]
        + table["swvl3"] * SOIL_THICKNESS_M[2]
        + table["swvl4"] * SOIL_THICKNESS_M[3]
    )
    table = table.sort_values(["lat", "lon", "time"], kind="mergesort").reset_index(drop=True)
    n_cell = int(table.groupby(["lat", "lon"], sort=False).ngroups)
    n_time = int(len(table) // n_cell)
    if n_cell * n_time != len(table):
        raise ValueError("ERA5 panel is not rectangular")
    for column in ERA5_VALUE_COLS:
        arr = table[column].to_numpy(dtype=np.float32).reshape(n_cell, n_time)
        for lag in LAGS:
            lagged = np.full_like(arr, np.nan)
            lagged[:, lag:] = arr[:, :-lag]
            table[f"{column}_lag{lag}"] = lagged.reshape(-1)
        if column in ("tp_month", "wb", "sd", "soil_water", "t2m"):
            for window in ROLLS:
                rolled = np.full_like(arr, np.nan)
                for t in range(1, n_time):
                    start = max(0, t - window)
                    sl = arr[:, start:t]
                    rolled[:, t] = np.nanmean(sl, axis=1)
                table[f"{column}_roll{window}"] = rolled.reshape(-1)
    table["_month_key"] = _month_key(table["time"])
    return table.drop(columns=["time"])


def load_or_build_table(raw: pd.DataFrame) -> pd.DataFrame:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    cells = _unique_cells(raw)
    if CACHE.exists():
        table = pd.read_parquet(CACHE)
        cached = table[["lat", "lon"]].drop_duplicates()
        merged = cells.merge(cached, on=["lat", "lon"], how="left", indicator=True)
        if (merged["_merge"] == "both").all():
            return table
        cells = pd.concat([cached, cells], ignore_index=True).drop_duplicates()
    table = build_cell_month_table(cells)
    table.to_parquet(CACHE, index=False)
    return table


def add_era5_features(raw: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    if not {"lat", "lon", "time"}.issubset(raw.columns):
        return features
    table = load_or_build_table(raw)
    keys = pd.DataFrame(
        {
            "lat": raw["lat"].to_numpy(dtype=np.float64),
            "lon": raw["lon"].to_numpy(dtype=np.float64),
            "_month_key": _month_key(raw["time"]),
        }
    )
    extra = keys.merge(table, on=["lat", "lon", "_month_key"], how="left")
    extra = extra.drop(columns=["lat", "lon", "_month_key"])
    out = features.copy()
    for column in extra.columns:
        out[f"era5_{column}"] = extra[column].to_numpy(dtype=np.float32)
    return out


if __name__ == "__main__":
    train = pd.read_csv(ROOT / "data" / "Train.csv", usecols=["lat", "lon"])
    test = pd.read_csv(ROOT / "data" / "Test.csv", usecols=["lat", "lon"])
    cells = pd.concat([train, test], ignore_index=True)
    table = load_or_build_table(cells)
    print({"rows": len(table), "cols": int(table.shape[1]), "cache": str(CACHE), "size": CACHE.stat().st_size})

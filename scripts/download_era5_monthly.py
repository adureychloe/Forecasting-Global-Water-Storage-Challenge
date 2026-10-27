"""Download ERA5 monthly single-level fields for the Zindi TWS grid.

Dataset (accept the licence in the browser first):
https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means

Uses ERA5 Final (`monthly_averaged_reanalysis`) as the documented retrospective
proxy for ERA5T. No GRACE / TWS-proxy products. Coordinates are only used later
as a lookup to sample the grid; they must not enter the model.

Requires ~/.cdsapirc (CDS API key) and: pip install cdsapi
"""

from __future__ import annotations

from pathlib import Path

import cdsapi

OUT = Path(__file__).resolve().parents[1] / "data" / "era5"
DATASET = "reanalysis-era5-single-levels-monthly-means"
YEARS = [str(year) for year in range(2002, 2019)]
MONTHS = [f"{month:02d}" for month in range(1, 13)]
# Water-balance + snow + temperature. Soil layers overlap official SOIL_MOISTURE_t
# but let us rebuild storage; skip radiation/wind on the first pull.
VARIABLES = [
    "total_precipitation",
    "evaporation",
    "runoff",
    "potential_evaporation",
    "snow_depth",
    "snowfall",
    "2m_temperature",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    for year in YEARS:
        target = OUT / f"era5_monthly_{year}.nc"
        if target.exists() and target.stat().st_size > 1_000_000:
            print(f"skip {target.name}")
            continue
        request = {
            "product_type": ["monthly_averaged_reanalysis"],
            "variable": VARIABLES,
            "year": [year],
            "month": MONTHS,
            "time": ["00:00"],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": [84, -180, -56, 180],
        }
        print(f"request {year} -> {target}")
        client.retrieve(DATASET, request, str(target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

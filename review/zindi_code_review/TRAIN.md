# Training commands

Run these from the repository root, not from this folder. Official `Train.csv`, `Test.csv`, and `SampleSubmission.csv` must be in `data/`. ERA5 monthly files go under `data/era5/` (see `download_era5_monthly.py`); the cell-month cache is `results/era5_cell_month.parquet`.

Python on the training box: `/venv/main/bin/python` (Vast.ai RTX 4090). After `stage=complete` appears in the JSON log, stop the process (the scripts sleep for a supervisor).

```text
python download_era5_monthly.py

python gpu_catboost_idw_era5_aligned.py
python gpu_lgbm_idw_era5_aligned.py
python gpu_lgbm_idw_era5_aligned_extra.py
python gpu_xgb_idw_era5_aligned.py
```

Needed outputs in `submission/`:

```text
submission_catboost_horizon_idw_era5_aligned_seed77_aligned_raw.csv
submission_catboost_horizon_idw_era5_aligned_seed2027_aligned_raw.csv
submission_lgbm_horizon_idw_era5_aligned_seed2027_aligned_raw.csv
submission_lgbm_horizon_idw_era5_aligned_seed77_aligned_raw.csv
submission_lgbm_horizon_idw_era5_aligned_extra_seed123_aligned_raw.csv
submission_xgb_horizon_idw_era5_aligned_seed77_aligned_raw.csv
```

The 0.7139 base file is the earlier public leader:

```text
submission_onlineleader_idw_era5_lgbmpattern_seed2027_raw_w025.csv
```

Then, from this package:

```text
python reproduce.py --source-dir <submission_or_sources>
```

Coordinate rule: `coordinate_compliance.py` drops raw lat/lon encodings. Neighbourhood and ERA5 joins may use lat/lon as keys only.

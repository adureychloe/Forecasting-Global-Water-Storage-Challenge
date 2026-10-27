# Documentation

**Challenge:** A Step Ahead of Drought: Forecasting Global Water Storage Challenge  
**Zindi username:** jinneng  
**Selected submission ID:** KJhqz4jM  
**Selected file:** `submission_onlineleader_idw_era5_lgbmaligned_extra_seed123_raw_w025.csv`  
**Public LB RMSE:** 0.6955  
**Private LB:** this file was the best of our submissions (email: top 30). The email did not print the private number.

This document follows Zindi’s documentation guideline and code-review article.

## Overview and objectives

The task is to forecast next-month terrestrial water storage (TWS) on a global grid. Official Test spans 40 calendar months with only 18 observed months, and about two thirds of Test rows have `TWS_t` masked. The label is next-month TWS.

The solution is a leakage-aware tabular ensemble:

1. Build past-only features from official Train/Test plus ERA5 monthly fields (coordinates used as lookup keys only).
2. Train five age-bucket heads per model family (CatBoost Ordered, LightGBM, XGBoost) on Train months from 2010-01 onward.
3. Run Test inference with a time-aligned walk: the forecast for month *t* is written into month *t+1* only, matching how training lags are built.
4. Mix the aligned families into the earlier public-leader ensemble with small convex weights, then mix LightGBM aligned seed 123 at 2.5%.

The selected file is that last 2.5% mix. On the public board it tied 0.6955 with the XGB-heavy parent; on private it was the best of our files.

Objectives:

- Do not use raw latitude/longitude as model features (lookup only).
- Do not use GRACE or other TWS-proxy products.
- Keep Test inference causal: no future TWS, no future ERA5.
- Reproduce the selected CSV exactly from saved component predictions; retrain from official data with fixed seeds.

## Architecture

```text
Official Train.csv / Test.csv
        |                          ERA5 monthly (CDS, lat/lon lookup)
        v                                      |
  past-only features <-------------------------+
  (lags, IDW 250 km neighbourhood TWS,         |
   block-frozen target stats, ERA5 lags/rolls, |
   ERA5 calendar anomalies; no raw lat/lon)    |
        |
        v
  2010+ full-fit, 5 age heads
  (observed / gap1 / gap2 / gap3_6 / gap7plus)
  CatBoost Ordered | LightGBM GPU | XGBoost CUDA
        |
        v
  walk_aligned Test (inner iterations = 1)
  forecast(t+1) written into month t+1 only
        |
        v
  nested convex blends (documented weights)
        |
        v
  selected submission KJhqz4jM
```

Component roles in the selected mix (expanded from the nested weights below):

| Component | Role |
|---|---|
| 0.7139 public-leader base (`lgbmpattern` seed2027 2.5% stack) | Earlier IDW + ERA5 + recursive + 2010+ pattern ensemble |
| CatBoost aligned seeds 77 and 2027 | First aligned-walk family |
| LightGBM aligned seeds 2027 and 77 | Second aligned-walk family |
| XGBoost aligned seed 77 | Dominant aligned family after the 10/20/30% public ramp |
| LightGBM aligned extra seed 123 | Final 2.5% mix (selected file) |

## ETL process

### Extract

| Source | Location | Notes |
|---|---|---|
| Official Train / Test / SampleSubmission | challenge data page | `data/Train.csv`, `data/Test.csv`, `data/SampleSubmission.csv` |
| ERA5 monthly means | Copernicus CDS `reanalysis-era5-single-levels-monthly-means` | Variables: total precipitation, evaporation, runoff, potential evaporation, snow depth, snowfall, 2 m temperature, volumetric soil water layers 1–4. Years 2002–2018. Script: `download_era5_monthly.py`. Licence accepted in the CDS UI. |

No Excel preprocessing. No GRACE mascon or other TWS reconstructions.

### Transform

All transforms run in Python:

- Parse `time` to timestamps.
- Official mask: Test rows with `TWS_t_masked` have `TWS_t` set to missing before inference.
- Age of last observed TWS (`TWS_t_age`) routes each row to one of five heads.
- Neighbourhood TWS within 250 km, inverse-distance weighted, using only observations at or before the query month.
- Block-frozen 12-month target statistics computed on the fit window only.
- ERA5 fields joined by nearest cell and calendar month; lags 1/2/3 and rolls 3/6/12; calendar-month anomalies from the fit window only.
- Banned coordinate columns are dropped: `lat`, `lon`, `lat_sin`, `lat_cos`, `lon_sin`, `lon_cos`, `loc_id`. Coordinates remain as join keys.

### Load

- ERA5 cell-month table is cached as `results/era5_cell_month.parquet`.
- Neighbour lookup arrays are cached under `results/`.
- Each model family writes a 280,961-row CSV in SampleSubmission ID order.
- The selected file is written by `reproduce.py`.

## Data modeling

### Assumptions

- Masked Test TWS is missing at inference time, not zero.
- Consecutive Test months can reuse the previous aligned forecast as the next month’s TWS lag; non-consecutive months must not.
- Age of the last observation is a better split than a single global model.
- Recency (Train ≥ 2010) matches the Test era better than using the full 2002–2015 history for the final heads.

### Features

Shared by CatBoost, LightGBM, and XGBoost aligned models:

- Base temporal and TWS lag/roll features from `enhanced_features.py`.
- IDW neighbourhood TWS from `neighborhood_tws_features.py`.
- ERA5 cell features from `era5_cell_features.py`.
- Block-frozen target statistics from `target_stats_blockfrozen_features.py`.
- ERA5 calendar anomalies from `era5_anomaly_features.py`.

### Models

Five heads per seed, 500 iterations, RMSE, GPU:

**CatBoost Ordered** (`gpu_catboost_idw_era5_aligned.py`)

- depth 8, learning_rate 0.04, l2_leaf_reg 5, random_strength 1, SymmetricTree, Ordered, border_count 128
- seeds 2027 and 77

**LightGBM** (`gpu_lgbm_idw_era5_aligned.py`, extra seeds in `gpu_lgbm_idw_era5_aligned_extra.py`)

- n_estimators 500, learning_rate 0.04, num_leaves 63, max_depth 8, min_child_samples 40, subsample 0.90, colsample_bytree 0.90, reg_lambda 10
- seeds 2027, 77, and extra seed 123

**XGBoost CUDA** (`gpu_xgb_idw_era5_aligned.py`)

- n_estimators 500, max_depth 8, learning_rate 0.04, min_child_weight 8, subsample 0.90, colsample_bytree 0.90, reg_lambda 12, reg_alpha 0.10, hist, max_bin 256
- seed 77

Heads with fewer than 1,000 fit rows, or with no Test rows, are skipped. Official Test has no age ≥ 7 rows, so `gap7plus` is unused at Test time.

### Validation used while building

- Ordinary two-fold chronological CV on older recipes.
- Sparse-calendar protocol matching Test: 68-month recency, anchors 2007-09 / 2008-06 / 2009-01, seeds 2027/77 (`#262`). Aligned walk pooled RMSE 0.522228 versus static 0.526434.
- Online public mixes were taken in 2.5% (or 10/20/30% for XGB aligned seed 77) steps. Families that failed online were not increased.

## Inference

Test months are walked in calendar order (`aligned_feedback.walk_aligned`):

1. History contains only rows strictly before the current month.
2. If a forecast from the previous calendar month exists, it fills masked `TWS_t` on the current month.
3. Features are rebuilt on history + current month.
4. Age heads predict next-month TWS.
5. That forecast is stored for the following calendar month only. Intermediate missing months are not synthesized.

`inner_iters = 1` (no extra spatial overlay at inference).

The selected CSV is not a single model. It is this nested mix (each line is `(1-w)*current_leader + w*source`):

| Step | Public LB after mix | Weight | Source |
|---|---|---|---|
| Base | 0.7139 | — | `lgbmpattern` seed2027 2.5% stack |
| + CatBoost aligned 77 | 0.7121 | 2.5% | `..._aligned_seed77_aligned_raw.csv` |
| + CatBoost aligned 2027 | 0.7105 | 2.5% | seed 2027 aligned |
| + LightGBM aligned 2027 | 0.7090 | 2.5% | LGBM aligned seed 2027 |
| + LightGBM aligned 77 | 0.7076 | 2.5% | LGBM aligned seed 77 |
| + XGB aligned 77 | 0.7027 | 10% | XGB aligned seed 77 |
| + XGB aligned 77 | 0.6957 | 30% into the 0.7027 leader | same source |
| remade XGB aligned 77 | 0.6955 | 20% into the 0.6957 leader | same source |
| + LightGBM aligned extra 123 | **0.6955 public / best private** | 2.5% | LGBM aligned seed 123 |

`reproduce.py` applies the last step as `0.975 * (XGB aligned 77 remade w200) + 0.025 * (LightGBM aligned seed 123)` and checks that this agrees with the full nested history. SHA256 of the selected file: `c2f4c5018bdc0a578c055fa0d2179d3e9785df35f989c13335f01bab9e483b84`.

## Run time

Hardware: Vast.ai NVIDIA RTX 4090, 32+ GB RAM, Python 3.12 (`/venv/main/bin/python`). Local blend machine: Windows, CPU only.

| Step | Hardware | Time |
|---|---|---|
| ERA5 download + cell-month cache | CPU + CDS | several hours, once |
| Neighbour cache | CPU | minutes, once |
| One aligned seed (5 heads + 18-month walk) | RTX 4090 | ~2.5 hours |
| CatBoost aligned 77 and 2027 | RTX 4090 | ~5 hours |
| LightGBM aligned 77, 2027, 123 | RTX 4090 | ~7.5 hours |
| XGBoost aligned 77 | RTX 4090 | ~2.5 hours |
| `reproduce.py` nested blend | CPU | < 1 minute |

Original training scripts log JSON and then sleep so a supervisor can see `stage=complete`. Stop the process after that line.

## Performance metrics

| Metric | Value |
|---|---|
| Public LB RMSE (selected file) | 0.6955 |
| Public LB RMSE of parent (XGB aligned 77 remade 20%) | 0.6955 (tie) |
| Sparse-calendar aligned walk vs static (#262) | 0.522228 vs 0.526434 |
| Selected-file vs parent unlabeled RMS | 0.0048 (2.5% mix) |

Other public probes that did not beat 0.6955: XGB aligned seed2027 2.5% (0.6956), neighbourhood persist (0.6965). Those files are not in the selected mix.

## Error handling and logging

- Training scripts write one JSON object per line (`stage`, seed, bucket, month).
- Non-finite predictions abort before a CSV is written.
- Submission CSVs are checked for 280,961 rows, SampleSubmission ID order, and finite values.
- `reproduce.py` compares against the selected file and exits non-zero on mismatch.

## Maintenance and monitoring

To refresh on new months: rebuild ERA5 through the new month, rebuild features with the same past-only rule, refit 2010+ heads, rerun `walk_aligned`, then apply the same blend weights or re-tune only the last mix weight. Do not feed future TWS or future ERA5. Keep coordinates as lookup keys only.

## Reproducibility notes

- Random seeds are 2027, 77, and 123 as listed above.
- CatBoost `allow_writing_files=False`; LightGBM `random_state`; XGBoost `random_state`.
- GPU floating-point can differ slightly if heads are retrained on another driver/library build. The selected CSV is reproduced exactly by `reproduce.py` from the component CSVs in `sources/`.
- All processing is in Python. Requirements are in `requirements.txt`. Use GPU builds of CatBoost, LightGBM, and XGBoost for training.

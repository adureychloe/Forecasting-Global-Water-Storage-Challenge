# Code review package

Zindi challenge: **A Step Ahead of Drought: Forecasting Global Water Storage Challenge**

Selected submission ID: **KJhqz4jM**  
File: `submission_onlineleader_idw_era5_lgbmaligned_extra_seed123_raw_w025.csv`  
Public LB RMSE: **0.6955** (this file tied the public leader and was the best private file)

Deadline for this package: **17:00 GMT, 25 September 2026**.

## What to run first (exact CSV)

This reproduces the selected file from the nested blend of already-generated model CSVs. It is the bit-stable path.

```text
python reproduce.py
```

It writes `output/submission_onlineleader_idw_era5_lgbmaligned_extra_seed123_raw_w025.csv` and checks SHA256 `c2f4c5018bdc0a578c055fa0d2179d3e9785df35f989c13335f01bab9e483b84`.

Place official `data/SampleSubmission.csv` at the repo `data/` path, or pass `--sample`. Source CSVs are read from `sources/` in this folder, or from the repo `submission/` directory.

## Full training path (regenerate the model CSVs)

See `DOCUMENTATION.md` and `TRAIN.md`. Training used one NVIDIA RTX 4090. Each aligned seed takes about 2.5 hours. The original training scripts sleep after `stage=complete` so a process supervisor can detect success; stop them after that log line.

## Layout

| Path | Role |
|---|---|
| `DOCUMENTATION.md` | Full write-up required by Zindi |
| `TRAIN.md` | Commands to rebuild model CSVs |
| `reproduce.py` | Nested blend → selected submission |
| `requirements.txt` | Python libraries |
| `sources/` | Component CSVs used by `reproduce.py` (aligned families + 0.7139 base + 0.6955 parent) |
| `expected/` | Copy of the selected submission |
| `code/` | Training/inference modules (run from the original repo root; see TRAIN.md) |

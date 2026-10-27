# Code review package

Zindi challenge: **A Step Ahead of Drought: Forecasting Global Water Storage Challenge**

Selected submission ID: **KJhqz4jM**  
File: `submission_onlineleader_idw_era5_lgbmaligned_extra_seed123_raw_w025.csv`  
Public LB RMSE: **0.6955** (this file tied the public leader and was the best private file)

Historical package deadline: **17:00 GMT, 25 September 2026**. This is an archived competition package, not a current deadline.

## What to run first (exact CSV)

This reproduces the selected file from the nested blend of already-generated model CSVs. It is the bit-stable path.

```text
python reproduce.py --sample /path/to/SampleSubmission.csv
```

It writes `output/submission_onlineleader_idw_era5_lgbmaligned_extra_seed123_raw_w025.csv` and compares predictions to `expected/` with RMSE tolerance `1e-12`. The reference SHA256 is `c2f4c5018bdc0a578c055fa0d2179d3e9785df35f989c13335f01bab9e483b84`. NumPy 1.26.4 / pandas 2.2.2 reproduce the reference bytes; other versions can differ in floating-point serialization.

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
| `scripts/` | Training/inference modules in the exported ZIP; run from the package root. In the repository, use the top-level `scripts/` directory |

## Build the review ZIP

From the repository root in PowerShell:

```powershell
powershell -File review/make_review_zip.ps1
```

This reads the archived `sources/` and `expected/` directly, so the ignored local `submission/` directory is not required. It exports the 17 training/inference modules needed by the aligned path into `scripts/` and writes `review/zindi_code_review.zip`. Generated ZIPs, staging directories and `output/` are ignored by Git.

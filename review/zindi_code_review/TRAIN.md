# Training and data preparation

Run the commands below from the **repository root**. In the exported review ZIP, run from the package root containing `scripts/`. Python 3.12 is the historical training version. The original machine used Linux and an NVIDIA RTX 4090, with Python at `/venv/main/bin/python`.

## 1. Dependencies and official data

In the repository:

```bash
python -m pip install -r requirements.txt
```

The exported package has a self-contained `requirements.txt`. NumPy 1.26.4 and pandas 2.2.2 match the archived CSV serialization environment. Install builds that support the model backends in the scripts: CatBoost GPU, LightGBM `device=gpu`, and XGBoost CUDA. A standard wheel does not guarantee the required GPU support on every platform.

Obtain the official challenge files separately and place them at:

```text
data/Train.csv
data/Test.csv
data/SampleSubmission.csv
```

Create the output directories if they do not exist:

```bash
python -c "from pathlib import Path; [Path(p).mkdir(parents=True, exist_ok=True) for p in ('results', 'submission', 'logs')]"
```

## 2. ERA5 download and extraction

Accept the licence for [ERA5 monthly averaged single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means), and configure CDS API credentials in your local `~/.cdsapirc` following the CDS instructions. Credentials are not included in this repository.

```bash
python scripts/download_era5_monthly.py
```

The historical downloader requests 2002–2018 and writes `data/era5/era5_monthly_<year>.nc`. Despite the `.nc` suffix, the archived downloads are ZIP containers with these two members:

```text
data_stream-moda_stepType-avgua.nc
data_stream-moda_stepType-avgad.nc
```

`era5_cell_features.py` expects them unpacked and renamed as:

```text
data/era5/extracted/era5_2002_avgua.nc
data/era5/extracted/era5_2002_avgad.nc
...
data/era5/extracted/era5_2018_avgua.nc
data/era5/extracted/era5_2018_avgad.nc
```

For downloads with that ZIP layout, run this Python snippet from the repository/package root (for example, save it to a temporary Python file and execute it):

```python
from pathlib import Path
import shutil
import zipfile

base = Path("data/era5")
out = base / "extracted"
out.mkdir(parents=True, exist_ok=True)
for year in range(2002, 2019):
    source = base / f"era5_monthly_{year}.nc"
    if not zipfile.is_zipfile(source):
        raise ValueError(f"Inspect the CDS response format before unpacking: {source}")
    with zipfile.ZipFile(source) as archive:
        for kind in ("avgua", "avgad"):
            member = f"data_stream-moda_stepType-{kind}.nc"
            target = out / f"era5_{year}_{kind}.nc"
            with archive.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
```

CDS response formats can change; inspect the actual container and variable names if these members are absent. The feature loader expects the historical `valid_time` dimension, `latitude` / `longitude` coordinates, instantaneous fields `sd`, `t2m`, `swvl1`–`swvl4`, and accumulated fields `tp`, `e`, `ro`, `pev`, `sf`.

The first training run builds `results/era5_cell_month.parquet` and neighbourhood caches under `results/`. These are local generated files.

## 3. Train aligned model families

Run **one command at a time**. Each historical aligned script enters an idle loop after logging `"stage": "complete"`. Stop that completed command with `Ctrl+C` or Supervisor before starting the next command.

```bash
python scripts/gpu_catboost_idw_era5_aligned.py
```

```bash
python scripts/gpu_lgbm_idw_era5_aligned.py
```

```bash
python scripts/gpu_lgbm_idw_era5_aligned_extra.py
```

```bash
python scripts/gpu_xgb_idw_era5_aligned.py
```

Selected aligned outputs in `submission/`:

```text
submission_catboost_horizon_idw_era5_aligned_seed77_aligned_raw.csv
submission_catboost_horizon_idw_era5_aligned_seed2027_aligned_raw.csv
submission_lgbm_horizon_idw_era5_aligned_seed2027_aligned_raw.csv
submission_lgbm_horizon_idw_era5_aligned_seed77_aligned_raw.csv
submission_lgbm_horizon_idw_era5_aligned_extra_seed123_aligned_raw.csv
submission_xgb_horizon_idw_era5_aligned_seed77_aligned_raw.csv
```

The extra LightGBM script also trains seed 9001; the XGBoost script also trains seed 2027. Those additional outputs are not part of the selected final blend. Historical timing was about 2.5 hours per aligned seed; the old end-to-end estimate was 18–20 hours and is not a runtime guarantee.

## 4. Rebuild the selected blend

For the archived, byte-stable result, use the tracked `sources/`:

```bash
python review/zindi_code_review/reproduce.py
```

In the exported ZIP, use an explicit official sample path:

```bash
python reproduce.py --sample /path/to/SampleSubmission.csv
```

To read a separate directory containing the complete compatible input set:

```bash
python review/zindi_code_review/reproduce.py --source-dir /path/to/components
```

Training the four families above regenerates the aligned model components. The final selected submission also depends on the earlier `0.7139` ensemble base and the `0.6955` parent. Both are archived in `sources/`; the earlier base is not rebuilt by these four training commands. With newly trained components, rebuild the full nested history in `DOCUMENTATION.md`, including the parent file, before using the reproduction checker. The checker intentionally rejects a parent that disagrees with the supplied components. GPU retraining is not guaranteed to produce identical bytes or the same score.

Coordinate rule for the final aligned path: `coordinate_compliance.py` drops raw lat/lon encodings; neighbourhood and ERA5 joins use coordinates as lookup keys only. Earlier exploratory baselines are not the final competition recipe.

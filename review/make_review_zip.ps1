$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$pkg = Join-Path $PSScriptRoot "zindi_code_review"
$sub = Join-Path $root "submission"
$staging = Join-Path $PSScriptRoot "_zip_staging"
$zip = Join-Path $PSScriptRoot "zindi_code_review.zip"

$sources = @(
    "submission_onlineleader_idw_era5_lgbmpattern_seed2027_raw_w025.csv",
    "submission_catboost_horizon_idw_era5_aligned_seed77_aligned_raw.csv",
    "submission_catboost_horizon_idw_era5_aligned_seed2027_aligned_raw.csv",
    "submission_lgbm_horizon_idw_era5_aligned_seed2027_aligned_raw.csv",
    "submission_lgbm_horizon_idw_era5_aligned_seed77_aligned_raw.csv",
    "submission_xgb_horizon_idw_era5_aligned_seed77_aligned_raw.csv",
    "submission_lgbm_horizon_idw_era5_aligned_extra_seed123_aligned_raw.csv",
    "submission_onlineleader_idw_era5_xgbaligned_seed77_raw_w200.csv"
)
$code = @(
    "gpu_catboost_idw_era5_aligned.py",
    "gpu_lgbm_idw_era5_aligned.py",
    "gpu_lgbm_idw_era5_aligned_extra.py",
    "gpu_xgb_idw_era5_aligned.py",
    "gpu_horizon_runner.py",
    "gpu_nightly_runner.py",
    "gpu_dualhead_runner.py",
    "aligned_feedback.py",
    "enhanced_features.py",
    "era5_cell_features.py",
    "era5_anomaly_features.py",
    "neighborhood_tws_features.py",
    "target_stats_blockfrozen_features.py",
    "target_stats_features.py",
    "coordinate_compliance.py",
    "download_era5_monthly.py",
    "baseline.py"
)

if (Test-Path $staging) { Remove-Item $staging -Recurse -Force }
New-Item -ItemType Directory -Path (Join-Path $staging "zindi_code_review\sources") | Out-Null
New-Item -ItemType Directory -Path (Join-Path $staging "zindi_code_review\expected") | Out-Null
New-Item -ItemType Directory -Path (Join-Path $staging "zindi_code_review\code") | Out-Null

Copy-Item (Join-Path $pkg "README.md") (Join-Path $staging "zindi_code_review\")
Copy-Item (Join-Path $pkg "DOCUMENTATION.md") (Join-Path $staging "zindi_code_review\")
Copy-Item (Join-Path $pkg "TRAIN.md") (Join-Path $staging "zindi_code_review\")
Copy-Item (Join-Path $pkg "requirements.txt") (Join-Path $staging "zindi_code_review\")
Copy-Item (Join-Path $pkg "reproduce.py") (Join-Path $staging "zindi_code_review\")

foreach ($name in $sources) {
    $src = Join-Path $sub $name
    if (-not (Test-Path $src)) { throw "missing $src" }
    Copy-Item $src (Join-Path $staging "zindi_code_review\sources\$name")
}
$win = "submission_onlineleader_idw_era5_lgbmaligned_extra_seed123_raw_w025.csv"
Copy-Item (Join-Path $sub $win) (Join-Path $staging "zindi_code_review\expected\$win")
Copy-Item (Join-Path $sub $win) (Join-Path $staging "zindi_code_review\sources\$win")

foreach ($name in $code) {
    $src = Join-Path $root $name
    if (-not (Test-Path $src)) { throw "missing $src" }
    Copy-Item $src (Join-Path $staging "zindi_code_review\code\$name")
}

if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path (Join-Path $staging "zindi_code_review") -DestinationPath $zip -Force
Remove-Item $staging -Recurse -Force
Get-Item $zip | Select-Object FullName, Length

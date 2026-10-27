$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$pkg = Join-Path $PSScriptRoot "zindi_code_review"
$staging = Join-Path $PSScriptRoot ("_zip_staging/" + [guid]::NewGuid().ToString("N"))
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

# Stage only files owned by this invocation; bundled predictions work on a fresh clone.
$staging = [IO.Path]::GetFullPath($staging)
$allowedStaging = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "_zip_staging")) + [IO.Path]::DirectorySeparatorChar
if (-not $staging.StartsWith($allowedStaging, [StringComparison]::OrdinalIgnoreCase)) {
    throw "staging path must stay inside review/_zip_staging"
}
$packageRoot = Join-Path $staging "zindi_code_review"
try {
    foreach ($folder in @("sources", "expected", "scripts")) {
        New-Item -ItemType Directory -Path (Join-Path $packageRoot $folder) -Force | Out-Null
    }
    foreach ($name in @("README.md", "DOCUMENTATION.md", "TRAIN.md", "requirements.txt", "reproduce.py")) {
        Copy-Item -LiteralPath (Join-Path $pkg $name) -Destination $packageRoot
    }
    foreach ($name in $sources) {
        $src = Join-Path (Join-Path $pkg "sources") $name
        if (-not (Test-Path -LiteralPath $src)) { throw "missing $src" }
        Copy-Item -LiteralPath $src -Destination (Join-Path $packageRoot "sources/$name")
    }
    $win = "submission_onlineleader_idw_era5_lgbmaligned_extra_seed123_raw_w025.csv"
    Copy-Item -LiteralPath (Join-Path $pkg "expected/$win") -Destination (Join-Path $packageRoot "expected/$win")
    foreach ($name in $code) {
        $src = Join-Path (Join-Path $root "scripts") $name
        if (-not (Test-Path -LiteralPath $src)) { throw "missing $src" }
        Copy-Item -LiteralPath $src -Destination (Join-Path $packageRoot "scripts/$name")
    }
    Compress-Archive -LiteralPath $packageRoot -DestinationPath $zip -Force
} finally {
    if (Test-Path -LiteralPath $staging) {
        Remove-Item -LiteralPath $staging -Recurse -Force
    }
}
Get-Item -LiteralPath $zip | Select-Object FullName, Length

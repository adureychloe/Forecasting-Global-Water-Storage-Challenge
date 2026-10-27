"""Check entry points and cache paths after moving scripts out of the root."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_baseline_entry_point_works_outside_repository(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "baseline.py"), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--data-dir" in result.stdout


def test_feature_caches_stay_in_repository_root():
    import era5_cell_features as era5
    import neighborhood_tws_features as neighbors
    import gpu_sparse_replay as replay

    assert era5.ROOT == ROOT
    assert era5.CACHE == ROOT / "results" / "era5_cell_month.parquet"
    assert era5.ERA5_DIR == ROOT / "data" / "era5" / "extracted"
    assert neighbors.CACHE.parent == ROOT / "results"
    assert neighbors.DIST_CACHE.parent == ROOT / "results"
    assert replay.ROOT == ROOT


def test_training_outputs_stay_in_repository_root():
    import gpu_horizon_runner as runner

    assert runner.ROOT == ROOT
    assert runner.RESULTS == ROOT / "results"
    assert runner.SUBMISSIONS == ROOT / "submission"
    assert runner.core.DATA == ROOT / "data"

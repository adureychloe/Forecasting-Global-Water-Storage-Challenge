"""After #267, train remaining aligned-search jobs one by one. Do not submit."""

import json
import subprocess
import time
from pathlib import Path

SEQUENCE = (
    ("water_gpu_catboost_idw_era5_aligned_delta", Path("/workspace/water/logs/gpu-catboost-idw-era5-aligned-delta.log")),
    ("water_gpu_catboost_idw_era5_aligned_spatialiter", Path("/workspace/water/logs/gpu-catboost-idw-era5-aligned-spatialiter.log")),
    ("water_gpu_catboost_idw_era5_aligned_climcluster", Path("/workspace/water/logs/gpu-catboost-idw-era5-aligned-climcluster.log")),
    ("water_gpu_aligned_inner3_cv", Path("/workspace/water/logs/gpu-aligned-inner3-cv.log")),
    ("water_gpu_aligned_delta_cv", Path("/workspace/water/logs/gpu-aligned-delta-cv.log")),
    ("water_gpu_catboost_idw_era5_aligned_extra", Path("/workspace/water/logs/gpu-catboost-idw-era5-aligned-extra.log")),
    ("water_gpu_lgbm_idw_era5_aligned_extra", Path("/workspace/water/logs/gpu-lgbm-idw-era5-aligned-extra.log")),
    ("water_gpu_xgb_idw_era5_aligned_extra", Path("/workspace/water/logs/gpu-xgb-idw-era5-aligned-extra.log")),
)
XGB = ("water_gpu_xgb_idw_era5_aligned", Path("/workspace/water/logs/gpu-xgb-idw-era5-aligned.log"))


def completed(log: Path) -> bool:
    try:
        lines = log.read_text(encoding="utf-8", errors="ignore").splitlines()
        return bool(lines) and json.loads(lines[-1]).get("stage") == "complete"
    except Exception:
        return False


def wait_complete(name: str, log: Path) -> None:
    while not completed(log):
        time.sleep(30)
    subprocess.run(["supervisorctl", "stop", name], check=False)


def main() -> None:
    if not completed(XGB[1]):
        while not completed(XGB[1]):
            time.sleep(30)
        subprocess.run(["supervisorctl", "stop", XGB[0]], check=False)
    for job, log in SEQUENCE:
        if completed(log):
            subprocess.run(["supervisorctl", "stop", job], check=False)
            continue
        subprocess.run(["supervisorctl", "start", job], check=False)
        wait_complete(job, log)


if __name__ == "__main__":
    main()

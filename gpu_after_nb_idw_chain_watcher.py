"""Run the post-#214 GPU chain: IDW neighbourhood, persist residual, XGB, extra seeds."""

import json
import subprocess
import time
from pathlib import Path

SEQUENCE = (
    (
        "water_gpu_catboost_blockfrozen_nb_idw",
        Path("/workspace/water/logs/gpu-catboost-blockfrozen-nb-idw.log"),
    ),
    (
        "water_gpu_catboost_blockfrozen_nb_residual",
        Path("/workspace/water/logs/gpu-catboost-blockfrozen-nb-residual.log"),
    ),
    (
        "water_gpu_xgb_blockfrozen_neighborhood",
        Path("/workspace/water/logs/gpu-xgb-blockfrozen-neighborhood.log"),
    ),
    (
        "water_gpu_catboost_blockfrozen_neighborhood_extra",
        Path("/workspace/water/logs/gpu-catboost-blockfrozen-neighborhood-extra.log"),
    ),
)


def completed(log: Path) -> bool:
    try:
        lines = log.read_text(encoding="utf-8", errors="ignore").splitlines()
        return bool(lines) and json.loads(lines[-1]).get("stage") == "complete"
    except Exception:
        return False


def main() -> None:
    for job, log in SEQUENCE:
        if not completed(log):
            subprocess.run(["supervisorctl", "start", job], check=False)
            while not completed(log):
                time.sleep(30)
        subprocess.run(["supervisorctl", "stop", job], check=False)


if __name__ == "__main__":
    main()

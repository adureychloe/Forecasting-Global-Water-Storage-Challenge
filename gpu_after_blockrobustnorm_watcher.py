"""Hand off robust block normalization to center-only residuals."""

import json
import subprocess
import time
from pathlib import Path


LOG = Path("/workspace/water/logs/gpu-catboost-targetstats-blockrobustnorm.log")


def completed() -> bool:
    try:
        lines = LOG.read_text(encoding="utf-8", errors="ignore").splitlines()
        return bool(lines) and json.loads(lines[-1]).get("stage") == "complete"
    except Exception:
        return False


def main() -> None:
    while True:
        if completed():
            subprocess.run(["supervisorctl", "stop", "water_gpu_catboost_targetstats_blockrobustnorm"], check=False)
            subprocess.run(["supervisorctl", "start", "water_gpu_catboost_targetstats_blockcentered"], check=False)
            return
        time.sleep(30)


if __name__ == "__main__":
    main()

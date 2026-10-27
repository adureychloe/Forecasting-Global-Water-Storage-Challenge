"""Hand off extra XGBoost RMSE seeds to Ordered Bernoulli CatBoost."""

import json
import subprocess
import time
from pathlib import Path


LOG = Path("/workspace/water/logs/gpu-xgb-enhanced-rmse-extra.log")


def completed() -> bool:
    try:
        lines = LOG.read_text(encoding="utf-8", errors="ignore").splitlines()
        return bool(lines) and json.loads(lines[-1]).get("stage") == "complete"
    except Exception:
        return False


def main() -> None:
    while True:
        if completed():
            subprocess.run(["supervisorctl", "stop", "water_gpu_xgb_enhanced_rmse_extra"], check=False)
            subprocess.run(["supervisorctl", "start", "water_gpu_catboost_enhanced_ordered_bernoulli"], check=False)
            return
        time.sleep(30)


if __name__ == "__main__":
    main()

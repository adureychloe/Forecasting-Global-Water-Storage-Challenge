"""Supervisor-managed handoff from one GPU experiment to the next."""

import json
import subprocess
import time
from pathlib import Path

LOG = Path("/workspace/water/logs/gpu-lgbm-enhanced-goss.log")


def status(name):
    try:
        out = subprocess.check_output(["supervisorctl", "status", name], text=True)
    except Exception:
        return ""
    return out.strip()


def completed():
    try:
        lines = LOG.read_text(encoding="utf-8", errors="ignore").splitlines()
        if not lines:
            return False
        payload = json.loads(lines[-1])
        return payload.get("stage") == "complete"
    except Exception:
        return False


def main():
    while True:
        if completed():
            subprocess.run(["supervisorctl", "stop", "water_gpu_lgbm_enhanced_goss"], check=False)
            subprocess.run(["supervisorctl", "start", "water_gpu_catboost_enhanced_ordered_extra"], check=False)
            return
        time.sleep(30)


if __name__ == "__main__":
    main()

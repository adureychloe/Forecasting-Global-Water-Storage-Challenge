"""Hand off block-memory XGBoost to locally normalized Temporal CNN."""

import json
import subprocess
import time
from pathlib import Path


LOG = Path("/workspace/water/logs/gpu-xgb-enhanced-targetstats-blockfrozen.log")


def completed() -> bool:
    try:
        lines = LOG.read_text(encoding="utf-8", errors="ignore").splitlines()
        return bool(lines) and json.loads(lines[-1]).get("stage") == "complete"
    except Exception:
        return False


def main() -> None:
    while True:
        if completed():
            subprocess.run(
                ["supervisorctl", "stop", "water_gpu_xgb_enhanced_targetstats_blockfrozen"],
                check=False,
            )
            subprocess.run(["supervisorctl", "start", "water_gpu_torch_tcn_sequence_localnorm"], check=False)
            return
        time.sleep(30)


if __name__ == "__main__":
    main()

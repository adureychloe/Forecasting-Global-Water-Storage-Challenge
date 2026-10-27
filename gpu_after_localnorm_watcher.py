"""Hand off locally normalized Temporal CNN to block-memory LightGBM."""

import json
import subprocess
import time
from pathlib import Path


LOG = Path("/workspace/water/logs/gpu-torch-tcn-sequence-localnorm.log")


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
                ["supervisorctl", "stop", "water_gpu_torch_tcn_sequence_localnorm"],
                check=False,
            )
            subprocess.run(
                ["supervisorctl", "start", "water_gpu_lgbm_enhanced_targetstats_blockfrozen"],
                check=False,
            )
            return
        time.sleep(30)


if __name__ == "__main__":
    main()

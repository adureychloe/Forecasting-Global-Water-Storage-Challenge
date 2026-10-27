#!/bin/bash
set -euo pipefail
source /venv/main/bin/activate
cd /workspace/water
exec python -u gpu_xgb_blockfrozen_neighborhood.py

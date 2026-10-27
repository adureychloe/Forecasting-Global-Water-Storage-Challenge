#!/bin/bash
set -euo pipefail
source /venv/main/bin/activate
cd /workspace/water
exec python -u gpu_catboost_blockfrozen_nb_idw_era5.py

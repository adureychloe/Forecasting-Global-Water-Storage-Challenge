#!/bin/bash
set -e
cd /workspace/water
export HORIZON_MAX_LEAVES=127
export HORIZON_CAPACITY_PREFIX=horizon_capacity127
exec /venv/main/bin/python -u gpu_horizon_capacity.py

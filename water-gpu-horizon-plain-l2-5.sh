#!/bin/bash
set -e
cd /workspace/water
exec env HORIZON_PLAIN_SEEDS=77,2027 HORIZON_PLAIN_PREFIX=horizon_plain_l2_5 HORIZON_PLAIN_L2=5 /venv/main/bin/python -u gpu_horizon_plain.py

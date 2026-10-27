#!/bin/bash
set -e
cd /workspace/water
exec /venv/main/bin/python -u gpu_horizon_ordered_l2_5.py

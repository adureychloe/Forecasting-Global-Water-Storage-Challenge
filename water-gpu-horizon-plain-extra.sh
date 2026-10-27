#!/bin/bash
set -e
cd /workspace/water
exec env HORIZON_PLAIN_SEEDS=123,9001 HORIZON_PLAIN_PREFIX=horizon_plain_extra /venv/main/bin/python -u gpu_horizon_plain.py

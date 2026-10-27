#!/bin/bash
set -e
cd /workspace/water
exec env HORIZON_PLAIN_SEEDS=77,2027 HORIZON_PLAIN_PREFIX=horizon_plain_bern HORIZON_PLAIN_BOOTSTRAP=Bernoulli /venv/main/bin/python -u gpu_horizon_plain.py

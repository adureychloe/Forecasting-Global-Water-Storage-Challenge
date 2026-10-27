#!/bin/bash
set -e
cd /workspace/water
export HORIZON_LOSS='MAE'
export HORIZON_LOSS_PREFIX='horizon_loss_mae'
exec /venv/main/bin/python -u gpu_horizon_loss.py

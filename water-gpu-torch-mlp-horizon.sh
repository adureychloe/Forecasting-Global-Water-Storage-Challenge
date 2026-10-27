#!/bin/bash
set -e
cd /workspace/water
exec /venv/main/bin/python -u gpu_torch_mlp_horizon.py

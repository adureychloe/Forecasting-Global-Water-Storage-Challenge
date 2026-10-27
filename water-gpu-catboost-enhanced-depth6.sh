#!/bin/bash
set -e
cd /workspace/water
exec /venv/main/bin/python -u gpu_catboost_enhanced_depth6.py

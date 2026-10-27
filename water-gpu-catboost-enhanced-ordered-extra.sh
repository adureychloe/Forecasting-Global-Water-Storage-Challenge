#!/bin/bash
set -e
cd /workspace/water
exec /venv/main/bin/python -u gpu_catboost_enhanced_ordered_extra.py

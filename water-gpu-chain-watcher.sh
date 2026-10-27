#!/bin/bash
set -e
cd /workspace/water
exec /venv/main/bin/python -u gpu_chain_watcher.py

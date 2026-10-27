#!/bin/bash
set -euo pipefail
cd /workspace/water
marker=/workspace/water/results/window_aligned_v2_started.flag
while [ ! -f "$marker" ]; do
  if grep -q '"stage": "complete"' /workspace/water/results/explore.jsonl 2>/dev/null; then
    cp /workspace/water/gpu_explore_runner_new.py /workspace/water/gpu_explore_runner.py
    touch "$marker"
    supervisorctl restart water_gpu_explore
    echo "next queue started" >&2
  fi
  sleep 30
done
while true; do sleep 3600; done

#!/bin/bash
source /venv/main/bin/activate
cd /workspace/water
exec /venv/main/bin/python -u auto_promote.py

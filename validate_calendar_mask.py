"""Smoke-check the staged calendar-mask runner in its GPU environment."""

import importlib.util
from importlib.machinery import SourceFileLoader

import pandas as pd


loader = SourceFileLoader("gpu_explore_next", "gpu_explore_runner.py.next")
spec = importlib.util.spec_from_loader(loader.name, loader)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

frame = pd.read_csv("data/Train.csv", nrows=200_000, parse_dates=["time"]).drop(columns=["target"])
masked = module.apply_calendar_mask(frame.copy(), seed=42)
print(
    {
        "rows": len(masked),
        "mask_rate": float(masked["TWS_t_masked"].mean()),
        "rates_by_month": masked.groupby(masked["time"].dt.month)["TWS_t_masked"].mean().round(3).to_dict(),
    }
)

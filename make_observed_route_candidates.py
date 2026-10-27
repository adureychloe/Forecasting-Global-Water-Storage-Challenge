"""Create leakage-safe observed-row persistence route candidates."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TEST = pd.read_csv(ROOT / "data" / "Test.csv", usecols=["TWS_t", "TWS_t_masked"])
SUB = ROOT / "submission"
for src in sorted(SUB.glob("submission_cb90_all_lossguide63_seed2027_*.csv")):
    frame = pd.read_csv(src)
    pred = frame.iloc[:, 1].to_numpy(dtype=float)
    observed = ~TEST["TWS_t_masked"].to_numpy(dtype=bool)
    current = TEST["TWS_t"].to_numpy(dtype=float)
    for weight in (0.25, 0.50, 0.75, 1.00):
        values = pred.copy()
        values[observed] = (1.0 - weight) * pred[observed] + weight * current[observed]
        out = SUB / f"{src.stem}_obsroute{weight:.2f}.csv"
        result = frame.copy()
        result.iloc[:, 1] = values
        result.to_csv(out, index=False)
        print(out.name, int(observed.sum()), float(np.nanmin(values)), float(np.nanmax(values)))

"""Blend 2010+ aligned seed2027 into the new 0.7121 aligned seed77 2.5% leader.

Do not use the old seed2027 w025 that was mixed into 0.7139. Analog of
LGBM seed2027 into LGBM seed77. Do not agent-submit. First probe is 2.5%.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SUB = ROOT / "submission"
SAMPLE = ROOT / "data" / "SampleSubmission.csv"
LEADER = SUB / "submission_onlineleader_idw_era5_aligned_seed77_raw_w025.csv"
SRC = SUB / "submission_catboost_horizon_idw_era5_aligned_seed2027_aligned_raw.csv"
WEIGHTS = (0.025, 0.05, 0.10)


def main() -> None:
    ids = pd.read_csv(SAMPLE).iloc[:, 0]
    leader = pd.read_csv(LEADER)
    src = pd.read_csv(SRC)
    if not leader.iloc[:, 0].equals(ids) or not src.iloc[:, 0].equals(ids):
        raise ValueError("ID mismatch")
    lv = leader.iloc[:, 1].to_numpy(np.float64)
    sv = src.iloc[:, 1].to_numpy(np.float64)
    if not np.isfinite(lv).all() or not np.isfinite(sv).all():
        raise ValueError("non-finite")
    wrote = []
    for weight in WEIGHTS:
        out = leader.copy()
        mixed = (1.0 - weight) * lv + weight * sv
        out.iloc[:, 1] = mixed
        name = f"submission_onlineleader_idw_era5_aligned_seed2027_raw_w{int(weight * 1000):03d}.csv"
        path = SUB / name
        out.to_csv(path, index=False)
        check = pd.read_csv(path)
        if not check.iloc[:, 0].equals(ids):
            raise ValueError(f"ID rewrite {name}")
        cv = check.iloc[:, 1].to_numpy(np.float64)
        if np.max(np.abs(cv - mixed)) > 1e-12:
            raise ValueError(f"rewrite drift {name}")
        wrote.append(name)
    print({"leader": LEADER.name, "src": SRC.name, "rows": int(len(ids)), "wrote": wrote})


if __name__ == "__main__":
    main()

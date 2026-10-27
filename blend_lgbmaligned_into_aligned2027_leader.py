"""Blend 2010+ LGBM aligned Test CSVs into the 0.7105 CatBoost aligned seed2027 leader.

First probe is seed2027 2.5% (the LGBM seed already in the stack). Do not
agent-submit. Do not spend CatBoost aligned 5/10%.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SUB = ROOT / "submission"
SAMPLE = ROOT / "data" / "SampleSubmission.csv"
LEADER = SUB / "submission_onlineleader_idw_era5_aligned_seed2027_raw_w025.csv"
WEIGHTS = (0.025, 0.05, 0.10)
SOURCES = (
    ("lgbmaligned", "submission_lgbm_horizon_idw_era5_aligned_seed2027_aligned_raw.csv", 2027),
    ("lgbmaligned", "submission_lgbm_horizon_idw_era5_aligned_seed77_aligned_raw.csv", 77),
)


def load_pred(path: Path, ids: pd.Series) -> np.ndarray:
    frame = pd.read_csv(path)
    if len(frame) != len(ids):
        raise ValueError(f"{path.name}: {len(frame)} rows, expected {len(ids)}")
    if not frame.iloc[:, 0].equals(ids):
        raise ValueError(f"{path.name}: ID mismatch")
    values = frame.iloc[:, 1].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{path.name}: non-finite predictions")
    return values


def main() -> None:
    ids = pd.read_csv(SAMPLE).iloc[:, 0]
    leader = load_pred(LEADER, ids)
    wrote = []
    missing = []
    for tag, filename, seed in SOURCES:
        src_path = SUB / filename
        if not src_path.exists():
            missing.append(filename)
            continue
        src = load_pred(src_path, ids)
        for weight in WEIGHTS:
            mixed = (1.0 - weight) * leader + weight * src
            name = f"submission_onlineleader_idw_era5_{tag}_seed{seed}_raw_w{int(weight * 1000):03d}.csv"
            out = pd.read_csv(LEADER)
            out.iloc[:, 1] = mixed
            path = SUB / name
            out.to_csv(path, index=False)
            check = load_pred(path, ids)
            if np.max(np.abs(check - mixed)) > 1e-12:
                raise ValueError(f"rewrite drift {name}")
            wrote.append(name)
    print({"leader": LEADER.name, "rows": int(len(ids)), "wrote": wrote, "missing": missing})


if __name__ == "__main__":
    main()

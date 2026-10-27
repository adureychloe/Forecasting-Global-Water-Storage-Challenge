"""Local ensembles of existing 100% aligned sources. No labels, no submit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SUB = ROOT / "submission"
SAMPLE = ROOT / "data" / "SampleSubmission.csv"
LEADER = SUB / "submission_onlineleader_idw_era5_lgbmaligned_seed77_raw_w025.csv"

SOURCES_6 = (
    "submission_catboost_horizon_idw_era5_aligned_seed77_aligned_raw.csv",
    "submission_catboost_horizon_idw_era5_aligned_seed2027_aligned_raw.csv",
    "submission_lgbm_horizon_idw_era5_aligned_seed77_aligned_raw.csv",
    "submission_lgbm_horizon_idw_era5_aligned_seed2027_aligned_raw.csv",
    "submission_xgb_horizon_idw_era5_aligned_seed77_aligned_raw.csv",
    "submission_xgb_horizon_idw_era5_aligned_seed2027_aligned_raw.csv",
)
SOURCES_3 = (
    "submission_catboost_horizon_idw_era5_aligned_seed77_aligned_raw.csv",
    "submission_lgbm_horizon_idw_era5_aligned_seed77_aligned_raw.csv",
    "submission_xgb_horizon_idw_era5_aligned_seed77_aligned_raw.csv",
)
XGB77 = "submission_xgb_horizon_idw_era5_aligned_seed77_aligned_raw.csv"


def load(path: Path, ids: pd.Series) -> np.ndarray:
    frame = pd.read_csv(path)
    if len(frame) != len(ids) or not frame.iloc[:, 0].equals(ids):
        raise ValueError(path.name)
    values = frame.iloc[:, 1].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite {path.name}")
    return values


def write_pred(sample: pd.DataFrame, ids: pd.Series, values: np.ndarray, name: str) -> Path:
    out = sample.copy()
    out.iloc[:, 1] = values.astype(np.float64)
    path = SUB / name
    out.to_csv(path, index=False)
    check = pd.read_csv(path)
    if not check.iloc[:, 0].equals(ids):
        raise ValueError(f"ID rewrite {name}")
    cv = check.iloc[:, 1].to_numpy(np.float64)
    if np.max(np.abs(cv - values)) > 1e-12:
        raise ValueError(f"rewrite drift {name}")
    return path


def mix(leader: np.ndarray, src: np.ndarray, weight: float) -> np.ndarray:
    return (1.0 - weight) * leader + weight * src


def main() -> None:
    sample = pd.read_csv(SAMPLE)
    ids = sample.iloc[:, 0]
    leader = load(LEADER, ids)
    wrote = []
    stack6 = np.column_stack([load(SUB / name, ids) for name in SOURCES_6])
    stack3 = np.column_stack([load(SUB / name, ids) for name in SOURCES_3])
    mean6 = stack6.mean(axis=1)
    median6 = np.median(stack6, axis=1)
    mean3 = stack3.mean(axis=1)
    xgb77 = load(SUB / XGB77, ids)
    jobs = (
        ("submission_ens_aligned6_mean_raw.csv", mean6),
        ("submission_ens_aligned6_median_raw.csv", median6),
        ("submission_ens_aligned3_mean_raw.csv", mean3),
        ("submission_onlineleader_ens_aligned6_mean_w025.csv", mix(leader, mean6, 0.025)),
        ("submission_onlineleader_ens_aligned6_median_w025.csv", mix(leader, median6, 0.025)),
        ("submission_onlineleader_ens_aligned3_mean_w025.csv", mix(leader, mean3, 0.025)),
        ("submission_onlineleader_idw_era5_xgbaligned_seed77_raw_w025.csv", mix(leader, xgb77, 0.025)),
        ("submission_onlineleader_idw_era5_xgbaligned_seed77_raw_w050.csv", mix(leader, xgb77, 0.05)),
        ("submission_onlineleader_idw_era5_xgbaligned_seed77_raw_w100.csv", mix(leader, xgb77, 0.10)),
    )
    for name, values in jobs:
        write_pred(sample, ids, values, name)
        wrote.append(name)
    print({"rows": int(len(ids)), "wrote": wrote}, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

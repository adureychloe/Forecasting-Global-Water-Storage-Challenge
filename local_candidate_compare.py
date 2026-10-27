"""Compare existing Test CSVs against the 0.7076 leader. No labels, no submit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SUB = ROOT / "submission"
SAMPLE = ROOT / "data" / "SampleSubmission.csv"
LEADER = SUB / "submission_onlineleader_idw_era5_lgbmaligned_seed77_raw_w025.csv"
OUT = ROOT / "results" / "local_candidate_compare.csv"

FIXED = [
    "submission_onlineleader_idw_era5_lgbmaligned_seed77_raw_w025.csv",
    "submission_onlineleader_idw_era5_lgbmaligned_seed2027_raw_w025.csv",
    "submission_onlineleader_idw_era5_aligned_seed2027_raw_w025.csv",
    "submission_onlineleader_idw_era5_aligned_seed77_raw_w025.csv",
    "submission_onlineleader_idw_era5_xgbaligned_seed2027_raw_w025.csv",
    "submission_onlineleader_idw_era5_xgbaligned_seed77_raw_w025.csv",
    "submission_catboost_horizon_idw_era5_aligned_seed77_aligned_raw.csv",
    "submission_catboost_horizon_idw_era5_aligned_seed2027_aligned_raw.csv",
    "submission_lgbm_horizon_idw_era5_aligned_seed77_aligned_raw.csv",
    "submission_lgbm_horizon_idw_era5_aligned_seed2027_aligned_raw.csv",
    "submission_xgb_horizon_idw_era5_aligned_seed77_aligned_raw.csv",
    "submission_xgb_horizon_idw_era5_aligned_seed2027_aligned_raw.csv",
    "submission_catboost_horizon_idw_era5_aligned_seed77_static_raw.csv",
    "submission_lgbm_horizon_idw_era5_aligned_seed77_static_raw.csv",
    "submission_xgb_horizon_idw_era5_aligned_seed77_static_raw.csv",
    "submission_xgb_horizon_idw_era5_aligned_seed2027_static_raw.csv",
    "submission_catboost_horizon_idw_era5_pattern_seed77_pattern_raw.csv",
    "submission_lgbm_horizon_idw_era5_pattern_seed77_lgbmpattern_raw.csv",
    "submission_xgb_horizon_idw_era5_pattern_seed77_xgbpattern_raw.csv",
    "submission_ens_aligned6_mean_raw.csv",
    "submission_ens_aligned6_median_raw.csv",
    "submission_ens_aligned3_mean_raw.csv",
    "submission_onlineleader_ens_aligned6_mean_w025.csv",
    "submission_onlineleader_ens_aligned6_median_w025.csv",
    "submission_onlineleader_ens_aligned3_mean_w025.csv",
    "submission_onlineleader_lgbmaligned77_nbrec_a3_w15.csv",
    "submission_onlineleader_lgbmaligned77_nbrec_a3_w25.csv",
    "submission_onlineleader_lgbmaligned77_nbrec_a0_w15.csv",
]

GLOBS = (
    "*aligned_delta*",
    "*aligned_spatialiter*",
    "*aligned_climcluster*",
    "*aligned_extra*",
    "*ens_aligned*",
)


def load(path: Path, ids: pd.Series) -> np.ndarray:
    frame = pd.read_csv(path)
    if len(frame) != len(ids) or not frame.iloc[:, 0].equals(ids):
        raise ValueError(path.name)
    values = frame.iloc[:, 1].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite {path.name}")
    return values


def discover() -> list[str]:
    names = list(FIXED)
    seen = set(names)
    for pattern in GLOBS:
        for path in sorted(SUB.glob(pattern)):
            if path.suffix.lower() != ".csv" or path.name in seen:
                continue
            names.append(path.name)
            seen.add(path.name)
    return names


def main() -> None:
    ids = pd.read_csv(SAMPLE).iloc[:, 0]
    leader = load(LEADER, ids)
    rows = []
    for name in discover():
        path = SUB / name
        if not path.exists():
            rows.append({"file": name, "status": "missing"})
            continue
        pred = load(path, ids)
        rms = float(np.sqrt(np.mean((pred - leader) ** 2)))
        corr = float(np.corrcoef(pred, leader)[0, 1])
        rows.append(
            {
                "file": name,
                "status": "ok",
                "rms_vs_leader": rms,
                "corr_vs_leader": corr,
                "mean": float(pred.mean()),
                "std": float(pred.std()),
                "min": float(pred.min()),
                "max": float(pred.max()),
                "bytes": path.stat().st_size,
            }
        )
    frame = pd.DataFrame(rows)
    if "rms_vs_leader" in frame.columns:
        frame = frame.sort_values(["status", "rms_vs_leader"], ascending=[True, True], na_position="last")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)
    print(frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

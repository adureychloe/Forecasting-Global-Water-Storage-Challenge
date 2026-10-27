"""Validate #233 sources and write pattern_raw blends vs the 0.7179 leader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SUB = ROOT / "submission"
SAMPLE = ROOT / "data" / "SampleSubmission.csv"
LEADER = SUB / "submission_onlineleader_idw_era5_spatialiter_seed77_raw_w100.csv"
SPATIAL_77 = SUB / "submission_catboost_horizon_idw_era5_spatialiter_seed77_spatialiter_raw.csv"
SPATIAL_2027 = SUB / "submission_catboost_horizon_idw_era5_spatialiter_seed2027_spatialiter_raw.csv"
WEIGHTS = (0.025, 0.05, 0.10)


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


def rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    ids = pd.read_csv(SAMPLE).iloc[:, 0]
    leader = load_pred(LEADER, ids)
    spatial = {
        77: load_pred(SPATIAL_77, ids),
        2027: load_pred(SPATIAL_2027, ids),
    }
    sources = {}
    for seed in (77, 2027):
        for kind in ("pattern", "static"):
            path = SUB / f"submission_catboost_horizon_idw_era5_pattern_seed{seed}_{kind}_raw.csv"
            sources[(seed, kind)] = load_pred(path, ids)

    report = {
        "rows": int(len(ids)),
        "leader": LEADER.name,
        "pattern_vs_spatialiter": {},
        "pattern_vs_static": {},
        "existing_blend_max_abs": {},
        "wrote": [],
    }
    for seed in (77, 2027):
        report["pattern_vs_spatialiter"][seed] = {
            "rms": rms(sources[(seed, "pattern")], spatial[seed]),
            "corr": corr(sources[(seed, "pattern")], spatial[seed]),
        }
        report["pattern_vs_static"][seed] = {
            "rms": rms(sources[(seed, "pattern")], sources[(seed, "static")]),
            "corr": corr(sources[(seed, "pattern")], sources[(seed, "static")]),
        }

    for seed in (77, 2027):
        src = sources[(seed, "pattern")]
        for weight in WEIGHTS:
            out = pd.read_csv(LEADER)
            out.iloc[:, 1] = (1.0 - weight) * leader + weight * src
            values = out.iloc[:, 1].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"non-finite blend seed{seed} w{weight}")
            name = f"submission_onlineleader_idw_era5_pattern_seed{seed}_raw_w{int(weight * 1000):03d}.csv"
            path = SUB / name
            if path.exists() and seed == 2027:
                old = load_pred(path, ids)
                report["existing_blend_max_abs"][name] = float(np.max(np.abs(old - values)))
            out.to_csv(path, index=False)
            check = load_pred(path, ids)
            if np.max(np.abs(check - values)) > 1e-12:
                raise ValueError(f"rewrite drift {name}")
            report["wrote"].append(name)

    print(report)


if __name__ == "__main__":
    main()

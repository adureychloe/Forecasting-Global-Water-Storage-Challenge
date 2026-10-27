"""Recalibrate the existing leakage-safe shrinkage files for long gaps.

The calibrated files retain the baseline blend and fit-only bucket center.  For
gap7plus their recorded parameters are alpha=1 and blend=0.5, so the original
head prediction can be recovered algebraically and re-shrunk without retraining.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import gpu_nightly_runner as core
from gpu_dualhead_runner import mask_by_template, public_states

ROOT = Path(__file__).resolve().parent
SUB = ROOT / "submission"
DATA = ROOT / "data"
BUCKETS = ("observed", "gap1", "gap2", "gap3_6", "gap7plus")
FILES = {
    2027: "submission_horizon_shrink_seed2027_calibrated.csv",
    77: "submission_horizon_shrink_seed77_calibrated.csv",
    123: "submission_horizon_shrink_seed123_calibrated.csv",
    42: "submission_horizon_shrink_extra_seed42_calibrated.csv",
    9001: "submission_horizon_shrink_extra_seed9001_calibrated.csv",
}
PARAMS = {
    2027: {"observed": (1.0, 0.6), "gap1": (1.25, 0.5), "gap2": (1.5, 0.3), "gap3_6": (0.75, 0.6), "gap7plus": (1.0, 0.5)},
    77: {"observed": (1.0, 0.7), "gap1": (1.25, 0.5), "gap2": (1.5, 0.3), "gap3_6": (0.75, 0.6), "gap7plus": (1.0, 0.5)},
    123: {"observed": (1.0, 0.6), "gap1": (1.25, 0.5), "gap2": (1.5, 0.3), "gap3_6": (0.75, 0.6), "gap7plus": (1.0, 0.5)},
    42: {"observed": (1.0, 0.6), "gap1": (1.25, 0.5), "gap2": (1.5, 0.3), "gap3_6": (0.75, 0.6), "gap7plus": (1.0, 0.5)},
    9001: {"observed": (1.0, 0.7), "gap1": (1.25, 0.5), "gap2": (1.5, 0.3), "gap3_6": (0.75, 0.6), "gap7plus": (1.0, 0.5)},
}


def bucket(age: np.ndarray) -> np.ndarray:
    return np.select([age <= 0.5, age <= 1.5, age <= 2.5, age <= 6.5], BUCKETS[:4], default=BUCKETS[4])


def main() -> None:
    train = pd.read_csv(DATA / "Train.csv")
    test = pd.read_csv(DATA / "Test.csv")
    sample = pd.read_csv(DATA / "SampleSubmission.csv")
    train.time = pd.to_datetime(train.time)
    test.time = pd.to_datetime(test.time)
    states = public_states(test)
    train_input = mask_by_template(train.drop(columns=["target"]), states, align_validation=False)
    test_input = test.copy()
    test_input.loc[test_input.TWS_t_masked.astype(bool), "TWS_t"] = np.nan
    combined = pd.concat([train_input, test_input], ignore_index=True, sort=False)
    features = core.build_features(combined).replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    btrain = bucket(features.iloc[: len(train)]["TWS_t_age"].to_numpy(np.float32))
    btest = bucket(features.iloc[len(train) :]["TWS_t_age"].to_numpy(np.float32))
    target = train.target.to_numpy(float)
    centers = {name: float(np.mean(target[btrain == name])) for name in BUCKETS}
    baseline = pd.read_csv(SUB / "submission_aug35_ageflag_ens3_ridge30_a100.csv").iloc[:, 1].to_numpy(float)
    for seed, filename in FILES.items():
        frame = pd.read_csv(SUB / filename)
        calibrated = frame.iloc[:, 1].to_numpy(float)
        params = PARAMS[seed]
        blend = params["gap7plus"][1]
        keep = btest == "gap7plus"
        center = centers["gap7plus"]
        raw = center + (calibrated[keep] - (1 - blend) * baseline[keep] - blend * center) / blend
        for alpha in (0.75, 0.875):
            values = calibrated.copy()
            values[keep] = blend * (center + alpha * (raw - center)) + (1 - blend) * baseline[keep]
            out = frame.copy()
            out.iloc[:, 1] = values
            out.to_csv(SUB / f"submission_horizon_shrink_seed{seed}_gap7p_a{str(alpha).replace('.', '')}.csv", index=False)
    print({"buckets": {name: int(np.sum(btest == name)) for name in BUCKETS}, "center_gap7plus": centers["gap7plus"]})


if __name__ == "__main__":
    main()

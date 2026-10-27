"""Create stable three/four-seed block-frozen CatBoost ensembles."""

from pathlib import Path

import numpy as np
import pandas as pd


SUB = Path(__file__).resolve().parent / "submission"
BASE_PREFIX = "submission_catboost_horizon_enhanced_ordered_targetstats_blockfrozen"


def raw_path(seed: int) -> Path:
    extra = "_extra" if seed in (123, 9001) else ""
    return SUB / f"{BASE_PREFIX}{extra}_seed{seed}_raw.csv"


def main() -> None:
    ids = pd.read_csv(SUB.parent / "data" / "SampleSubmission.csv").iloc[:, 0]
    leader = pd.read_csv(SUB / "submission_onlinecontrol_ordered_l2_5_seed77_w05.csv")
    baseline = pd.read_csv(SUB / "submission_aug35_ageflag_ens3_ridge30_a100.csv")
    if not leader.iloc[:, 0].equals(ids) or not baseline.iloc[:, 0].equals(ids):
        raise ValueError("leader/base ID mismatch")

    aligned = {}
    for seed in (77, 2027, 123, 9001):
        raw = pd.read_csv(raw_path(seed))
        if not raw.iloc[:, 0].equals(ids):
            raise ValueError(f"raw ID mismatch: {seed}")
        frame = raw.copy()
        frame.iloc[:, 1] = (
            0.8 * raw.iloc[:, 1].to_numpy(dtype=float)
            + 0.2 * baseline.iloc[:, 1].to_numpy(dtype=float)
        )
        aligned[seed] = frame

    made = []
    for name, seeds in (("ens4", (77, 2027, 123, 9001)), ("ens3best", (2027, 123, 9001))):
        ensemble = aligned[seeds[0]].copy()
        ensemble.iloc[:, 1] = np.mean(
            [aligned[seed].iloc[:, 1].to_numpy(dtype=float) for seed in seeds], axis=0
        )
        source = SUB / f"{BASE_PREFIX}_{name}_b08.csv"
        ensemble.to_csv(source, index=False)
        made.append(source.name)
        for weight in (0.025, 0.05):
            out = leader.copy()
            out.iloc[:, 1] = (
                (1.0 - weight) * leader.iloc[:, 1].to_numpy(dtype=float)
                + weight * ensemble.iloc[:, 1].to_numpy(dtype=float)
            )
            path = SUB / f"submission_onlineleader_targetstats_blockfrozen_{name}_b08_w{int(weight * 1000):03d}.csv"
            out.to_csv(path, index=False)
            made.append(path.name)

    for name in made:
        frame = pd.read_csv(SUB / name)
        if len(frame) != len(ids) or not frame.iloc[:, 0].equals(ids):
            raise ValueError(f"ID mismatch: {name}")
        if not np.isfinite(frame.iloc[:, 1].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite predictions: {name}")
    print({"made": made, "rows": len(ids)})


if __name__ == "__main__":
    main()

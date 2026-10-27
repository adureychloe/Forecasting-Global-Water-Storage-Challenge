"""Create conservative online-leader blends for categorical-location CatBoost."""

from pathlib import Path

import numpy as np
import pandas as pd


SUB = Path(__file__).resolve().parent / "submission"


def validate(frame: pd.DataFrame, ids: pd.Series, name: str) -> None:
    if len(frame) != len(ids) or not frame.iloc[:, 0].equals(ids):
        raise ValueError(f"ID mismatch: {name}")
    if not np.isfinite(frame.iloc[:, 1].to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite predictions: {name}")


def main() -> None:
    leader = pd.read_csv(SUB / "submission_onlinecontrol_ordered_l2_5_seed77_w05.csv")
    base = pd.read_csv(SUB / "submission_aug35_ageflag_ens3_ridge30_a100.csv")
    ids = pd.read_csv(SUB.parent / "data" / "SampleSubmission.csv").iloc[:, 0]
    validate(leader, ids, "leader")
    validate(base, ids, "base")

    aligned = {}
    made = []
    for path in sorted(SUB.glob("submission_catboost_horizon_enhanced_ordered_loccat_seed*.csv")):
        validate(pd.read_csv(path), ids, path.name)

    for seed in (77, 2027):
        raw = pd.read_csv(
            SUB / f"submission_catboost_horizon_enhanced_ordered_loccat_seed{seed}_raw.csv"
        )
        frame = raw.copy()
        frame.iloc[:, 1] = (
            0.7 * raw.iloc[:, 1].to_numpy(dtype=float)
            + 0.3 * base.iloc[:, 1].to_numpy(dtype=float)
        )
        aligned[seed] = frame
        source = SUB / f"submission_catboost_horizon_enhanced_ordered_loccat_seed{seed}_blend0.7.csv"
        frame.to_csv(source, index=False)
        made.append(source.name)

        for weight in (0.025, 0.05):
            out = leader.copy()
            out.iloc[:, 1] = (
                (1.0 - weight) * leader.iloc[:, 1].to_numpy(dtype=float)
                + weight * frame.iloc[:, 1].to_numpy(dtype=float)
            )
            path = SUB / f"submission_onlineleader_loccat_seed{seed}_b07_w{int(weight * 1000):03d}.csv"
            out.to_csv(path, index=False)
            made.append(path.name)

    ensemble = aligned[77].copy()
    ensemble.iloc[:, 1] = (
        0.5 * aligned[77].iloc[:, 1].to_numpy(dtype=float)
        + 0.5 * aligned[2027].iloc[:, 1].to_numpy(dtype=float)
    )
    ensemble_path = SUB / "submission_catboost_horizon_enhanced_ordered_loccat_ens2_blend0.7.csv"
    ensemble.to_csv(ensemble_path, index=False)
    made.append(ensemble_path.name)
    for weight in (0.025, 0.05):
        out = leader.copy()
        out.iloc[:, 1] = (
            (1.0 - weight) * leader.iloc[:, 1].to_numpy(dtype=float)
            + weight * ensemble.iloc[:, 1].to_numpy(dtype=float)
        )
        path = SUB / f"submission_onlineleader_loccat_ens2_b07_w{int(weight * 1000):03d}.csv"
        out.to_csv(path, index=False)
        made.append(path.name)

    for name in made:
        validate(pd.read_csv(SUB / name), ids, name)
    print({"made": made, "rows": len(ids)})


if __name__ == "__main__":
    main()

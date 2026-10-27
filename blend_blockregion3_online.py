"""Create conservative online-leader blends for block-frozen region3 CatBoost."""

from pathlib import Path

import numpy as np
import pandas as pd


SUB = Path(__file__).resolve().parent / "submission"


def main() -> None:
    ids = pd.read_csv(SUB.parent / "data" / "SampleSubmission.csv").iloc[:, 0]
    leader = pd.read_csv(SUB / "submission_onlinecontrol_ordered_l2_5_seed77_w05.csv")
    base = pd.read_csv(SUB / "submission_aug35_ageflag_ens3_ridge30_a100.csv")
    raw = pd.read_csv(
        SUB / "submission_catboost_horizon_enhanced_ordered_targetstats_blockregion3_seed2027_raw.csv"
    )
    for name, frame in (("leader", leader), ("base", base), ("raw", raw)):
        if len(frame) != len(ids) or not frame.iloc[:, 0].equals(ids):
            raise ValueError(f"ID mismatch: {name}")
        if not np.isfinite(frame.iloc[:, 1].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite predictions: {name}")

    aligned = raw.copy()
    aligned.iloc[:, 1] = (
        0.7 * raw.iloc[:, 1].to_numpy(dtype=float)
        + 0.3 * base.iloc[:, 1].to_numpy(dtype=float)
    )
    aligned_path = SUB / "submission_catboost_horizon_enhanced_ordered_targetstats_blockregion3_seed2027_blend0.7.csv"
    aligned.to_csv(aligned_path, index=False)
    made = [aligned_path.name]
    for weight in (0.025, 0.05):
        out = leader.copy()
        out.iloc[:, 1] = (
            (1.0 - weight) * leader.iloc[:, 1].to_numpy(dtype=float)
            + weight * aligned.iloc[:, 1].to_numpy(dtype=float)
        )
        path = SUB / f"submission_onlineleader_blockregion3_seed2027_b07_w{int(weight * 1000):03d}.csv"
        out.to_csv(path, index=False)
        made.append(path.name)
    print({"made": made, "rows": len(ids)})


if __name__ == "__main__":
    main()

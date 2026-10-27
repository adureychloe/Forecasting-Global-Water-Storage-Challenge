"""Build local-only blends for the age-conditioned shrinkage candidates."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SUB = ROOT / "submission"


def read(name):
    frame = pd.read_csv(SUB / name)
    return frame


def main():
    groups = {
        "shrink_seed2027": read("submission_horizon_shrink_seed2027_calibrated.csv"),
        "shrink_seed77": read("submission_horizon_shrink_seed77_calibrated.csv"),
        "shrink_seed123": read("submission_horizon_shrink_seed123_calibrated.csv"),
        "shrink_seed42": read("submission_horizon_shrink_extra_seed42_calibrated.csv"),
        "shrink_seed9001": read("submission_horizon_shrink_extra_seed9001_calibrated.csv"),
        "horizon5_ens5": read("submission_horizon5_ens5_blend0.3.csv"),
        "online_control": read("submission_horizon_heads_seed2027_blend0.3.csv"),
    }
    ids = groups["shrink_seed2027"].iloc[:, 0]
    if any(not frame.iloc[:, 0].equals(ids) for frame in groups.values()):
        raise ValueError("submission ID order mismatch")
    made = []
    pairs = [
        ("shrink_ens2", ("shrink_seed2027", "shrink_seed77"), (0.5, 0.5)),
        ("shrink_ens3", ("shrink_seed2027", "shrink_seed77", "shrink_seed123"), (1 / 3, 1 / 3, 1 / 3)),
        ("shrink_ens3_control25", ("shrink_seed2027", "shrink_seed77", "online_control"), (0.375, 0.375, 0.25)),
        ("shrink_ens5", ("shrink_seed2027", "shrink_seed77", "shrink_seed123", "shrink_seed42", "shrink_seed9001"), (0.2, 0.2, 0.2, 0.2, 0.2)),
        ("shrink_ens2_control25", ("shrink_seed2027", "online_control"), (0.75, 0.25)),
        ("shrink_ens2_control50", ("shrink_seed2027", "online_control"), (0.5, 0.5)),
        ("shrink_ens2_horizon5_75", ("shrink_seed2027", "horizon5_ens5"), (0.75, 0.25)),
        ("shrink_ens2_horizon5_50", ("shrink_seed2027", "horizon5_ens5"), (0.5, 0.5)),
        ("shrink_ens2_horizon5_25", ("shrink_seed2027", "horizon5_ens5"), (0.25, 0.75)),
    ]
    for name, source_names, weights in pairs:
        out = groups[source_names[0]].copy()
        out.iloc[:, 1] = sum(w * groups[src].iloc[:, 1].to_numpy(float) for src, w in zip(source_names, weights))
        path = SUB / f"submission_{name}.csv"
        out.to_csv(path, index=False)
        made.append(path.name)
    print({"made": made})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

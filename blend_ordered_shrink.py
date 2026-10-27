"""Local-only low-weight Ordered/shrinkage blends."""

from pathlib import Path

import pandas as pd

SUB = Path(__file__).resolve().parent / "submission"


def main():
    pairs = [
        ("shrink_ordered10", "submission_horizon_shrink_seed2027_calibrated.csv", "submission_horizon_ordered_seed2027_blend0.3.csv", 0.9),
        ("shrink_ordered20", "submission_horizon_shrink_seed2027_calibrated.csv", "submission_horizon_ordered_seed2027_blend0.3.csv", 0.8),
        ("shrink_ordered_seed77_10", "submission_horizon_shrink_seed77_calibrated.csv", "submission_horizon_ordered_seed77_blend0.3.csv", 0.9),
        ("shrinkens5_ordered10", "submission_shrink_ens5.csv", "submission_horizon_ordered_seed2027_blend0.3.csv", 0.9),
    ]
    for name, left_name, right_name, lw in pairs:
        left, right = pd.read_csv(SUB / left_name), pd.read_csv(SUB / right_name)
        if not left.iloc[:, 0].equals(right.iloc[:, 0]): raise ValueError("ID mismatch")
        out = left.copy(); out.iloc[:, 1] = lw * left.iloc[:, 1].to_numpy(float) + (1 - lw) * right.iloc[:, 1].to_numpy(float)
        out.to_csv(SUB / f"submission_{name}.csv", index=False)
    print({"made": len(pairs)})


if __name__ == "__main__": main()

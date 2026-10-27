"""Create conservative online-leader blends for enhanced depth8 XGBoost."""

from pathlib import Path

import pandas as pd


SUB = Path(__file__).resolve().parent / "submission"


def main() -> None:
    leader = pd.read_csv(SUB / "submission_onlinecontrol_ordered_l2_5_seed77_w05.csv")
    made = []
    for seed in (42, 9001):
        source = SUB / f"submission_xgb_horizon_enhanced_depth8_seed{seed}_blend0.3.csv"
        pred = pd.read_csv(source)
        if not leader.iloc[:, 0].equals(pred.iloc[:, 0]):
            raise ValueError(source.name)
        for weight in (0.025, 0.05, 0.075):
            out = leader.copy()
            out.iloc[:, 1] = (
                (1 - weight) * leader.iloc[:, 1].to_numpy(float)
                + weight * pred.iloc[:, 1].to_numpy(float)
            )
            path = SUB / f"submission_onlineleader_xgb_depth8_seed{seed}_w{int(weight * 1000):03d}.csv"
            out.to_csv(path, index=False)
            made.append(path.name)
    print({"made": made, "rows": len(leader)})


if __name__ == "__main__":
    main()

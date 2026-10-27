"""Create validation-aligned and conservative online-leader target-stat blends."""

from pathlib import Path

import pandas as pd


SUB = Path(__file__).resolve().parent / "submission"


def main() -> None:
    leader = pd.read_csv(SUB / "submission_onlinecontrol_ordered_l2_5_seed77_w05.csv")
    base = pd.read_csv(SUB / "submission_aug35_ageflag_ens3_ridge30_a100.csv")
    aligned = {}
    made = []
    for seed in (77, 2027):
        raw = pd.read_csv(SUB / f"submission_catboost_horizon_enhanced_ordered_targetstats_seed{seed}_raw.csv")
        if not leader.iloc[:, 0].equals(raw.iloc[:, 0]) or not base.iloc[:, 0].equals(raw.iloc[:, 0]):
            raise ValueError(seed)
        source = raw.copy()
        source.iloc[:, 1] = 0.7 * raw.iloc[:, 1].to_numpy(float) + 0.3 * base.iloc[:, 1].to_numpy(float)
        source_path = SUB / f"submission_catboost_horizon_enhanced_ordered_targetstats_seed{seed}_blend0.7.csv"
        source.to_csv(source_path, index=False)
        made.append(source_path.name)
        aligned[seed] = source
        for source_name, frame in (("raw", raw), ("b07", source)):
            for weight in (0.025, 0.05):
                out = leader.copy()
                out.iloc[:, 1] = (
                    (1 - weight) * leader.iloc[:, 1].to_numpy(float)
                    + weight * frame.iloc[:, 1].to_numpy(float)
                )
                path = SUB / f"submission_onlineleader_targetstats_seed{seed}_{source_name}_w{int(weight * 1000):03d}.csv"
                out.to_csv(path, index=False)
                made.append(path.name)
    ensemble = aligned[77].copy()
    ensemble.iloc[:, 1] = 0.5 * aligned[77].iloc[:, 1].to_numpy(float) + 0.5 * aligned[2027].iloc[:, 1].to_numpy(float)
    ensemble_path = SUB / "submission_catboost_horizon_enhanced_ordered_targetstats_ens2_blend0.7.csv"
    ensemble.to_csv(ensemble_path, index=False)
    made.append(ensemble_path.name)
    for weight in (0.025, 0.05):
        out = leader.copy()
        out.iloc[:, 1] = (
            (1 - weight) * leader.iloc[:, 1].to_numpy(float)
            + weight * ensemble.iloc[:, 1].to_numpy(float)
        )
        path = SUB / f"submission_onlineleader_targetstats_ens2_b07_w{int(weight * 1000):03d}.csv"
        out.to_csv(path, index=False)
        made.append(path.name)
    print({"made": made, "rows": len(leader)})


if __name__ == "__main__":
    main()

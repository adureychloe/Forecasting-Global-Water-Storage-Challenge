"""Create conservative online-leader blends for block-frozen target statistics."""

from pathlib import Path

import numpy as np
import pandas as pd


SUB = Path(__file__).resolve().parent / "submission"
PREFIX = "catboost_horizon_enhanced_ordered_targetstats_blockfrozen"


def main() -> None:
    leader = pd.read_csv(SUB / "submission_onlinecontrol_ordered_l2_5_seed77_w05.csv")
    baseline = pd.read_csv(SUB / "submission_aug35_ageflag_ens3_ridge30_a100.csv")
    targets = {}
    made = []
    for seed in (77, 2027):
        raw = pd.read_csv(SUB / f"submission_{PREFIX}_seed{seed}_raw.csv")
        if not leader.iloc[:, 0].equals(raw.iloc[:, 0]) or not baseline.iloc[:, 0].equals(raw.iloc[:, 0]):
            raise ValueError(f"ID mismatch for seed {seed}")
        for model_weight in (0.7, 0.8):
            tag = f"b{int(model_weight * 10):02d}"
            source = raw.copy()
            source.iloc[:, 1] = (
                model_weight * raw.iloc[:, 1].to_numpy(float)
                + (1 - model_weight) * baseline.iloc[:, 1].to_numpy(float)
            )
            source_path = SUB / f"submission_{PREFIX}_seed{seed}_{tag}.csv"
            source.to_csv(source_path, index=False)
            made.append(source_path.name)
            targets[(seed, tag)] = source
            for online_weight in (0.025, 0.05):
                out = leader.copy()
                out.iloc[:, 1] = (
                    (1 - online_weight) * leader.iloc[:, 1].to_numpy(float)
                    + online_weight * source.iloc[:, 1].to_numpy(float)
                )
                path = SUB / (
                    f"submission_onlineleader_targetstats_blockfrozen_seed{seed}_{tag}"
                    f"_w{int(online_weight * 1000):03d}.csv"
                )
                out.to_csv(path, index=False)
                made.append(path.name)
    for tag in ("b07", "b08"):
        ensemble = targets[(77, tag)].copy()
        ensemble.iloc[:, 1] = 0.5 * (
            targets[(77, tag)].iloc[:, 1].to_numpy(float)
            + targets[(2027, tag)].iloc[:, 1].to_numpy(float)
        )
        source_path = SUB / f"submission_{PREFIX}_ens2_{tag}.csv"
        ensemble.to_csv(source_path, index=False)
        made.append(source_path.name)
        for online_weight in (0.025, 0.05):
            out = leader.copy()
            out.iloc[:, 1] = (
                (1 - online_weight) * leader.iloc[:, 1].to_numpy(float)
                + online_weight * ensemble.iloc[:, 1].to_numpy(float)
            )
            path = SUB / (
                f"submission_onlineleader_targetstats_blockfrozen_ens2_{tag}"
                f"_w{int(online_weight * 1000):03d}.csv"
            )
            out.to_csv(path, index=False)
            made.append(path.name)
    for name in made:
        frame = pd.read_csv(SUB / name)
        if len(frame) != len(leader) or not frame.iloc[:, 0].equals(leader.iloc[:, 0]):
            raise ValueError(f"invalid rows or IDs: {name}")
        if not np.isfinite(frame.iloc[:, 1].to_numpy(float)).all():
            raise ValueError(f"non-finite predictions: {name}")
    print({"made": len(made), "rows": len(leader), "files": made})


if __name__ == "__main__":
    main()

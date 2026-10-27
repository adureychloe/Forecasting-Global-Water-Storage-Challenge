"""Generate local diversity blends for the strongest new GPU candidate."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
SUB = ROOT / "submission"
new_prefix = "submission_cb665_d6_lr06_l2_5_seed42"
partners = {
    "old2027": "submission_cb_mask665_d6_seed2027",
    "oldens4": "submission_cb_mask665_d6_ens4",
}
seed_groups = {
    "ens42_9001": (42, 9001),
    "ens5": (42, 77, 123, 2027, 9001),
}
weights = ("blend0.2", "blend0.3", "blend0.4", "blend0.5", "blend0.6", "raw")


def main() -> int:
    made = 0
    for partner_name, partner_prefix in partners.items():
        for weight in weights:
            left_path = SUB / f"{new_prefix}_{weight}.csv"
            right_path = SUB / f"{partner_prefix}_{weight}.csv"
            if not left_path.exists() or not right_path.exists():
                continue
            left = pd.read_csv(left_path)
            right = pd.read_csv(right_path)
            if not left.iloc[:, 0].equals(right.iloc[:, 0]):
                raise ValueError(f"ID order mismatch: {left_path.name} vs {right_path.name}")
            out = left.copy()
            out.iloc[:, 1] = 0.5 * left.iloc[:, 1].to_numpy(float) + 0.5 * right.iloc[:, 1].to_numpy(float)
            out_path = SUB / f"submission_cb665_d6_lr06_l2_5_seed42_ens_{partner_name}_{weight}.csv"
            out.to_csv(out_path, index=False)
            made += 1
    for group_name, seeds in seed_groups.items():
        for weight in weights:
            frames = [pd.read_csv(SUB / f"submission_cb665_d6_lr06_l2_5_seed{seed}_{weight}.csv") for seed in seeds]
            if any(not frames[0].iloc[:, 0].equals(frame.iloc[:, 0]) for frame in frames[1:]):
                raise ValueError(f"ID order mismatch in {group_name}/{weight}")
            out = frames[0].copy()
            out.iloc[:, 1] = sum(frame.iloc[:, 1].to_numpy(float) for frame in frames) / len(frames)
            out_path = SUB / f"submission_cb665_d6_lr06_l2_5_{group_name}_{weight}.csv"
            out.to_csv(out_path, index=False)
            made += 1
    print({"made": made})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

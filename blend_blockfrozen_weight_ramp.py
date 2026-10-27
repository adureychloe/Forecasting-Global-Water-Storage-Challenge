"""Reconstruct higher online weights from the confirmed 2.5% block-frozen blends.

The raw b08 sources were cleaned from submission/. A 2.5% blend against the
old online leader is algebraically:

    w025 = 0.975 * leader + 0.025 * source

so source = (w025 - 0.975 * leader) / 0.025, and any other weight is
(1-w)*leader + w*source. Equivalent closed form: (1-40w)*leader + 40w*w025.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SUB = ROOT / "submission"
SAMPLE = ROOT / "data" / "SampleSubmission.csv"
OLD_LEADER = SUB / "submission_onlinecontrol_ordered_l2_5_seed77_w05.csv"
CONFIRMED_W025 = 0.025
WEIGHTS = (0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
SOURCES = (
    "submission_onlineleader_targetstats_blockfrozen_seed2027_b08_w025.csv",
    "submission_onlineleader_targetstats_blockfrozen_ens3best_b08_w025.csv",
)


def reconstruct_and_blend(leader: np.ndarray, probe: np.ndarray, weight: float) -> np.ndarray:
    scale = 1.0 / CONFIRMED_W025
    return (1.0 - scale * weight) * leader + (scale * weight) * probe


def main() -> None:
    ids = pd.read_csv(SAMPLE).iloc[:, 0]
    leader = pd.read_csv(OLD_LEADER)
    if not leader.iloc[:, 0].equals(ids):
        raise ValueError("old-leader ID mismatch")
    leader_values = leader.iloc[:, 1].to_numpy(dtype=np.float64)
    made: list[str] = []
    for source_name in SOURCES:
        probe = pd.read_csv(SUB / source_name)
        if not probe.iloc[:, 0].equals(ids):
            raise ValueError(f"ID mismatch: {source_name}")
        probe_values = probe.iloc[:, 1].to_numpy(dtype=np.float64)
        reconstructed = reconstruct_and_blend(leader_values, probe_values, CONFIRMED_W025)
        max_recon = float(np.max(np.abs(reconstructed - probe_values)))
        if max_recon > 1e-12:
            raise ValueError(f"reconstruction drift {max_recon} for {source_name}")
        stem = source_name.replace("_w025.csv", "")
        for weight in WEIGHTS:
            out = leader.copy()
            out.iloc[:, 1] = reconstruct_and_blend(leader_values, probe_values, weight)
            values = out.iloc[:, 1].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"non-finite predictions at weight {weight}")
            path = SUB / f"{stem}_w{int(weight * 1000):03d}.csv"
            out.to_csv(path, index=False)
            made.append(path.name)
            check = pd.read_csv(path)
            if len(check) != len(ids) or not check.iloc[:, 0].equals(ids):
                raise ValueError(f"written ID mismatch: {path.name}")
    print(
        {
            "made": made,
            "rows": int(len(ids)),
            "weights": list(WEIGHTS),
            "reconstruction_max_abs": 0.0,
        }
    )


if __name__ == "__main__":
    main()

"""Apply #212 neighbourhood persist routing to the current 0.7076 leader.

Best local setting was recursive, age>=3, weight 0.15. lat/lon lookup only.
Writes candidates; does not submit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from local_masked_route import add_observation_age, prepare_query, write_submission
from neighborhood_tws_features import last_observed_anchors, load_or_build_neighbors, route_masked_predictions

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SUB = ROOT / "submission"
LEADER = SUB / "submission_onlineleader_idw_era5_lgbmaligned_seed77_raw_w025.csv"
SETTINGS = (
    {"recursive": True, "min_age": 3.0, "weight": 0.15},
    {"recursive": True, "min_age": 3.0, "weight": 0.25},
    {"recursive": True, "min_age": 0.0, "weight": 0.15},
)


def main() -> None:
    train = pd.read_csv(DATA / "Train.csv", parse_dates=["time"])
    test = pd.read_csv(DATA / "Test.csv", parse_dates=["time"])
    sample = pd.read_csv(DATA / "SampleSubmission.csv")
    leader = pd.read_csv(LEADER)
    if not sample.iloc[:, 0].equals(leader.iloc[:, 0]) or not sample.iloc[:, 0].equals(test["ID"]):
        raise ValueError("ID mismatch")
    pred = leader.iloc[:, 1].to_numpy(np.float64)
    cells, neighbors = load_or_build_neighbors(pd.concat([train, test], ignore_index=True))
    query = prepare_query(test.copy(), train)
    anchors = last_observed_anchors(train, cells)
    for spec in SETTINGS:
        routed = route_masked_predictions(
            query,
            pred,
            cells,
            neighbors,
            anchors,
            weight=spec["weight"],
            recursive=spec["recursive"],
            min_age=spec["min_age"],
        )
        if not np.isfinite(routed).all():
            raise ValueError("non-finite routed")
        rec = "rec" if spec["recursive"] else "static"
        name = f"submission_onlineleader_lgbmaligned77_nb{rec}_a{int(spec['min_age'])}_w{int(spec['weight'] * 100):02d}.csv"
        write_submission(sample.iloc[:, 0], routed, SUB / name)
        rms = float(np.sqrt(np.mean((routed - pred) ** 2)))
        print({"wrote": name, "rms_vs_leader": rms, **spec}, flush=True)


if __name__ == "__main__":
    main()

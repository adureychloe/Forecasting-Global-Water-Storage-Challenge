"""Score leakage-safe persist vs neighbourhood-anchor persist on fold 0."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from coordinate_compliance import drop_coordinate_features
from enhanced_features import build_features
from local_neighborhood_train import FIT_ROWS, fold_frame, public_states, score
from neighborhood_tws_features import add_neighborhood_features


def rmse(y, pred):
    return float(np.sqrt(mean_squared_error(y, pred)))


def main() -> int:
    train = pd.read_csv("data/Train.csv")
    test = pd.read_csv("data/Test.csv")
    train["time"] = pd.to_datetime(train["time"])
    test["time"] = pd.to_datetime(test["time"])
    states = public_states(test)
    fit, valid, combined, cutoff, max_date = fold_frame(train, states, 0)
    y = valid["target"].to_numpy(dtype=np.float32)
    masked = combined["TWS_t_masked"].iloc[len(fit) :].to_numpy(dtype=bool)
    features = drop_coordinate_features(
        add_neighborhood_features(combined, build_features(combined).replace([np.inf, -np.inf], np.nan))
    )
    x_valid = features.iloc[len(fit) :]
    persist = x_valid["TWS_t_ffill"].to_numpy(dtype=np.float32)
    neighbor = x_valid["tws_neighbor_anchor_mean"].to_numpy(dtype=np.float32)
    neighbor = np.where(np.isfinite(neighbor), neighbor, persist)
    blend = 0.5 * persist + 0.5 * neighbor
    out = {
        "cutoff": str(cutoff.date()),
        "n_valid": int(len(valid)),
        "mask_rate": float(masked.mean()),
        "persist_ffill": score(y, persist, masked),
        "persist_neighbor": score(y, neighbor, masked),
        "persist_blend50": score(y, blend, masked),
    }
    print(json.dumps(out, ensure_ascii=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

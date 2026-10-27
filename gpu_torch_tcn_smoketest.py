"""Small runtime shape check for the remote Temporal CNN environment."""

import numpy as np
import pandas as pd

import gpu_torch_tcn_sequence as model


frame = pd.read_csv("data/Train.csv", nrows=50_000).drop(columns=["target"])
frame["time"] = pd.to_datetime(frame["time"])
coordinates = frame[["lat", "lon"]].drop_duplicates().head(5)
subset = frame.merge(coordinates, on=["lat", "lon"], how="inner")
sequence, static = model.make_arrays(subset)
print({"rows": len(subset), "seq_shape": sequence.shape, "static_shape": static.shape})
assert sequence.shape == (len(subset), 7, 12)
assert static.shape == (len(subset), 7)
assert np.isfinite(sequence).all() and np.isfinite(static).all()

"""Small runtime shape check for graph sequence construction."""

import numpy as np
import pandas as pd

import gpu_torch_graph_tcn_sequence as model


frame = pd.read_csv("data/Train.csv", nrows=100_000).drop(columns=["target"])
frame["time"] = pd.to_datetime(frame["time"])
coordinates = frame[["lat", "lon"]].drop_duplicates().head(20)
subset = frame.merge(coordinates, on=["lat", "lon"], how="inner")
sequence, static = model.graph_make_arrays(subset)
print({"rows": len(subset), "seq_shape": sequence.shape, "static_shape": static.shape})
assert sequence.shape == (len(subset), 13, 12)
assert static.shape == (len(subset), 7)
assert np.isfinite(sequence).all() and np.isfinite(static).all()

import pandas as pd
from pathlib import Path

path = Path("/workspace/water/results/era5_cell_month.parquet")
table = pd.read_parquet(path)
print({"rows": len(table), "cols": int(table.shape[1]), "cells": int(table[["lat", "lon"]].drop_duplicates().shape[0]), "bytes": path.stat().st_size})

"""Check optimized query features against the existing full-history builder."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import enhanced_features
import neighborhood_tws_features as nb
from sparse_model_features import dynamic_query_features


class QueryFeatureTest(unittest.TestCase):
    def test_query_only_neighbors_equal_full_history_with_missing_cells(self):
        # A future implementation that drops history before forward filling,
        # or shifts across cells, must fail this comparison.
        rows = []
        for t in pd.date_range('2010-01-01', periods=16, freq='MS'):
            for lon in (1., 2., 3.):
                if t.month == 4 and lon == 3.:
                    continue
                rows.append({'time': t, 'lat': 0., 'lon': lon,
                             'TWS_t': np.nan if t.month in (3, 4) else lon + t.month,
                             'SPEI_03_t': lon, 'SOIL_MOISTURE_t': t.month / 10,
                             'TWS_t_masked': t.month in (3, 4)})
        raw = pd.DataFrame(rows)
        history = raw.loc[raw.time < raw.time.max()].reset_index(drop=True)
        current = raw.loc[raw.time == raw.time.max()].reset_index(drop=True)
        old_cache, old_dist = nb.CACHE, nb.DIST_CACHE
        try:
            with tempfile.TemporaryDirectory() as folder:
                nb.CACHE = Path(folder) / 'neighbors.npz'
                nb.DIST_CACHE = Path(folder) / 'dist.npz'
                combined = pd.concat([history, current], ignore_index=True)
                full = nb.add_rich_neighborhood_features(
                    combined, enhanced_features.build_features(combined), fill_missing_with_anchor=True)
                expected = full.iloc[-len(current):].reset_index(drop=True)
                actual = dynamic_query_features(history, current)
                pd.testing.assert_frame_equal(actual, expected)
        finally:
            nb.CACHE, nb.DIST_CACHE = old_cache, old_dist


if __name__ == '__main__':
    unittest.main()

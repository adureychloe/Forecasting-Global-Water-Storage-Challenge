"""Query feature reuse for sparse replay without rebuilding fit-period neighbors."""
import pandas as pd
import enhanced_features
import neighborhood_tws_features


def dynamic_query_features(history, current):
    combined = pd.concat([history, current], ignore_index=True, sort=False)
    # Compute lags/forward fill with all available history before slicing.
    temporal = enhanced_features.build_features(combined).iloc[-len(current):].reset_index(drop=True)
    # Rich neighborhood columns depend only on this month's rows and their
    # already-computed past anchors. Absent cells have no field value either way.
    return neighborhood_tws_features.add_rich_neighborhood_features(
        current.reset_index(drop=True), temporal, fill_missing_with_anchor=True)

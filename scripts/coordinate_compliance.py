"""Drop banned coordinate predictors while keeping lat/lon as lookup keys."""

COORDINATE_FEATURE_COLUMNS = (
    "lat",
    "lon",
    "lat_sin",
    "lat_cos",
    "lon_sin",
    "lon_cos",
    "loc_id",
)


def drop_coordinate_features(features):
    return features.drop(
        columns=[column for column in COORDINATE_FEATURE_COLUMNS if column in features.columns],
        errors="ignore",
    )

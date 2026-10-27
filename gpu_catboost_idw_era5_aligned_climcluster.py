"""2010+ CatBoost aligned walk plus fit-only ERA5 climate clusters."""

import gpu_catboost_idw_era5_aligned as base
import climate_cluster_features
from coordinate_compliance import drop_coordinate_features

_BASE_AUGMENT = base.augment_features


def augment_features(x_fit, y_fit, x_query, fit_raw, query_raw):
    x_fit, x_query = _BASE_AUGMENT(x_fit, y_fit, x_query, fit_raw, query_raw)
    x_fit, x_query = climate_cluster_features.augment_features(x_fit, x_query, fit_raw, query_raw)
    return drop_coordinate_features(x_fit), drop_coordinate_features(x_query)


base.augment_features = augment_features
base.runner.augment_supervised_features = augment_features
base.runner.PREFIX = "catboost_horizon_idw_era5_aligned_climcluster"
base.runner.LOG = base.runner.RESULTS / f"{base.runner.PREFIX}.jsonl"
if __name__ == "__main__":
    raise SystemExit(base.custom_main())

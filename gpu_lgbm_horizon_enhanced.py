"""LightGBM horizon heads with exact-month and drought interaction features."""

import enhanced_features
import gpu_lgbm_horizon as runner

runner.core.build_features = enhanced_features.build_features
runner.runner.core.build_features = enhanced_features.build_features
runner.runner.PREFIX = "lgbm_horizon_enhanced_exact"
runner.runner.LOG = runner.runner.RESULTS / "lgbm_horizon_enhanced_exact.jsonl"
runner.runner.SEEDS = (77, 2027)

raise SystemExit(runner.main())

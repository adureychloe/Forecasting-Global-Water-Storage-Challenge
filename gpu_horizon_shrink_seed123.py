"""Independent seed123 confirmation for the age-conditioned shrinkage route."""

import gpu_horizon_shrink as route

route.runner.SEEDS = (123,)
route.runner.PREFIX = "horizon_shrink_seed123"
route.runner.LOG = route.runner.RESULTS / "horizon_shrink_seed123.jsonl"
raise SystemExit(route.main())

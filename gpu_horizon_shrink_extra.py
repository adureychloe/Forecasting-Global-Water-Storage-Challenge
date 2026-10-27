"""Complete the five-seed age-conditioned shrinkage ensemble."""

import gpu_horizon_shrink as route

route.runner.SEEDS = (42, 9001)
route.runner.PREFIX = "horizon_shrink_extra"
route.runner.LOG = route.runner.RESULTS / "horizon_shrink_extra.jsonl"
raise SystemExit(route.main())

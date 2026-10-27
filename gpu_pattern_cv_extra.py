"""Exact public-pattern CV for additional horizon seeds 42 and 9001."""

import gpu_pattern_cv as route

route.SEEDS = (42, 9001)
route.LOG = route.RESULTS / "pattern_cv_extra.jsonl"
raise SystemExit(route.main())

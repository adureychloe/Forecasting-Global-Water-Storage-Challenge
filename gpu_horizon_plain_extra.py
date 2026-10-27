"""Extra-seed continuation for the SymmetricTree/Plain horizon probe."""

import gpu_horizon_plain as base

base.runner.SEEDS = (123, 9001)
base.runner.PREFIX = "horizon_plain_extra"
base.runner.LOG = base.runner.RESULTS / "horizon_plain_extra.jsonl"
raise SystemExit(base.runner.main())

"""Run a distinct TWS age boundary variant using the horizon runner engine."""
import gpu_horizon_runner as runner

runner.PREFIX = "horizon_variant_0_1_2_4_5"
runner.LOG = runner.RESULTS / f"{runner.PREFIX}.jsonl"
runner.BUCKETS = ("observed", "gap1", "gap2_4", "gap5plus")


def variant_bucket(age):
    import numpy as np
    return np.select([age <= 0.5, age <= 1.5, age <= 4.5], runner.BUCKETS[:3], default=runner.BUCKETS[3])


runner.bucket = variant_bucket
raise SystemExit(runner.main())

"""Additional seeds for the five-head horizon router."""

import gpu_horizon_runner as runner


runner.PREFIX = "horizon_variant_age0_1_2_3_6_7p_extra"
runner.LOG = runner.RESULTS / f"{runner.PREFIX}.jsonl"
runner.SEEDS = (123, 42, 9001)
runner.BUCKETS = ("observed", "gap1", "gap2", "gap3_6", "gap7plus")


def variant_bucket(age):
    import numpy as np

    return np.select(
        [age <= 0.5, age <= 1.5, age <= 2.5, age <= 6.5],
        runner.BUCKETS[:4],
        default=runner.BUCKETS[4],
    )


runner.bucket = variant_bucket
raise SystemExit(runner.main())

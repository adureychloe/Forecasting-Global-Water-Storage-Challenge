"""Five-head horizon-age routing variant for the official mask schedule."""

import os
import gpu_horizon_runner as runner


runner.PREFIX = os.environ.get("HORIZON_PREFIX", "horizon_variant_age0_1_2_3_6_7p")
runner.LOG = runner.RESULTS / f"{runner.PREFIX}.jsonl"
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

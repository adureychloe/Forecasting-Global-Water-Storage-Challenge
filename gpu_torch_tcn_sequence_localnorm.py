"""Temporal CNN with Web-Traffic-style sample-wise TWS normalization."""

import gpu_torch_tcn_sequence as runner


runner.LOG = runner.RESULTS / "torch_tcn_sequence_localnorm.jsonl"
runner.PREFIX = "torch_tcn_sequence_localnorm"
base_make_arrays = runner.make_arrays


def make_arrays(frame):
    return base_make_arrays(frame, local_normalize=True)


runner.make_arrays = make_arrays
raise SystemExit(runner.main())

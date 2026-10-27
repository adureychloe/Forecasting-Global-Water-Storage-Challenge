"""Rebuild the selected private-leaderboard submission from nested blends.

Selected file:
  submission_onlineleader_idw_era5_lgbmaligned_extra_seed123_raw_w025.csv
Zindi submission ID: KJhqz4jM
Public LB RMSE: 0.6955

The mix is a convex combination of previously generated aligned-walk CSVs
plus the 0.7139 public-leader base. This script is bit-stable given those
inputs. It does not retrain models.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1] if (ROOT.parents[1] / "submission").exists() else ROOT.parent.parent
EXPECTED_NAME = "submission_onlineleader_idw_era5_lgbmaligned_extra_seed123_raw_w025.csv"
EXPECTED_SHA256 = "c2f4c5018bdc0a578c055fa0d2179d3e9785df35f989c13335f01bab9e483b84"
N_ROWS = 280961

PARENT = "submission_onlineleader_idw_era5_xgbaligned_seed77_raw_w200.csv"
LGBM123 = "submission_lgbm_horizon_idw_era5_aligned_extra_seed123_aligned_raw.csv"
CHAIN = (
    ("base_7139", "submission_onlineleader_idw_era5_lgbmpattern_seed2027_raw_w025.csv"),
    ("cb77", "submission_catboost_horizon_idw_era5_aligned_seed77_aligned_raw.csv"),
    ("cb2027", "submission_catboost_horizon_idw_era5_aligned_seed2027_aligned_raw.csv"),
    ("lgbm2027", "submission_lgbm_horizon_idw_era5_aligned_seed2027_aligned_raw.csv"),
    ("lgbm77", "submission_lgbm_horizon_idw_era5_aligned_seed77_aligned_raw.csv"),
    ("xgb77", "submission_xgb_horizon_idw_era5_aligned_seed77_aligned_raw.csv"),
    ("lgbm123", LGBM123),
    ("parent_6955", PARENT),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_source_dir(explicit: Path | None) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend([ROOT / "sources", REPO / "submission"])
    for path in candidates:
        if path.exists() and (path / CHAIN[0][1]).exists():
            return path
    raise FileNotFoundError("Could not find source CSVs. Pass --source-dir.")


def load_pred(path: Path, ids: pd.Series) -> np.ndarray:
    frame = pd.read_csv(path)
    if len(frame) != len(ids):
        raise ValueError(f"{path.name}: {len(frame)} rows, expected {len(ids)}")
    if not frame.iloc[:, 0].equals(ids):
        raise ValueError(f"{path.name}: ID order does not match SampleSubmission")
    values = frame.iloc[:, 1].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{path.name}: non-finite predictions")
    return values


def nested_blend(parts: dict[str, np.ndarray]) -> np.ndarray:
    leader = 0.975 * parts["base_7139"] + 0.025 * parts["cb77"]
    leader = 0.975 * leader + 0.025 * parts["cb2027"]
    leader = 0.975 * leader + 0.025 * parts["lgbm2027"]
    leader = 0.975 * leader + 0.025 * parts["lgbm77"]
    leader = 0.900 * leader + 0.100 * parts["xgb77"]
    leader_6957 = 0.700 * leader + 0.300 * parts["xgb77"]
    leader_6955 = 0.800 * leader_6957 + 0.200 * parts["xgb77"]
    return 0.975 * leader_6955 + 0.025 * parts["lgbm123"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce submission KJhqz4jM")
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--sample", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "output" / EXPECTED_NAME)
    parser.add_argument("--expected", type=Path, default=None)
    args = parser.parse_args()

    source_dir = find_source_dir(args.source_dir)
    sample_path = args.sample or REPO / "data" / "SampleSubmission.csv"
    if not sample_path.exists():
        raise FileNotFoundError(f"SampleSubmission not found: {sample_path}")
    sample = pd.read_csv(sample_path)
    if len(sample) != N_ROWS:
        raise ValueError(f"SampleSubmission has {len(sample)} rows, expected {N_ROWS}")
    ids = sample.iloc[:, 0]

    parts = {key: load_pred(source_dir / name, ids) for key, name in CHAIN}
    pred = 0.975 * parts["parent_6955"] + 0.025 * parts["lgbm123"]
    nested = nested_blend(parts)
    nested_rms = float(np.sqrt(np.mean((pred - nested) ** 2)))
    if nested_rms > 1e-12:
        raise ValueError(f"nested history disagrees with last-step mix: rms={nested_rms}")
    if not np.isfinite(pred).all():
        raise ValueError("non-finite blended predictions")
    print(f"nested-history rms vs last-step mix: {nested_rms:.3e}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_frame = pd.DataFrame({sample.columns[0]: ids, sample.columns[1]: pred})
    out_frame.to_csv(args.out, index=False)

    expected_path = args.expected
    if expected_path is None:
        for candidate in (
            ROOT / "expected" / EXPECTED_NAME,
            source_dir / EXPECTED_NAME,
            REPO / "submission" / EXPECTED_NAME,
        ):
            if candidate.exists():
                expected_path = candidate
                break
    print(f"wrote {args.out}")
    print(f"sha256 {sha256_file(args.out)}")
    if expected_path is not None:
        expected = load_pred(expected_path, ids)
        rms = float(np.sqrt(np.mean((pred - expected) ** 2)))
        max_abs = float(np.max(np.abs(pred - expected)))
        print(f"vs expected {expected_path.name}: rms={rms:.3e} maxabs={max_abs:.3e}")
        print(f"expected sha256 {sha256_file(expected_path)}")
        if rms > 1e-12:
            raise SystemExit("reproduced file does not match expected submission")
    elif sha256_file(args.out) != EXPECTED_SHA256:
        print("warning: no expected file present; compare SHA256 to DOCUMENTATION.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

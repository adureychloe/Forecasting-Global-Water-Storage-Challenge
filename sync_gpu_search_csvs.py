"""Download new search-queue CSVs via staging. Never treat a transfer as complete."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SUB = ROOT / "submission"
SAMPLE = ROOT / "data" / "SampleSubmission.csv"
STAGING = ROOT / "results" / "csv_staging_search"
REMOTE = "/workspace/water/submission"
HOST = "root@175.155.64.164"
PORT = "19689"
KEY = Path.home() / ".ssh" / "id_ed25519"
PREFIXES = (
    "submission_catboost_horizon_idw_era5_aligned_delta_",
    "submission_catboost_horizon_idw_era5_aligned_spatialiter_",
    "submission_catboost_horizon_idw_era5_aligned_climcluster_",
    "submission_catboost_horizon_idw_era5_aligned_extra_",
    "submission_lgbm_horizon_idw_era5_aligned_extra_",
    "submission_xgb_horizon_idw_era5_aligned_extra_",
)


def ssh_cmd() -> list[str]:
    return [
        "ssh",
        "-F",
        "NUL",
        "-i",
        str(KEY),
        "-o",
        "StrictHostKeyChecking=no",
        "-p",
        PORT,
        HOST,
    ]


def scp_cmd() -> list[str]:
    return [
        "scp",
        "-F",
        "NUL",
        "-i",
        str(KEY),
        "-o",
        "StrictHostKeyChecking=no",
        "-P",
        PORT,
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def list_remote() -> list[tuple[str, int]]:
    patterns = " ".join(f"{REMOTE}/{prefix}*.csv" for prefix in PREFIXES)
    command = f"ls -1 {patterns} 2>/dev/null | xargs -r stat -c '%n %s'"
    result = subprocess.run(ssh_cmd() + [command], check=False, capture_output=True, text=True)
    rows = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or " " not in line:
            continue
        path, size = line.rsplit(" ", 1)
        rows.append((Path(path).name, int(size)))
    return rows


def validate(path: Path, ids: pd.Series) -> None:
    frame = pd.read_csv(path)
    if len(frame) != len(ids) or not frame.iloc[:, 0].equals(ids):
        raise ValueError(f"ID mismatch {path.name}")
    values = frame.iloc[:, 1].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite {path.name}")


def main() -> None:
    STAGING.mkdir(parents=True, exist_ok=True)
    ids = pd.read_csv(SAMPLE).iloc[:, 0]
    remote = list_remote()
    copied = []
    skipped = []
    for name, size in remote:
        dest = SUB / name
        if dest.exists() and dest.stat().st_size == size:
            skipped.append(name)
            continue
        part = STAGING / f"{name}.part"
        remote_path = f"{HOST}:{REMOTE}/{name}"
        subprocess.run(scp_cmd() + [remote_path, str(part)], check=True)
        if part.stat().st_size != size:
            raise ValueError(f"size mismatch {name}: {part.stat().st_size} != {size}")
        validate(part, ids)
        digest = sha256(part)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        shutil.copy2(part, tmp)
        tmp.replace(dest)
        copied.append({"file": name, "bytes": size, "sha256": digest})
    print({"copied": copied, "skipped": skipped}, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

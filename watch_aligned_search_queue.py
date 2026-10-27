"""Poll GPU search-queue health. Print only DONE/FAILED/ACTION_REQUIRED."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HOST = "root@175.155.64.164"
PORT = "19689"
KEY = Path.home() / ".ssh" / "id_ed25519"
LOG_DIR = Path.home() / ".grok" / "long-running-background-tasks"
POLL_SEC = 30
STALL_SEC = 90 * 60
JOBS = (
    ("water_gpu_catboost_idw_era5_aligned_delta", "/workspace/water/logs/gpu-catboost-idw-era5-aligned-delta.log"),
    ("water_gpu_catboost_idw_era5_aligned_spatialiter", "/workspace/water/logs/gpu-catboost-idw-era5-aligned-spatialiter.log"),
    ("water_gpu_catboost_idw_era5_aligned_climcluster", "/workspace/water/logs/gpu-catboost-idw-era5-aligned-climcluster.log"),
    ("water_gpu_aligned_inner3_cv", "/workspace/water/logs/gpu-aligned-inner3-cv.log"),
    ("water_gpu_aligned_delta_cv", "/workspace/water/logs/gpu-aligned-delta-cv.log"),
    ("water_gpu_catboost_idw_era5_aligned_extra", "/workspace/water/logs/gpu-catboost-idw-era5-aligned-extra.log"),
    ("water_gpu_lgbm_idw_era5_aligned_extra", "/workspace/water/logs/gpu-lgbm-idw-era5-aligned-extra.log"),
    ("water_gpu_xgb_idw_era5_aligned_extra", "/workspace/water/logs/gpu-xgb-idw-era5-aligned-extra.log"),
)
QUEUE = "water_gpu_aligned_search_queue"
REMOTE_CMD = """
/usr/local/bin/supervisorctl status {queue} {jobs}
echo '__LOGS__'
{tails}
echo '__ERR__'
{errs}
""".format(
    queue=QUEUE,
    jobs=" ".join(name for name, _ in JOBS),
    tails="\n".join(f"echo FILE {log}; tail -n 1 {log} 2>/dev/null || true" for _, log in JOBS),
    errs="\n".join(
        f"echo ERR {log.replace('.log', '.err.log')}; tail -n 8 {log.replace('.log', '.err.log')} 2>/dev/null || true"
        for _, log in JOBS
    ),
)


def ssh_raw() -> str:
    cmd = [
        "ssh",
        "-F",
        "NUL",
        "-i",
        str(KEY),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=20",
        "-p",
        PORT,
        HOST,
        REMOTE_CMD,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout)[-800:] or f"ssh {result.returncode}")
    return result.stdout


def parse(raw: str) -> dict:
    text = raw.split("Welcome to vast.ai", 1)[-1] if "Welcome to vast.ai" in raw else raw
    if "__LOGS__" not in text:
        raise RuntimeError("missing log marker")
    status_blob, rest = text.split("__LOGS__", 1)
    log_blob, err_blob = rest.split("__ERR__", 1) if "__ERR__" in rest else (rest, "")
    alive = {}
    for line in status_blob.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("water_gpu_"):
            alive[parts[0]] = parts[1]
    jobs = {}
    current_file = None
    current_last = None
    for line in log_blob.splitlines():
        if line.startswith("FILE "):
            current_file = line[5:].strip()
            current_last = None
            continue
        if current_file:
            current_last = line.strip()
            name = None
            for job_name, log in JOBS:
                if log == current_file:
                    name = job_name
                    break
            payload = {}
            if current_last:
                try:
                    payload = json.loads(current_last)
                except Exception:
                    payload = {"raw": current_last[:400]}
            jobs[name or current_file] = {
                "log": current_file,
                "supervisor": alive.get(name, "MISSING") if name else "MISSING",
                "last": payload,
            }
            current_file = None
    for name, log in JOBS:
        jobs.setdefault(name, {"log": log, "supervisor": alive.get(name, "MISSING"), "last": {}})
    err_text = err_blob.strip()
    current = None
    last_complete = None
    for name, _ in JOBS:
        row = jobs[name]
        stage = (row.get("last") or {}).get("stage")
        if stage == "complete":
            last_complete = name
        elif row.get("supervisor") == "RUNNING" and current is None:
            current = name
        elif stage and current is None and row.get("supervisor") != "STOPPED":
            if stage != "complete":
                current = name
    return {
        "queue": alive.get(QUEUE, "MISSING"),
        "current": current,
        "last_complete": last_complete,
        "jobs": jobs,
        "err": err_text[-1200:],
        "alive": alive,
    }


def snapshot(info: dict) -> tuple:
    current = info.get("current")
    last = ((info.get("jobs") or {}).get(current) or {}).get("last") or {}
    return (
        info.get("queue"),
        current,
        last.get("stage"),
        last.get("seed"),
        last.get("month"),
        last.get("bucket"),
        info.get("last_complete"),
    )


def all_complete(info: dict) -> bool:
    jobs = info.get("jobs") or {}
    return all(((jobs.get(name) or {}).get("last") or {}).get("stage") == "complete" for name, _ in JOBS)


def failed(info: dict) -> str | None:
    jobs = info.get("jobs") or {}
    for name, _ in JOBS:
        row = jobs.get(name) or {}
        last = row.get("last") or {}
        state = row.get("supervisor")
        raw = last.get("raw")
        if isinstance(raw, str) and ("Traceback" in raw or "Error" in raw):
            return f"{name} log error {raw[:200]}"
        if state in {"FATAL", "BACKOFF"}:
            return f"{name} {state}"
        if state == "EXITED" and last.get("stage") != "complete":
            return f"{name} EXITED before complete last={last}"
    queue = info.get("queue")
    if queue in {"FATAL", "BACKOFF"}:
        return f"queue {queue}"
    if queue == "EXITED" and not all_complete(info):
        return f"queue EXITED before extra-xgb complete current={info.get('current')}"
    if queue == "MISSING" and not info.get("current") and not info.get("last_complete"):
        return "queue MISSING and no job logs"
    return None


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    diag = LOG_DIR / "watch_aligned_search_queue.log"
    prev = None
    last_change = time.time()
    ssh_fails = 0
    while True:
        try:
            info = parse(ssh_raw())
            ssh_fails = 0
        except Exception as exc:
            ssh_fails += 1
            with diag.open("a", encoding="utf-8") as handle:
                handle.write(f"ssh_fail {ssh_fails} {exc}\n")
            if ssh_fails >= 4:
                print(f"FAILED: ssh unreachable x{ssh_fails}: {exc}", flush=True)
                return 1
            time.sleep(POLL_SEC)
            continue
        now = snapshot(info)
        with diag.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"snap": list(now)}) + "\n")
        reason = failed(info)
        if reason:
            print(f"FAILED: {reason}", flush=True)
            return 1
        if all_complete(info):
            print("DONE: search queue extra-xgb complete", flush=True)
            return 0
        if prev is None:
            prev = now
            last_change = time.time()
        elif now != prev:
            prev_current, prev_stage, prev_complete = prev[1], prev[2], prev[6]
            queue, current, stage, seed, month, bucket, last_complete = now
            prev = now
            last_change = time.time()
            if last_complete and last_complete != prev_complete:
                print(f"ACTION_REQUIRED: job complete {last_complete} next={current}", flush=True)
            elif stage == "full" and prev_stage != "full":
                print(f"ACTION_REQUIRED: seed CSV {current} seed={seed}", flush=True)
            elif current and current != prev_current:
                print(f"ACTION_REQUIRED: job started {current}", flush=True)
        elif time.time() - last_change >= STALL_SEC:
            print(f"FAILED: stalled {int((time.time() - last_change) / 60)} min at {now}", flush=True)
            return 1
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    sys.exit(main())

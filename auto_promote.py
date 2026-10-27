"""Promote only genuinely better two-fold GPU configs to full submissions."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FOLLOW = ROOT / "results" / "followup.jsonl"
STATE = ROOT / "results" / "auto_promote_state.json"
THRESHOLD = 0.663713635

CONFIGS = {
    "cb_mask50_d8": ("cat", 0.50, "RMSE", 8),
    "cb_mask665_d8": ("cat", 0.665, "RMSE", 8),
    "cb_mask50_huber": ("cat", 0.50, "Huber:delta=1", 8),
    "cb_mask665_huber": ("cat", 0.665, "Huber:delta=1", 8),
    "xgb_mask50_pseudohuber": ("xgb", 0.50, "reg:pseudohubererror", 8),
    "xgb_mask665_pseudohuber": ("xgb", 0.665, "reg:pseudohubererror", 8),
    "cb_mask665_d6": ("cat", 0.665, "RMSE", 6),
    "cb_mask665_d10": ("cat", 0.665, "RMSE", 10),
    "xgb_mask665_sq_d8": ("xgb", 0.665, "reg:squarederror", 8),
}


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            pass
    # This candidate was already trained and synced before the promoter existed.
    return {"cb_mask665_d8:77": {"status": "done", "mean": 0.663713635}}


def save_state(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def latest_pairs() -> list[tuple[str, int, float, int]]:
    if not FOLLOW.exists():
        return []
    grouped: dict[tuple[str, int], dict[int, dict]] = {}
    for line in FOLLOW.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            if row.get("stage") != "cycle_result" or row.get("model") not in CONFIGS:
                continue
            key = (row["model"], int(row["seed"]))
            grouped.setdefault(key, {})[int(row["fold"])] = row
        except Exception:
            continue
    pairs = []
    for (model_name, seed), folds in grouped.items():
        if 0 not in folds or 1 not in folds:
            continue
        f0, f1 = folds[0], folds[1]
        mean = (float(f0["best_blend"]["all"]) + float(f1["best_blend"]["all"])) / 2.0
        iters = round((int(f0.get("best_iteration", 60)) + int(f1.get("best_iteration", 60))) / 2.0)
        pairs.append((model_name, seed, mean, max(30, min(300, iters))))
    return sorted(pairs, key=lambda item: item[2])


def promote(model_name: str, seed: int, mean: float, iterations: int, state: dict) -> None:
    kind, mask_rate, objective, depth = CONFIGS[model_name]
    key = f"{model_name}:{seed}"
    output_name = f"{model_name}_seed{seed}"
    log_path = ROOT / "logs" / f"auto_{output_name}.log"
    state[key] = {"status": "running", "mean": mean, "started": time.time()}
    save_state(state)
    command = [
        sys.executable,
        str(ROOT / "full_gpu_candidate.py"),
        "--mask-rate",
        str(mask_rate),
        "--kind",
        kind,
        "--objective",
        objective,
        "--seed",
        str(seed),
        "--iterations",
        str(iterations),
        "--depth",
        str(depth),
        "--name",
        output_name,
    ]
    try:
        # Full-data feature construction/model fit can briefly use nearly all
        # GPU memory. Pause the fold queue while promoting, then restore it.
        subprocess.run(["supervisorctl", "stop", "water_gpu_followup"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with log_path.open("a", encoding="utf-8") as handle:
            result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
        subprocess.run(["supervisorctl", "start", "water_gpu_followup"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        state[key] = {"status": "done" if result.returncode == 0 else "error", "mean": mean, "returncode": result.returncode}
    except Exception as exc:
        subprocess.run(["supervisorctl", "start", "water_gpu_followup"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        state[key] = {"status": "error", "mean": mean, "error": repr(exc)}
    save_state(state)


def main() -> int:
    state = load_state()
    save_state(state)
    while True:
        for model_name, seed, mean, iterations in latest_pairs():
            key = f"{model_name}:{seed}"
            if mean < THRESHOLD and key not in state:
                promote(model_name, seed, mean, iterations, state)
        time.sleep(20)


if __name__ == "__main__":
    raise SystemExit(main())

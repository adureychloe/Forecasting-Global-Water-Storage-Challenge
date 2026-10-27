"""Local masked-row routing toward neighbourhood persist, plus submission write."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from coordinate_compliance import drop_coordinate_features
from enhanced_features import build_features
from local_neighborhood_train import FIT_ROWS, fold_frame, make_lgbm, public_states
from neighborhood_tws_features import (
    last_observed_anchors,
    load_or_build_neighbors,
    route_masked_predictions,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SUB = ROOT / "submission"
SAMPLE = DATA / "SampleSubmission.csv"
LOG = RESULTS / "local_masked_route.jsonl"
LEADER = SUB / "submission_onlineleader_targetstats_blockfrozen_seed2027_b08_w100.csv"
WEIGHTS = (0.15, 0.25, 0.35, 0.50, 0.75, 1.00)
MIN_AGES = (0.0, 3.0, 7.0)


def log(row: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **row}
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
    print(json.dumps(payload, ensure_ascii=True), flush=True)


def rmse(y, pred) -> float:
    return float(np.sqrt(mean_squared_error(y, pred)))


def score(y, pred, masked) -> dict:
    return {
        "all": rmse(y, pred),
        "masked": rmse(y[masked], pred[masked]),
        "observed": rmse(y[~masked], pred[~masked]),
    }


def add_observation_age(frame: pd.DataFrame) -> pd.Series:
    work = frame[["lat", "lon", "time", "TWS_t"]].copy()
    work["time"] = pd.to_datetime(work["time"])
    work["ordinal"] = work["time"].dt.year * 12 + work["time"].dt.month
    observed = work["ordinal"].where(pd.to_numeric(work["TWS_t"], errors="coerce").notna())
    last = observed.groupby([work["lat"], work["lon"]], sort=False).ffill()
    return (work["ordinal"] - last).fillna(999.0)


def matrices(combined: pd.DataFrame, n_fit: int):
    features = drop_coordinate_features(
        build_features(combined).replace([np.inf, -np.inf], np.nan)
    ).fillna(0.0).astype(np.float32)
    return features.iloc[:n_fit], features.iloc[n_fit:]


def prepare_query(query: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    out = query.copy()
    out["time"] = pd.to_datetime(out["time"])
    stacked = pd.concat([history, out], ignore_index=True, sort=False)
    age = add_observation_age(stacked)
    out["TWS_t_age"] = age.iloc[len(history) :].to_numpy(dtype=np.float64)
    if "TWS_t_masked" not in out.columns:
        out["TWS_t_masked"] = pd.to_numeric(out["TWS_t"], errors="coerce").isna()
    return out


def write_submission(ids: pd.Series, pred: np.ndarray, path: Path) -> None:
    frame = pd.DataFrame({"ID": ids, "Target": np.asarray(pred, dtype=np.float64)})
    if len(frame) != len(ids):
        raise ValueError(f"row count {len(frame)}")
    if not np.isfinite(frame["Target"]).all():
        raise ValueError(f"non-finite: {path.name}")
    frame.to_csv(path, index=False)


def cross_validate(train: pd.DataFrame, test: pd.DataFrame, cells, neighbors) -> dict:
    states = public_states(test)
    model_scores = []
    route_rows = []
    oof_models = []
    for fold in (0, 1):
        fit, valid, combined, cutoff, max_date = fold_frame(train, states, fold)
        x_fit, x_valid = matrices(combined, len(fit))
        model = make_lgbm()
        model.fit(x_fit, fit["target"].to_numpy(dtype=np.float32))
        pred = np.asarray(model.predict(x_valid), dtype=np.float32)
        masked = combined["TWS_t_masked"].iloc[len(fit) :].to_numpy(dtype=bool)
        y = valid["target"].to_numpy(dtype=np.float32)
        base = score(y, pred, masked)
        model_scores.append(base)
        history = combined.iloc[: len(fit)]
        query = prepare_query(combined.iloc[len(fit) :].copy(), history)
        query.index = np.arange(len(query))
        anchors = last_observed_anchors(history, cells)
        log({"stage": "fold_model", "fold": fold, "cutoff": str(cutoff.date()), "metrics": base})
        for recursive in (False, True):
            for min_age in MIN_AGES:
                for weight in WEIGHTS:
                    routed = route_masked_predictions(
                        query,
                        pred,
                        cells,
                        neighbors,
                        anchors,
                        weight=weight,
                        recursive=recursive,
                        min_age=min_age,
                    )
                    metrics = score(y, routed, masked)
                    row = {
                        "fold": fold,
                        "recursive": recursive,
                        "min_age": min_age,
                        "weight": weight,
                        "metrics": metrics,
                    }
                    route_rows.append(row)
                    log({"stage": "fold_route", **row})
        oof_models.append(model)
    summary = []
    keys = {(r["recursive"], r["min_age"], r["weight"]) for r in route_rows}
    for recursive, min_age, weight in sorted(keys, key=lambda x: (x[0], x[1], x[2])):
        pair = [
            r["metrics"]
            for r in route_rows
            if r["recursive"] == recursive and r["min_age"] == min_age and r["weight"] == weight
        ]
        mean_all = float(np.mean([m["all"] for m in pair]))
        mean_masked = float(np.mean([m["masked"] for m in pair]))
        summary.append(
            {
                "recursive": recursive,
                "min_age": min_age,
                "weight": weight,
                "mean_all": mean_all,
                "mean_masked": mean_masked,
                "fold0": pair[0],
                "fold1": pair[1],
            }
        )
    summary.sort(key=lambda r: r["mean_all"])
    control_mean = float(np.mean([m["all"] for m in model_scores]))
    best = summary[0]
    log({"stage": "cv_control_mean", "mean_all": control_mean, "folds": model_scores})
    log({"stage": "cv_best_route", **{k: best[k] for k in ("recursive", "min_age", "weight", "mean_all", "mean_masked")}})
    return {"control_mean": control_mean, "best": best, "summary": summary[:12]}


def apply_to_leader(train: pd.DataFrame, test: pd.DataFrame, cells, neighbors, settings: list[dict]) -> list[str]:
    sample = pd.read_csv(SAMPLE)
    leader = pd.read_csv(LEADER)
    if not sample["ID"].equals(leader.iloc[:, 0]) or not sample["ID"].equals(test["ID"]):
        raise ValueError("ID order mismatch with SampleSubmission")
    pred = leader.iloc[:, 1].to_numpy(dtype=np.float64)
    query = prepare_query(test.copy(), train)
    anchors = last_observed_anchors(train, cells)
    made = []
    for spec in settings:
        routed = route_masked_predictions(
            query,
            pred,
            cells,
            neighbors,
            anchors,
            weight=spec["weight"],
            recursive=spec["recursive"],
            min_age=spec["min_age"],
        )
        tag = "rec" if spec["recursive"] else "stat"
        age = f"a{int(spec['min_age'])}"
        path = SUB / (
            f"submission_onlineleader_w100_nb{tag}_{age}_w{int(spec['weight'] * 100):02d}.csv"
        )
        write_submission(sample["ID"], routed, path)
        delta = float(np.sqrt(np.mean((routed - pred) ** 2)))
        masked = query["TWS_t_masked"].to_numpy(dtype=bool)
        made.append(path.name)
        log(
            {
                "stage": "wrote",
                "file": path.name,
                "weight": spec["weight"],
                "recursive": spec["recursive"],
                "min_age": spec["min_age"],
                "delta_rmse": delta,
                "masked_changed": int(masked.sum()),
                "rows": int(len(routed)),
            }
        )
    return made


def fit_local_and_write(train: pd.DataFrame, test: pd.DataFrame, cells, neighbors, best: dict) -> str:
    sample = pd.read_csv(SAMPLE)
    fit = train.tail(FIT_ROWS).copy()
    combined = pd.concat(
        [fit.drop(columns=["target"]), test.drop(columns=["TWS_t_masked"], errors="ignore")],
        ignore_index=True,
        sort=False,
    )
    if "TWS_t_masked" in test.columns:
        combined.loc[len(fit) :, "TWS_t"] = test["TWS_t"].to_numpy()
    x_fit, x_test = matrices(combined, len(fit))
    model = make_lgbm()
    model.fit(x_fit, fit["target"].to_numpy(dtype=np.float32))
    pred = np.asarray(model.predict(x_test), dtype=np.float32)
    query = prepare_query(test.copy(), train)
    anchors = last_observed_anchors(train, cells)
    routed = route_masked_predictions(
        query,
        pred,
        cells,
        neighbors,
        anchors,
        weight=best["weight"],
        recursive=best["recursive"],
        min_age=best["min_age"],
    )
    raw_path = SUB / "submission_local_lgbm_nocoord_raw.csv"
    routed_path = SUB / (
        f"submission_local_lgbm_nocoord_nb{'rec' if best['recursive'] else 'stat'}"
        f"_a{int(best['min_age'])}_w{int(best['weight'] * 100):02d}.csv"
    )
    write_submission(sample["ID"], pred, raw_path)
    write_submission(sample["ID"], routed, routed_path)
    log({"stage": "wrote_local", "raw": raw_path.name, "routed": routed_path.name})
    return routed_path.name


def main() -> int:
    train = pd.read_csv(DATA / "Train.csv")
    test = pd.read_csv(DATA / "Test.csv")
    train["time"] = pd.to_datetime(train["time"])
    test["time"] = pd.to_datetime(test["time"])
    cells, neighbors = load_or_build_neighbors(pd.concat([train[["lat", "lon"]], test[["lat", "lon"]]], ignore_index=True))
    log({"stage": "start", "n_cells": int(len(cells)), "leader": LEADER.name})
    cv = cross_validate(train, test, cells, neighbors)
    best = cv["best"]
    settings = [
        {"recursive": best["recursive"], "min_age": best["min_age"], "weight": best["weight"]},
        {"recursive": False, "min_age": 0.0, "weight": 0.25},
        {"recursive": False, "min_age": 0.0, "weight": 0.50},
        {"recursive": True, "min_age": 0.0, "weight": 0.25},
        {"recursive": True, "min_age": 0.0, "weight": 0.50},
        {"recursive": False, "min_age": 7.0, "weight": 0.50},
        {"recursive": True, "min_age": 7.0, "weight": 0.50},
    ]
    # unique settings
    uniq = []
    seen = set()
    for spec in settings:
        key = (spec["recursive"], spec["min_age"], spec["weight"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(spec)
    made = apply_to_leader(train, test, cells, neighbors, uniq)
    local_name = fit_local_and_write(train, test, cells, neighbors, best)
    log({"stage": "done", "leader_files": made, "local": local_name, "cv_best": best})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

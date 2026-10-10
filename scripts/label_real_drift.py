# -*- coding: utf-8 -*-
"""Annotate real time series using five data-only detectors and majority voting."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

# Keep PCA/BLAS from spawning many threads for every streaming observation.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from onlinetsf.drift import ADWINDetector, CapyMOADetector, HDDMWDetector, KSWINDetector


METHODS = ("seed", "adwin", "abcd", "kswin", "hddmw")
DATASETS = {"ETTh2": "ETTh2.csv", "ETTm1": "ETTm1.csv", "Electricity": "electricity.csv"}


def write_csv(path: Path, columns: list[str], rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: Path, time_column: str, max_rows: int | None):
    """Read unchanged source text and a finite numeric matrix with regular timestamps."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        if columns is None or time_column not in columns:
            raise ValueError(f"{path}: missing time column {time_column!r}")
        if len(columns) != len(set(columns)):
            raise ValueError(f"{path}: duplicate column names")
        features = [column for column in columns if column != time_column]
        rows = []
        for row in reader:
            rows.append(row)
            if max_rows is not None and len(rows) == max_rows:
                break
    if len(rows) < 2 or len(features) < 2:
        raise ValueError(f"{path}: at least two rows and two numeric features are required")
    values = np.asarray([[float(row[name]) for name in features] for row in rows], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: numeric data contains missing or non-finite values")
    dates = [datetime.fromisoformat(row[time_column]) for row in rows]
    deltas = np.asarray([(right - left).total_seconds() for left, right in zip(dates, dates[1:])])
    if deltas[0] <= 0 or not np.all(deltas == deltas[0]):
        raise ValueError(f"{path}: timestamps must be strictly increasing and regularly sampled")
    return columns, features, rows, values, float(deltas[0])


def detect_features(name: str, standardized: np.ndarray, parameters: dict, calibration_rows: int,
                    workers: int = 1, progress: bool = True):
    """Run independent scalar streams, or one ABCD on the full observation vector."""
    began = perf_counter()
    if name != "abcd" and workers > 1:
        chunks = np.array_split(standardized, min(workers, standardized.shape[1]), axis=1)
        with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
            futures = [pool.submit(detect_features, name, chunk, parameters, calibration_rows, 1, False)
                       for chunk in chunks]
            results = []
            for index, future in enumerate(futures, 1):
                results.append(future.result()[0])
                print(f"  {name}: {index}/{len(chunks)} feature groups", flush=True)
        return np.concatenate(results, axis=1), perf_counter() - began
    bounded = 0.5 * (1.0 + np.tanh(0.5 * standardized))
    width = 1 if name == "abcd" else standardized.shape[1]
    alarms = np.zeros((len(standardized), width), dtype=bool)
    if name == "abcd":
        detector = CapyMOADetector(name, **parameters)
        for index, vector in enumerate(bounded):
            detected = detector.update(vector).detected
            alarms[index, 0] = detected and index >= calibration_rows
            if progress and (index + 1) % 5000 == 0:
                print(f"  {name}: {index + 1}/{len(standardized)} rows", flush=True)
    else:
        for feature in range(width):
            if name == "seed":
                detector = CapyMOADetector(name, **parameters)
            elif name == "adwin":
                detector = ADWINDetector(**parameters)
            elif name == "kswin":
                detector = KSWINDetector(**parameters)
            else:
                detector = HDDMWDetector(source="features", **parameters)
            # HDDMW's feature adapter applies the same sigmoid itself.
            signal = standardized[:, feature] if name == "hddmw" else bounded[:, feature]
            for index, value in enumerate(signal):
                detected = detector.update(float(value)).detected
                alarms[index, feature] = detected and index >= calibration_rows
            if progress and ((feature + 1) % max(1, width // 10) == 0 or feature + 1 == width):
                print(f"  {name}: {feature + 1}/{width} features", flush=True)
    return alarms, perf_counter() - began


def majority_vote(alarms: dict[str, np.ndarray], window: int, min_feature_fraction: float):
    """One vote per method, using distinct affected features in a trailing window."""
    count = len(next(iter(alarms.values())))
    indices = np.arange(count)
    active_features = {}
    votes = {}
    for name in METHODS:
        matrix = alarms[name]
        active = np.zeros(count, dtype=np.int32)
        for feature in range(matrix.shape[1]):
            last = np.maximum.accumulate(np.where(matrix[:, feature], indices, -1))
            active += (last >= 0) & (indices - last < window)
        threshold = 1 if name == "abcd" else math.ceil(matrix.shape[1] * min_feature_fraction)
        active_features[name] = active
        votes[name] = active >= threshold
    vote_count = np.sum([votes[name] for name in METHODS], axis=0).astype(np.int32)
    return vote_count, votes, active_features


def label_intervals(consensus: np.ndarray, calibration_rows: int, merge_gap: int, transition: int):
    """Merge consensus episodes, expand transition shoulders, and partition all rows."""
    points = np.flatnonzero(consensus)
    episodes = []
    for point in points:
        point = int(point)
        if episodes and point - episodes[-1][1] <= merge_gap:
            episodes[-1][1] = point + 1
        else:
            episodes.append([point, point + 1])
    phases = np.full(len(consensus), "stable", dtype="<U11")
    event_ids = np.zeros(len(consensus), dtype=np.int32)
    for start, end in episodes:
        phases[max(calibration_rows, start - transition):min(len(phases), end + transition)] = "transition"
    for event_id, (start, end) in enumerate(episodes, 1):
        phases[start:end] = "drift"
        event_ids[start:end] = event_id
    phases[:calibration_rows] = "calibration"
    boundaries = np.r_[0, np.flatnonzero(phases[1:] != phases[:-1]) + 1, len(phases)]
    interval_ids = np.zeros(len(consensus), dtype=np.int32)
    intervals = []
    for interval_id, (start, end) in enumerate(zip(boundaries, boundaries[1:]), 1):
        start, end = int(start), int(end)
        interval_ids[start:end] = interval_id
        intervals.append({"interval_id": interval_id, "phase": str(phases[start]),
                          "start_index": start, "end_index_exclusive": end, "rows": end - start})
    return phases, event_ids, interval_ids, intervals, episodes


def annotate_dataset(dataset: str, source: Path, output: Path, profiles: dict, args):
    began = perf_counter()
    columns, features, rows, values, period_seconds = load_csv(source, args.time_column, args.max_rows)
    calibration_rows = math.ceil(args.calibration_hours * 3600 / period_seconds)
    if not 2 <= calibration_rows < len(rows):
        raise ValueError(f"{dataset}: calibration requires {calibration_rows} rows, but input has {len(rows)}")
    output.mkdir(parents=True, exist_ok=False)
    mean = values[:calibration_rows].mean(axis=0)
    std = values[:calibration_rows].std(axis=0)
    constant_features = [features[index] for index in np.flatnonzero(std == 0)]
    scale = np.where(std == 0, 1.0, std)
    standardized = (values - mean) / scale
    parameters, alarms, timings = {}, {}, {}
    print(f"{dataset}: {len(rows)} rows, {len(features)} features; calibration={calibration_rows}", flush=True)
    for name in METHODS:
        parameters[name] = dict(profiles["detectors"][name].get("parameters", {}))
        if name == "kswin":
            parameters[name]["seed"] = args.seed
        elif name == "hddmw":
            parameters[name]["two_sided_test"] = True
        elif name == "abcd":
            parameters[name].setdefault("max_size", 2048)
            if parameters[name].get("model_id", "pca") != "pca":
                raise ValueError("This data-only annotation script requires ABCD model_id: pca")
        np.random.seed(args.seed)
        alarms[name], timings[name] = detect_features(name, standardized, parameters[name], calibration_rows, args.workers)
        print(f"  {name}: completed in {timings[name]:.2f}s", flush=True)
    window = max(1, math.ceil(args.vote_window_hours * 3600 / period_seconds))
    merge_gap = args.merge_gap_samples
    transition = args.transition_samples
    vote_count, votes, active_features = majority_vote(alarms, window, args.min_feature_fraction)
    consensus = vote_count >= 3
    phases, event_ids, interval_ids, intervals, episodes = label_intervals(
        consensus, calibration_rows, merge_gap, transition,
    )
    point_mask = np.zeros(len(rows), dtype=bool)
    point_rows = []
    for event_id, (start, end) in enumerate(episodes, 1):
        point_mask[start] = True
        point_rows.append({"event_id": event_id, "detection_index": start,
                           "detection_time": rows[start][args.time_column],
                           "start_index": start, "end_index_exclusive": end,
                           "last_alarm_time": rows[end - 1][args.time_column],
                           "consensus_rows": int(consensus[start:end].sum()),
                           "peak_votes": int(vote_count[start:end].max()),
                           "transition_start_index": max(calibration_rows, start - transition),
                           "transition_end_index_exclusive": min(len(rows), end + transition)})
    write_csv(output / "drift_points.csv", ["event_id", "detection_index", "detection_time", "start_index",
              "end_index_exclusive", "last_alarm_time", "consensus_rows", "peak_votes",
              "transition_start_index", "transition_end_index_exclusive"], point_rows)
    for interval in intervals:
        interval["start_time"] = rows[interval["start_index"]][args.time_column]
        interval["last_time"] = rows[interval["end_index_exclusive"] - 1][args.time_column]
    write_csv(output / "intervals.csv", ["interval_id", "phase", "start_index", "end_index_exclusive",
              "rows", "start_time", "last_time"], intervals)
    annotation_columns = ["annotation_index", "annotation_phase", "annotation_interval_id", "drift_event_id",
                          "is_drift_point", "is_consensus_alarm", "majority_vote_count"]
    for name in METHODS:
        annotation_columns.extend([f"{name}_alarm", f"{name}_vote", f"{name}_active_features"])
    if set(columns).intersection(annotation_columns):
        raise ValueError("Source column names collide with annotation columns")
    raw_flags = {name: alarms[name].any(axis=1) for name in METHODS}
    with (output / "labeled_data.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns + annotation_columns)
        writer.writeheader()
        for index, row in enumerate(rows):
            annotated = {**row, "annotation_index": index, "annotation_phase": phases[index],
                         "annotation_interval_id": int(interval_ids[index]), "drift_event_id": int(event_ids[index]),
                         "is_drift_point": int(point_mask[index]), "is_consensus_alarm": int(consensus[index]),
                         "majority_vote_count": int(vote_count[index])}
            for name in METHODS:
                annotated[f"{name}_alarm"] = int(raw_flags[name][index])
                annotated[f"{name}_vote"] = int(votes[name][index])
                annotated[f"{name}_active_features"] = int(active_features[name][index])
            writer.writerow(annotated)
    evaluated = len(rows) - calibration_rows
    summaries, feature_stats = [], []
    with (output / "alarm_events.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "variable", "index", "time"])
        writer.writeheader()
        for name in METHODS:
            matrix = alarms[name]
            variables = ["all_features"] if name == "abcd" else features
            for index, feature in zip(*np.nonzero(matrix)):
                writer.writerow({"method": name, "variable": variables[feature], "index": int(index),
                                 "time": rows[index][args.time_column]})
            counts = matrix.sum(axis=0)
            for variable, count in zip(variables, counts):
                feature_stats.append({"dataset": dataset, "method": name, "variable": variable,
                                      "evaluated_rows": evaluated, "alarm_count": int(count),
                                      "alarm_rate": float(count / evaluated),
                                      "alarms_per_1000_rows": float(1000 * count / evaluated)})
            count = int(raw_flags[name].sum())
            summaries.append({"dataset": dataset, "method": name, "total_rows": len(rows),
                              "calibration_rows": calibration_rows, "evaluated_rows": evaluated,
                              "detector_instances": matrix.shape[1], "alarm_count": count,
                              "alarm_rate": count / evaluated, "alarms_per_1000_rows": 1000 * count / evaluated,
                              "variable_alarm_count": int(matrix.sum()),
                              "variable_alarm_rate": float(matrix.sum() / (evaluated * matrix.shape[1])),
                              "vote_rows": int(votes[name].sum()), "vote_rate": float(votes[name].sum() / evaluated),
                              "seconds": timings[name]})
    consensus_count = int(consensus.sum())
    summaries.append({"dataset": dataset, "method": "majority", "total_rows": len(rows),
                      "calibration_rows": calibration_rows, "evaluated_rows": evaluated,
                      "detector_instances": 5, "alarm_count": consensus_count,
                      "alarm_rate": consensus_count / evaluated,
                      "alarms_per_1000_rows": 1000 * consensus_count / evaluated,
                      "variable_alarm_count": "", "variable_alarm_rate": "",
                      "vote_rows": consensus_count, "vote_rate": consensus_count / evaluated,
                      "seconds": sum(timings.values())})
    write_csv(output / "alarm_summary.csv", list(summaries[0]), summaries)
    write_csv(output / "alarm_summary_by_variable.csv", list(feature_stats[0]), feature_stats)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        chunk = handle.read(1024 * 1024)
        while chunk:
            digest.update(chunk)
            chunk = handle.read(1024 * 1024)
    source_hash = digest.hexdigest()
    manifest = {
        "dataset": dataset, "source": str(source.resolve()), "source_sha256": source_hash,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_kind": "real_observed_time_series",
        "label_kind": "detector_consensus_pseudo_labels_not_ground_truth",
        "source_input": "numeric_observations_only_no_forecasts_or_forecast_residuals",
        "feature_columns": features, "time_column": args.time_column, "rows": len(rows),
        "max_rows": args.max_rows, "sample_period_seconds": period_seconds,
        "calibration_rows": calibration_rows, "evaluated_rows": evaluated,
        "normalization": {"formula": "sigmoid((value-prefix_mean)/prefix_std)",
                          "mean": mean.tolist(), "scale": scale.tolist(),
                          "constant_prefix_features": constant_features, "constant_prefix_scale": 1.0},
        "seed": args.seed, "workers": args.workers, "parameters": parameters,
        "versions": {name: version(name) for name in ("river", "capymoa")},
        "voting": {"required_methods": 3, "total_methods": 5, "window_samples": window,
                   "window_hours": args.vote_window_hours, "window_type": "trailing_including_current_row",
                   "min_feature_fraction": args.min_feature_fraction,
                   "scalar_min_features": math.ceil(len(features) * args.min_feature_fraction),
                   "scalar_vote": "distinct variables with an alarm inside the window; each method has one vote",
                   "abcd_vote": "at least one full-vector alarm inside the window"},
        "interval_rule": {"indices": "zero_based_half_open", "merge_gap_samples": merge_gap,
                          "transition_padding_samples": transition, "label_available": "offline_postprocessing",
                          "stable_meaning": "outside_consensus_episodes_and_transition_shoulders_not_verified_stationary",
                          "drift_meaning": "merged_consensus_alarm_episode_not_true_change_onset_or_duration"},
        "drift_event_count": len(episodes), "consensus_alarm_rows": consensus_count,
        "phase_rows": {phase: int((phases == phase).sum())
                       for phase in ("calibration", "stable", "transition", "drift")},
        "alarm_rate_denominator": "post_calibration_rows_including_detector_internal_warmup",
        "seconds": perf_counter() - began,
    }
    with (output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"{dataset}: {len(episodes)} events; phase rows={manifest['phase_rows']}", flush=True)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--profiles", type=Path, default=PROJECT_ROOT / "profiles.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--time-column", default="date")
    parser.add_argument("--calibration-hours", type=float, default=168.0,
                        help="initial data used to fit fixed normalization; excluded from labels and alarm rates")
    parser.add_argument("--vote-window-hours", type=float, default=6.0,
                        help="trailing voting tolerance; 0 means strict same-row voting")
    parser.add_argument("--min-feature-fraction", type=float, default=0.1,
                        help="minimum distinct-feature fraction for a scalar method's one vote")
    parser.add_argument("--merge-gap-samples", type=int, required=True,
                        help="maximum number of non-consensus rows merged inside a consensus episode")
    parser.add_argument("--transition-samples", type=int, required=True,
                        help="number of transition rows on each side of a merged episode")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel independent feature groups; ABCD always consumes the full vector")
    parser.add_argument("--max-rows", type=int, help="preview only the first N rows; recorded in manifest")
    parser.add_argument("--java-home", type=Path, help="JDK used by CapyMOA's SEED")
    args = parser.parse_args()
    if args.java_home is not None:
        os.environ["JAVA_HOME"] = str(args.java_home.resolve())
    if not math.isfinite(args.calibration_hours) or args.calibration_hours <= 0:
        parser.error("--calibration-hours must be finite and positive")
    if not math.isfinite(args.vote_window_hours) or args.vote_window_hours < 0:
        parser.error("--vote-window-hours must be finite and nonnegative")
    if args.merge_gap_samples < 0 or args.transition_samples < 0:
        parser.error("--merge-gap-samples and --transition-samples must be nonnegative integers")
    if not 0 < args.min_feature_fraction <= 1:
        parser.error("--min-feature-fraction must be in (0, 1]")
    if args.max_rows is not None and args.max_rows < 2:
        parser.error("--max-rows must be at least 2")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if not 0 <= args.seed < 2**32 or len(set(args.datasets)) != len(args.datasets):
        parser.error("seed must be in [0, 2**32); datasets must not contain duplicates")
    for dataset in args.datasets:
        if not (args.data_dir / DATASETS[dataset]).is_file():
            parser.error(f"Missing dataset: {args.data_dir / DATASETS[dataset]}")
    with args.profiles.open("r", encoding="utf-8") as handle:
        profiles = yaml.safe_load(handle)
    # Check dependencies before starting long scalar-detector runs or writing output.
    CapyMOADetector("seed", **profiles["detectors"]["seed"].get("parameters", {}))
    CapyMOADetector("abcd", **profiles["detectors"]["abcd"].get("parameters", {}))
    output = args.output or PROJECT_ROOT / "runs" / datetime.now(timezone.utc).strftime("real-drift-labels-%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for dataset in args.datasets:
        summaries.extend(annotate_dataset(dataset, args.data_dir / DATASETS[dataset], output / dataset, profiles, args))
        write_csv(output / "alarm_summary.csv", list(summaries[0]), summaries)
    print(f"Outputs: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Step 1: select stable intervals from an existing benchmark time series."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import yaml

# Allow direct execution from a source checkout as well as an editable install.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from onlinetsf.__main__ import collect_drift_records, run_forecast
from onlinetsf.config import load_config
from onlinetsf.data import load_benchmark_dataset
from onlinetsf.online import FeedbackEvent, OnlineRun
from scripts.stability import (
    CandidateWindow,
    DistributionMetrics,
    LossMetrics,
    condition_counts,
    covered_raw_length,
    distribution_metrics,
    loss_metrics,
    merge_stable_windows,
    quantile_threshold,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select stable source intervals using detector, distribution, and loss criteria"
    )
    parser.add_argument("--config", default="config.yaml", help="base experiment configuration")
    parser.add_argument(
        "--strategies",
        nargs="+",
        help="non-linear strategy profiles; defaults to every configured non-linear strategy",
    )
    parser.add_argument(
        "--detectors",
        nargs="+",
        default=["page_hinkley", "adwin", "kswin"],
        help="detector profiles used for both feature and residual inputs",
    )
    parser.add_argument("--window-size", type=int, default=256, help="forecast samples per candidate window")
    parser.add_argument("--mean-threshold", type=float, default=0.5)
    parser.add_argument("--std-threshold", type=float, default=0.5)
    parser.add_argument("--quantile-threshold", type=float, default=0.75)
    parser.add_argument("--distribution-feature-quantile", type=float, default=0.95)
    parser.add_argument(
        "--loss-low-quantile",
        type=float,
        default=0.5,
        help="quantile of candidate mean losses defining a low loss for each model",
    )
    parser.add_argument("--loss-cv-threshold", type=float, default=1.0)
    parser.add_argument(
        "--extra-min-length",
        type=int,
        help="extra raw rows beyond context+horizon; defaults to window-size",
    )
    parser.add_argument("--device", help="override online.device for every model")
    parser.add_argument("--output", help="output directory; defaults to runs/stability-<dataset>-<timestamp>")
    return parser.parse_args(argv)


def configured_non_linear_strategies(config_path: str | Path) -> list[str]:
    source = Path(config_path)
    with source.open("r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle)
    profile_path = Path(base["profiles"])
    if not profile_path.is_absolute():
        profile_path = source.parent / profile_path
    with profile_path.open("r", encoding="utf-8") as handle:
        profiles = yaml.safe_load(handle)
    return [
        name
        for name, profile in profiles["strategies"].items()
        if profile["forecasting"]["backbone"] != "linear"
    ]


def validate_args(args: argparse.Namespace, strategies: Sequence[str]) -> None:
    if args.window_size < 2:
        raise ValueError("--window-size must be at least 2")
    if args.extra_min_length is not None and args.extra_min_length < 0:
        raise ValueError("--extra-min-length must be non-negative")
    if any(value < 0.0 for value in (args.mean_threshold, args.std_threshold, args.quantile_threshold)):
        raise ValueError("distribution thresholds must be non-negative")
    if not 0.0 < args.distribution_feature_quantile <= 1.0:
        raise ValueError("--distribution-feature-quantile must be in (0, 1]")
    if not 0.0 < args.loss_low_quantile <= 1.0:
        raise ValueError("--loss-low-quantile must be in (0, 1]")
    if args.loss_cv_threshold < 0.0:
        raise ValueError("--loss-cv-threshold must be non-negative")
    if not strategies:
        raise ValueError("at least one non-linear strategy is required")


def prepare_config(
    path: str | Path,
    strategy: str,
    detector: str,
    device: str | None,
) -> dict[str, Any]:
    config = load_config(path, strategy_name=strategy, detector_name=detector)
    if config["forecasting"]["backbone"] == "linear":
        raise ValueError("linear backbones are intentionally excluded from stability selection")
    if device is not None:
        config["online"]["device"] = device
    config["online"]["keep_predictions"] = True
    config["output"]["write_per_value_errors"] = False
    return config


def aggregate_event_losses(events: Sequence[FeedbackEvent]) -> dict[int, float]:
    squared_error_sums: dict[int, float] = defaultdict(float)
    observed_counts: dict[int, int] = defaultdict(int)
    for event in events:
        squared_error_sums[event.index] += event.mse * event.observed_values
        observed_counts[event.index] += event.observed_values
    return {
        index: squared_error_sums[index] / observed_counts[index]
        for index in squared_error_sums
    }


def collect_alarm_indices(
    config_path: str | Path,
    detector_names: Sequence[str],
    strategy: str,
    run: OnlineRun,
    source: str,
    device: str | None,
) -> tuple[set[int], dict[str, int]]:
    alarm_indices: set[int] = set()
    alarm_counts: dict[str, int] = {}
    for detector_name in detector_names:
        config = prepare_config(config_path, strategy, detector_name, device)
        config["drift"]["source"] = source
        records = collect_drift_records(config, run)
        detected = [record for record in records if record.detected]
        alarm_indices.update(record.index for record in detected)
        alarm_counts[detector_name] = len(detected)
    return alarm_indices, alarm_counts


def raw_bounds(sample_start: int, sample_end: int, stride: int, context: int, horizon: int) -> tuple[int, int]:
    raw_start = sample_start * stride
    raw_end = (sample_end - 1) * stride + context + horizon
    return raw_start, raw_end


def read_timestamps(path: str | Path, time_column: str) -> list[str]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or time_column not in reader.fieldnames:
            raise ValueError(f"time column {time_column!r} not found in {path}")
        return [row[time_column] for row in reader]


def output_directory(args: argparse.Namespace, dataset_name: str) -> Path:
    if args.output:
        return Path(args.output)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return Path("runs") / f"stability-{dataset_name}-{timestamp}"


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    strategies = args.strategies or configured_non_linear_strategies(args.config)
    validate_args(args, strategies)

    first_config = prepare_config(args.config, strategies[0], args.detectors[0], args.device)
    data_config = first_config["data"]
    dataset = load_benchmark_dataset(
        data_config["name"],
        data_config["path"],
        context_length=data_config["context_length"],
        horizon=data_config["horizon"],
        stride=data_config.get("stride", 1),
    )
    context = dataset.context_length
    horizon = dataset.horizon
    stride = dataset.stride
    train_size = int(len(dataset) * first_config["offline"]["train_ratio"])
    online_start = max(train_size, first_config["online"].get("start", 0))
    online_stop = first_config["online"].get("stop") or len(dataset)

    # Do not judge any raw row that appeared in an offline training context or target.
    if train_size:
        offline_raw_end = (train_size - 1) * stride + context + horizon
        eligible_start = max(online_start, math.ceil(offline_raw_end / stride))
    else:
        offline_raw_end = 0
        eligible_start = online_start

    candidate_ranges = [
        (start, start + args.window_size)
        for start in range(eligible_start, online_stop - args.window_size + 1, args.window_size)
    ]
    if not candidate_ranges:
        raise ValueError("no complete candidate window remains after excluding offline-training data")

    runs: dict[str, OnlineRun] = {}
    losses: dict[str, dict[int, float]] = {}
    residual_alarms: set[int] = set()
    alarm_counts: dict[str, dict[str, int]] = {}
    for strategy in strategies:
        print(f"running strategy={strategy}", flush=True)
        config = prepare_config(args.config, strategy, args.detectors[0], args.device)
        run = run_forecast(config)
        runs[strategy] = run
        losses[strategy] = aggregate_event_losses(run.events)
        alarms, counts = collect_alarm_indices(
            args.config, args.detectors, strategy, run, "residual", args.device
        )
        residual_alarms.update(alarms)
        alarm_counts[f"residual:{strategy}"] = counts

    feature_alarms, counts = collect_alarm_indices(
        args.config,
        args.detectors,
        strategies[0],
        runs[strategies[0]],
        "features",
        args.device,
    )
    alarm_counts["features"] = counts
    all_alarm_indices = feature_alarms | residual_alarms

    distribution_by_range: dict[tuple[int, int], DistributionMetrics] = {}
    loss_by_range: dict[tuple[int, int], dict[str, LossMetrics]] = {}
    for sample_start, sample_end in candidate_ranges:
        raw_start, raw_end = raw_bounds(sample_start, sample_end, stride, context, horizon)
        distribution_by_range[(sample_start, sample_end)] = distribution_metrics(
            dataset.values[raw_start:raw_end],
            feature_quantile=args.distribution_feature_quantile,
        )
        per_model: dict[str, LossMetrics] = {}
        for strategy in strategies:
            window_losses = [losses[strategy][index] for index in range(sample_start, sample_end)]
            per_model[strategy] = loss_metrics(window_losses)
        loss_by_range[(sample_start, sample_end)] = per_model

    loss_thresholds = {
        strategy: quantile_threshold(
            [loss_by_range[candidate][strategy].mean for candidate in candidate_ranges],
            args.loss_low_quantile,
        )
        for strategy in strategies
    }

    windows: list[CandidateWindow] = []
    for sample_start, sample_end in candidate_ranges:
        raw_start, raw_end = raw_bounds(sample_start, sample_end, stride, context, horizon)
        distribution = distribution_by_range[(sample_start, sample_end)]
        distribution_ok = (
            distribution.mean_change <= args.mean_threshold
            and distribution.std_change <= args.std_threshold
            and distribution.quantile_change <= args.quantile_threshold
        )
        per_model = loss_by_range[(sample_start, sample_end)]
        loss_ok = all(
            per_model[strategy].mean <= loss_thresholds[strategy]
            and per_model[strategy].coefficient_of_variation <= args.loss_cv_threshold
            for strategy in strategies
        )
        detector_ok = not any(sample_start <= index < sample_end for index in all_alarm_indices)
        windows.append(
            CandidateWindow(
                sample_start=sample_start,
                sample_end=sample_end,
                raw_start=raw_start,
                raw_end=raw_end,
                detector_ok=detector_ok,
                distribution_ok=distribution_ok,
                loss_ok=loss_ok,
            )
        )

    extra_min_length = args.extra_min_length if args.extra_min_length is not None else args.window_size
    minimum_raw_length = context + horizon + extra_min_length
    intervals = merge_stable_windows(windows, minimum_raw_length=minimum_raw_length)
    analyzed_raw_start = windows[0].raw_start
    analyzed_raw_end = windows[-1].raw_end
    retained_rows = covered_raw_length(intervals)
    analyzed_rows = analyzed_raw_end - analyzed_raw_start
    timestamps = read_timestamps(data_config["path"], "date")

    destination = output_directory(args, data_config["name"])
    destination.mkdir(parents=True, exist_ok=False)
    window_fields = [
        "sample_start", "sample_end", "raw_start", "raw_end",
        "feature_detector_ok", "residual_detector_ok", "detector_ok",
        "feature_alarm_indices", "residual_alarm_indices",
        "distribution_ok", "loss_ok", "stable",
        "mean_change", "std_change", "quantile_change",
    ]
    for strategy in strategies:
        window_fields.extend((f"{strategy}_loss_mean", f"{strategy}_loss_cv"))
    with (destination / "candidate_windows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=window_fields)
        writer.writeheader()
        for window in windows:
            candidate = (window.sample_start, window.sample_end)
            distribution = distribution_by_range[candidate]
            row: dict[str, Any] = {
                **asdict(window),
                "feature_detector_ok": not any(
                    window.sample_start <= index < window.sample_end
                    for index in feature_alarms
                ),
                "residual_detector_ok": not any(
                    window.sample_start <= index < window.sample_end
                    for index in residual_alarms
                ),
                "feature_alarm_indices": sum(
                    window.sample_start <= index < window.sample_end
                    for index in feature_alarms
                ),
                "residual_alarm_indices": sum(
                    window.sample_start <= index < window.sample_end
                    for index in residual_alarms
                ),
                "stable": window.stable,
                "mean_change": distribution.mean_change,
                "std_change": distribution.std_change,
                "quantile_change": distribution.quantile_change,
            }
            for strategy in strategies:
                metrics = loss_by_range[candidate][strategy]
                row[f"{strategy}_loss_mean"] = metrics.mean
                row[f"{strategy}_loss_cv"] = metrics.coefficient_of_variation
            writer.writerow(row)

    interval_fields = [
        "interval_id", "sample_start", "sample_end", "raw_start", "raw_end_exclusive",
        "raw_length", "start_time", "end_time", "window_count",
    ]
    with (destination / "stable_intervals.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=interval_fields)
        writer.writeheader()
        for interval_id, interval in enumerate(intervals):
            writer.writerow(
                {
                    "interval_id": interval_id,
                    "sample_start": interval.sample_start,
                    "sample_end": interval.sample_end,
                    "raw_start": interval.raw_start,
                    "raw_end_exclusive": interval.raw_end,
                    "raw_length": interval.raw_length,
                    "start_time": timestamps[interval.raw_start],
                    "end_time": timestamps[interval.raw_end - 1],
                    "window_count": interval.window_count,
                }
            )

    summary = {
        "dataset": data_config["name"],
        "source_path": str(data_config["path"]),
        "strategies": list(strategies),
        "detectors": list(args.detectors),
        "offline_train_ratio": first_config["offline"]["train_ratio"],
        "offline_training_windows": train_size,
        "offline_raw_end_exclusive": offline_raw_end,
        "eligible_sample_start": eligible_start,
        "online_stop": online_stop,
        "candidate_window_size": args.window_size,
        "minimum_raw_length_strictly_greater_than": minimum_raw_length,
        "condition_pass_counts": condition_counts(windows),
        "alarm_counts": alarm_counts,
        "feature_alarm_indices": len(feature_alarms),
        "residual_alarm_indices": len(residual_alarms),
        "loss_mean_thresholds": loss_thresholds,
        "stable_interval_count": len(intervals),
        "analyzed_raw_rows": analyzed_rows,
        "retained_raw_rows": retained_rows,
        "retained_ratio": retained_rows / analyzed_rows,
        "parameters": {
            "mean_threshold": args.mean_threshold,
            "std_threshold": args.std_threshold,
            "quantile_threshold": args.quantile_threshold,
            "distribution_feature_quantile": args.distribution_feature_quantile,
            "loss_low_quantile": args.loss_low_quantile,
            "loss_cv_threshold": args.loss_cv_threshold,
            "extra_min_length": extra_min_length,
        },
    }
    with (destination / "summary.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output={destination.resolve()}")


if __name__ == "__main__":
    main()

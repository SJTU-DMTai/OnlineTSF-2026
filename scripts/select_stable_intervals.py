# -*- coding: utf-8 -*-
"""Step 1: select stable intervals from an existing benchmark time series."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
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
from onlinetsf.data import SlidingWindowDataset, load_benchmark_dataset
from onlinetsf.online import FeedbackEvent, OnlineMetrics, OnlineRun
from scripts.stability import (
    CandidateWindow,
    DistributionMetrics,
    LossMetrics,
    StableInterval,
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
    parser.add_argument(
        "--window-step", type=int, default=32,
        help="forecast samples between neighboring sliding windows",
    )
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
        "--loss-block-size", type=int, default=16,
        help="forecast steps averaged before computing loss CV",
    )
    parser.add_argument(
        "--detector-overrides", type=Path,
        help="YAML file with separate features/residual detector parameter overrides",
    )
    parser.add_argument(
        "--extra-min-length",
        type=int,
        help="extra raw rows beyond context+horizon; defaults to context+horizon",
    )
    parser.add_argument("--device", help="override online.device for every model")
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("runs/stability-cache"),
        help="persistent forecast and detector cache shared across selection runs",
    )
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
    if not 0 < args.window_step <= args.window_size:
        raise ValueError("--window-step must be in [1, window-size]")
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
    if args.loss_block_size <= 0:
        raise ValueError("--loss-block-size must be positive")
    if args.window_size // args.loss_block_size < 2:
        raise ValueError("each candidate window needs at least two loss blocks")
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


def cache_key(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def collect_alarm_indices(
    config_path: str | Path,
    detector_names: Sequence[str],
    strategy: str,
    run: OnlineRun,
    source: str,
    device: str | None,
    overrides: dict[str, Any],
    cache_dir: Path,
    input_key: str,
    code_digest: str,
) -> tuple[set[int], dict[str, int], list[dict[str, Any]], dict[str, bool]]:
    alarm_indices: set[int] = set()
    alarm_counts: dict[str, int] = {}
    alarm_events: list[dict[str, Any]] = []
    cache_hits: dict[str, bool] = {}
    for detector_name in detector_names:
        config = prepare_config(config_path, strategy, detector_name, device)
        config["drift"]["source"] = source
        config["drift"]["parameters"].update(overrides.get(source, {}).get(detector_name, {}))
        key = cache_key({
            "version": 1, "input": input_key, "code": code_digest,
            "drift": config["drift"],
        })
        cache_path = cache_dir / f"detector-{key}.json"
        cache_hits[detector_name] = cache_path.exists()
        if cache_hits[detector_name]:
            with cache_path.open("r", encoding="utf-8") as handle:
                detected_events = json.load(handle)
        else:
            records = collect_drift_records(config, run)
            detected_events = [
                {
                    "source": source,
                    "strategy": strategy if source == "residual" else "",
                    "detector": detector_name,
                    "sample_index": record.index,
                    "available_at": record.available_at,
                    "variable_name": record.variable_name,
                    "variable_index": record.variable_index,
                    "horizon_step": record.horizon_step,
                    "value": record.value,
                }
                for record in records if record.detected
            ]
            temporary_path = cache_dir / f".{key}.{os.getpid()}.tmp"
            with temporary_path.open("w", encoding="utf-8") as handle:
                json.dump(detected_events, handle)
            temporary_path.replace(cache_path)
        alarm_indices.update(event["sample_index"] for event in detected_events)
        alarm_counts[detector_name] = len(detected_events)
        alarm_events.extend(detected_events)
    return alarm_indices, alarm_counts, alarm_events, cache_hits


def raw_bounds(sample_start: int, sample_end: int, stride: int, context: int, horizon: int) -> tuple[int, int]:
    raw_start = sample_start * stride
    raw_end = (sample_end - 1) * stride + context + horizon
    return raw_start, raw_end


def evaluate_window(
    sample_start: int,
    sample_end: int,
    *,
    dataset: SlidingWindowDataset,
    strategies: Sequence[str],
    losses: dict[str, dict[int, float]],
    loss_thresholds: dict[str, float],
    alarm_prefix: Sequence[int],
    args: argparse.Namespace,
    distributions: dict[tuple[int, int], DistributionMetrics],
    model_losses: dict[tuple[int, int], dict[str, LossMetrics]],
) -> CandidateWindow:
    """Score one range without rerunning a forecasting model or detector."""

    key = (sample_start, sample_end)
    raw_start, raw_end = raw_bounds(
        sample_start, sample_end, dataset.stride, dataset.context_length, dataset.horizon
    )
    distribution = distributions.get(key)
    if distribution is None:
        distribution = distribution_metrics(
            dataset.values[raw_start:raw_end],
            feature_quantile=args.distribution_feature_quantile,
        )
        distributions[key] = distribution
    per_model = model_losses.get(key)
    if per_model is None:
        per_model = {
            strategy: loss_metrics(
                [losses[strategy][index] for index in range(sample_start, sample_end)],
                block_size=args.loss_block_size,
            )
            for strategy in strategies
        }
        model_losses[key] = per_model

    return CandidateWindow(
        sample_start=sample_start,
        sample_end=sample_end,
        raw_start=raw_start,
        raw_end=raw_end,
        detector_ok=alarm_prefix[sample_end] == alarm_prefix[sample_start],
        distribution_ok=(
            distribution.mean_change <= args.mean_threshold
            and distribution.std_change <= args.std_threshold
            and distribution.quantile_change <= args.quantile_threshold
        ),
        loss_ok=all(
            per_model[strategy].mean <= loss_thresholds[strategy]
            and per_model[strategy].coefficient_of_variation <= args.loss_cv_threshold
            for strategy in strategies
        ),
    )


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
    detector_overrides: dict[str, Any] = {}
    if args.detector_overrides is not None:
        with args.detector_overrides.open("r", encoding="utf-8") as handle:
            detector_overrides = yaml.safe_load(handle) or {}
        if not isinstance(detector_overrides, dict) or set(detector_overrides) - {"features", "residual"}:
            raise ValueError("detector overrides must contain only features and residual mappings")
        for source, settings in detector_overrides.items():
            if not isinstance(settings, dict) or set(settings) - set(args.detectors):
                raise ValueError(f"{source} overrides must use selected detector names")
            if any(not isinstance(parameters, dict) for parameters in settings.values()):
                raise ValueError(f"{source} detector parameters must be mappings")

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

    last_window_start = online_stop - args.window_size
    candidate_starts = list(range(eligible_start, last_window_start + 1, args.window_step))
    if candidate_starts and candidate_starts[-1] != last_window_start:
        candidate_starts.append(last_window_start)
    candidate_ranges = [(start, start + args.window_size) for start in candidate_starts]
    if not candidate_ranges:
        raise ValueError("no complete candidate window remains after excluding offline-training data")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_digest = file_digest(data_config["path"])
    code_digest = cache_key({
        str(path.relative_to(SOURCE_ROOT)): file_digest(path)
        for path in sorted((SOURCE_ROOT / "onlinetsf").rglob("*.py"))
    })
    runs: dict[str, OnlineRun] = {}
    losses: dict[str, dict[int, float]] = {}
    residual_alarms: set[int] = set()
    alarm_counts: dict[str, dict[str, int]] = {}
    alarm_events: list[dict[str, Any]] = []
    alarm_indices_by_stream: dict[str, set[int]] = {}
    forecast_cache_hits: dict[str, bool] = {}
    detector_cache_hits: dict[str, dict[str, bool]] = {}
    for strategy in strategies:
        config = prepare_config(args.config, strategy, args.detectors[0], args.device)
        key = cache_key({
            "version": 1, "dataset": dataset_digest, "code": code_digest,
            "seed": config.get("seed"),
            "data": config["data"], "forecasting": config["forecasting"],
            "method": config["method"], "offline": config["offline"],
            "online": config["online"],
        })
        cache_path = args.cache_dir / f"forecast-{key}.pt"
        forecast_cache_hits[strategy] = cache_path.exists()
        print(f"strategy={strategy} forecast_cache={'hit' if forecast_cache_hits[strategy] else 'miss'}", flush=True)
        if forecast_cache_hits[strategy]:
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
            run = OnlineRun(
                OnlineMetrics(**cached["metrics"]),
                tuple(FeedbackEvent(**event) for event in cached["events"]),
            )
        else:
            run = run_forecast(config)
            temporary_path = args.cache_dir / f".{key}.{os.getpid()}.tmp"
            torch.save({
                "metrics": asdict(run.metrics),
                "events": [vars(event) for event in run.events],
            }, temporary_path)
            temporary_path.replace(cache_path)
        runs[strategy] = run
        losses[strategy] = aggregate_event_losses(run.events)
        alarms, counts, events, hits = collect_alarm_indices(
            args.config, args.detectors, strategy, run, "residual", args.device,
            detector_overrides, args.cache_dir, key, code_digest,
        )
        detector_cache_hits[f"residual:{strategy}"] = hits
        residual_alarms.update(alarms)
        alarm_events.extend(events)
        alarm_counts[f"residual:{strategy}"] = counts
        for detector_name in args.detectors:
            alarm_indices_by_stream[f"residual:{strategy}:{detector_name}"] = {
                event["sample_index"] for event in events if event["detector"] == detector_name
            }

    feature_input_key = cache_key({
        "version": 1, "dataset": dataset_digest, "data": data_config,
        "train_ratio": first_config["offline"]["train_ratio"],
        "online": first_config["online"],
    })
    feature_alarms, counts, events, hits = collect_alarm_indices(
        args.config,
        args.detectors,
        strategies[0],
        runs[strategies[0]],
        "features",
        args.device,
        detector_overrides,
        args.cache_dir,
        feature_input_key,
        code_digest,
    )
    detector_cache_hits["features"] = hits
    alarm_events.extend(events)
    alarm_counts["features"] = counts
    for detector_name in args.detectors:
        alarm_indices_by_stream[f"features:{detector_name}"] = {
            event["sample_index"] for event in events if event["detector"] == detector_name
        }
    all_alarm_indices = feature_alarms | residual_alarms
    alarm_prefix = [0] * (len(dataset) + 1)
    for index in all_alarm_indices:
        alarm_prefix[index + 1] = 1
    for index in range(1, len(alarm_prefix)):
        alarm_prefix[index] += alarm_prefix[index - 1]

    distribution_by_range: dict[tuple[int, int], DistributionMetrics] = {}
    loss_by_range: dict[tuple[int, int], dict[str, LossMetrics]] = {}
    for sample_start, sample_end in candidate_ranges:
        per_model: dict[str, LossMetrics] = {}
        for strategy in strategies:
            window_losses = [losses[strategy][index] for index in range(sample_start, sample_end)]
            per_model[strategy] = loss_metrics(window_losses, block_size=args.loss_block_size)
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
        windows.append(
            evaluate_window(
                sample_start, sample_end,
                dataset=dataset,
                strategies=strategies,
                losses=losses,
                loss_thresholds=loss_thresholds,
                alarm_prefix=alarm_prefix,
                args=args,
                distributions=distribution_by_range,
                model_losses=loss_by_range,
            )
        )

    extra_min_length = args.extra_min_length if args.extra_min_length is not None else context + horizon
    minimum_raw_length = context + horizon + extra_min_length
    runs_before_length = merge_stable_windows(windows, minimum_raw_length=0)
    refined_runs = []
    for run in runs_before_length:
        start = run.sample_start
        end = run.sample_end
        left_limit = max(eligible_start, start - args.window_step + 1)
        for candidate_start in range(start - 1, left_limit - 1, -1):
            result = evaluate_window(
                candidate_start, candidate_start + args.window_size,
                dataset=dataset, strategies=strategies, losses=losses,
                loss_thresholds=loss_thresholds, alarm_prefix=alarm_prefix,
                args=args, distributions=distribution_by_range, model_losses=loss_by_range,
            )
            if not result.stable:
                break
            start = candidate_start
        right_limit = min(online_stop - args.window_size, end - args.window_size + args.window_step - 1)
        for candidate_start in range(end - args.window_size + 1, right_limit + 1):
            result = evaluate_window(
                candidate_start, candidate_start + args.window_size,
                dataset=dataset, strategies=strategies, losses=losses,
                loss_thresholds=loss_thresholds, alarm_prefix=alarm_prefix,
                args=args, distributions=distribution_by_range, model_losses=loss_by_range,
            )
            if not result.stable:
                break
            end = candidate_start + args.window_size
        refined_runs.append((start, end, run.window_count))

    intervals = []
    interval_candidates: list[dict[str, Any]] = []
    whole_interval_pass_count = 0
    runs_rejected_by_length = 0
    maximum_run_length = 0
    for start, end, window_count in refined_runs:
        # Only the centers of passing windows become the selected raw interval.
        # This keeps intervals disjoint when a failed window separates two runs.
        core_start = start + args.window_size // 2
        core_end = end - (args.window_size - 1) // 2
        raw_start = core_start * stride + context
        raw_end = core_end * stride + context
        maximum_run_length = max(maximum_run_length, raw_end - raw_start)
        candidate: dict[str, Any] = {
            "sample_start": core_start,
            "sample_end": core_end,
            "raw_start": raw_start,
            "raw_end_exclusive": raw_end,
            "raw_length": raw_end - raw_start,
            "window_count": window_count,
            "length_ok": (
                raw_end - raw_start > minimum_raw_length
                and core_end - core_start >= 2 * args.loss_block_size
            ),
            "detector_ok": alarm_prefix[core_end] == alarm_prefix[core_start],
        }
        if not candidate["length_ok"]:
            runs_rejected_by_length += 1
            interval_candidates.append(candidate)
            continue
        distribution = distribution_metrics(
            dataset.values[raw_start:raw_end],
            feature_quantile=args.distribution_feature_quantile,
        )
        distribution_ok = (
            distribution.mean_change <= args.mean_threshold
            and distribution.std_change <= args.std_threshold
            and distribution.quantile_change <= args.quantile_threshold
        )
        candidate.update(
            mean_change=distribution.mean_change,
            std_change=distribution.std_change,
            quantile_change=distribution.quantile_change,
            distribution_ok=distribution_ok,
        )
        loss_ok = True
        for strategy in strategies:
            metrics = loss_metrics(
                [losses[strategy][index] for index in range(core_start, core_end)],
                block_size=args.loss_block_size,
            )
            if (
                metrics.mean > loss_thresholds[strategy]
                or metrics.coefficient_of_variation > args.loss_cv_threshold
            ):
                loss_ok = False
            candidate[f"{strategy}_loss_mean"] = metrics.mean
            candidate[f"{strategy}_loss_mean_ok"] = metrics.mean <= loss_thresholds[strategy]
            candidate[f"{strategy}_loss_cv"] = metrics.coefficient_of_variation
            candidate[f"{strategy}_loss_cv_ok"] = (
                metrics.coefficient_of_variation <= args.loss_cv_threshold
            )
        candidate["loss_ok"] = loss_ok
        candidate["stable"] = candidate["detector_ok"] and distribution_ok and loss_ok
        interval_candidates.append(candidate)
        if candidate["stable"]:
            whole_interval_pass_count += 1
            intervals.append(
                StableInterval(core_start, core_end, raw_start, raw_end, window_count)
            )
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
        "mean_change", "mean_ok", "std_change", "std_ok",
        "quantile_change", "quantile_ok",
    ]
    for stream in alarm_indices_by_stream:
        window_fields.append(f"{stream}_alarm_indices")
    for strategy in strategies:
        window_fields.extend(
            (
                f"{strategy}_loss_mean", f"{strategy}_loss_mean_ok",
                f"{strategy}_loss_cv", f"{strategy}_loss_cv_ok",
            )
        )
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
                "mean_ok": distribution.mean_change <= args.mean_threshold,
                "std_change": distribution.std_change,
                "std_ok": distribution.std_change <= args.std_threshold,
                "quantile_change": distribution.quantile_change,
                "quantile_ok": distribution.quantile_change <= args.quantile_threshold,
            }
            for stream, alarm_indices in alarm_indices_by_stream.items():
                row[f"{stream}_alarm_indices"] = sum(
                    window.sample_start <= index < window.sample_end for index in alarm_indices
                )
            for strategy in strategies:
                metrics = loss_by_range[candidate][strategy]
                row[f"{strategy}_loss_mean"] = metrics.mean
                row[f"{strategy}_loss_mean_ok"] = metrics.mean <= loss_thresholds[strategy]
                row[f"{strategy}_loss_cv"] = metrics.coefficient_of_variation
                row[f"{strategy}_loss_cv_ok"] = (
                    metrics.coefficient_of_variation <= args.loss_cv_threshold
                )
            writer.writerow(row)

    event_fields = [
        "source", "strategy", "detector", "sample_index", "available_at",
        "variable_name", "variable_index", "horizon_step", "value",
    ]
    with (destination / "alarm_events.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=event_fields)
        writer.writeheader()
        writer.writerows(alarm_events)

    interval_candidate_fields = [
        "sample_start", "sample_end", "raw_start", "raw_end_exclusive", "raw_length",
        "window_count", "length_ok", "detector_ok", "mean_change", "std_change",
        "quantile_change", "distribution_ok", "loss_ok", "stable",
    ]
    for strategy in strategies:
        interval_candidate_fields.extend(
            (
                f"{strategy}_loss_mean", f"{strategy}_loss_mean_ok",
                f"{strategy}_loss_cv", f"{strategy}_loss_cv_ok",
            )
        )
    with (destination / "interval_candidates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=interval_candidate_fields)
        writer.writeheader()
        writer.writerows(interval_candidates)

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
        "cache": {
            "directory": str(args.cache_dir.resolve()),
            "forecast_hits": forecast_cache_hits,
            "detector_hits": detector_cache_hits,
        },
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
        "candidate_window_step": args.window_step,
        "minimum_raw_length_strictly_greater_than": minimum_raw_length,
        "stable_runs_before_length_filter": len(runs_before_length),
        "runs_rejected_by_length": runs_rejected_by_length,
        "runs_rejected_by_whole_interval_check": (
            len(refined_runs) - runs_rejected_by_length - whole_interval_pass_count
        ),
        "maximum_stable_run_raw_length": maximum_run_length,
        "condition_pass_counts": condition_counts(windows),
        "subcondition_pass_counts": {
            "feature_detector": sum(
                not any(window.sample_start <= index < window.sample_end for index in feature_alarms)
                for window in windows
            ),
            "residual_detector": sum(
                not any(window.sample_start <= index < window.sample_end for index in residual_alarms)
                for window in windows
            ),
            "mean": sum(
                distribution_by_range[(start, end)].mean_change <= args.mean_threshold
                for start, end in candidate_ranges
            ),
            "std": sum(
                distribution_by_range[(start, end)].std_change <= args.std_threshold
                for start, end in candidate_ranges
            ),
            "quantile": sum(
                distribution_by_range[(start, end)].quantile_change <= args.quantile_threshold
                for start, end in candidate_ranges
            ),
            "models": {
                strategy: {
                    "low_loss": sum(
                        loss_by_range[candidate][strategy].mean <= loss_thresholds[strategy]
                        for candidate in candidate_ranges
                    ),
                    "stable_loss": sum(
                        loss_by_range[candidate][strategy].coefficient_of_variation
                        <= args.loss_cv_threshold
                        for candidate in candidate_ranges
                    ),
                    "both": sum(
                        loss_by_range[candidate][strategy].mean <= loss_thresholds[strategy]
                        and loss_by_range[candidate][strategy].coefficient_of_variation
                        <= args.loss_cv_threshold
                        for candidate in candidate_ranges
                    ),
                }
                for strategy in strategies
            },
        },
        "alarm_counts": alarm_counts,
        "alarm_index_rates_per_1000_samples": {
            stream: 1000.0 * len(indices) / (online_stop - online_start)
            for stream, indices in alarm_indices_by_stream.items()
        },
        "globally_silent_detector_streams": [
            stream for stream, indices in alarm_indices_by_stream.items() if not indices
        ],
        "feature_alarm_indices": len(feature_alarms),
        "residual_alarm_indices": len(residual_alarms),
        "loss_mean_thresholds": loss_thresholds,
        "observed_window_quantiles": {
            "mean_change": {
                str(q): quantile_threshold(
                    [distribution_by_range[candidate].mean_change for candidate in candidate_ranges], q
                )
                for q in (0.25, 0.5, 0.75, 0.9)
            },
            "std_change": {
                str(q): quantile_threshold(
                    [distribution_by_range[candidate].std_change for candidate in candidate_ranges], q
                )
                for q in (0.25, 0.5, 0.75, 0.9)
            },
            "quantile_change": {
                str(q): quantile_threshold(
                    [distribution_by_range[candidate].quantile_change for candidate in candidate_ranges], q
                )
                for q in (0.25, 0.5, 0.75, 0.9)
            },
            "loss_cv": {
                strategy: {
                    str(q): quantile_threshold(
                        [
                            loss_by_range[candidate][strategy].coefficient_of_variation
                            for candidate in candidate_ranges
                        ], q
                    )
                    for q in (0.25, 0.5, 0.75, 0.9)
                }
                for strategy in strategies
            },
        },
        "effective_detector_parameters": {
            source: {
                detector_name: {
                    **prepare_config(args.config, strategies[0], detector_name, args.device)["drift"]["parameters"],
                    **detector_overrides.get(source, {}).get(detector_name, {}),
                }
                for detector_name in args.detectors
            }
            for source in ("features", "residual")
        },
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
            "loss_block_size": args.loss_block_size,
            "detector_overrides": str(args.detector_overrides) if args.detector_overrides else None,
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

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
from onlinetsf.config import DETECTOR_SOURCES, DRIFT_SOURCES, load_config
from onlinetsf.data import SlidingWindowDataset, load_benchmark_dataset
from onlinetsf.online import FeedbackEvent, OnlineMetrics, OnlineRun
from scripts.stability_calibration import calibrate_stream, calibration_streams
from scripts.stability import (
    CandidateWindow,
    DistributionMetrics,
    LossMetrics,
    StableInterval,
    condition_counts,
    covered_raw_length,
    distribution_metrics,
    loss_metrics,
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
        default=["page_hinkley", "adwin", "kswin", "seed", "stepd", "hddmw", "abcd"],
        help="detector profiles; each runs only on its supported feature or residual input",
    )
    parser.add_argument("--window-size", type=int, default=256, help="forecast samples per candidate window")
    parser.add_argument(
        "--window-step", type=int, default=8,
        help="forecast samples between neighboring sliding windows",
    )
    parser.add_argument("--mean-threshold", type=float, default=0.8)
    parser.add_argument("--std-threshold", type=float, default=0.6)
    parser.add_argument("--quantile-threshold", type=float, default=1.15)
    parser.add_argument("--distribution-feature-quantile", type=float, default=0.95)
    parser.add_argument(
        "--loss-low-quantile",
        type=float,
        default=0.60,
        help="quantile of candidate mean losses defining a low loss for each model",
    )
    parser.add_argument("--loss-cv-threshold", type=float, default=1.5)
    parser.add_argument(
        "--loss-block-size", type=int, default=16,
        help="forecast steps averaged before computing loss CV",
    )
    parser.add_argument(
        "--detector-overrides", type=Path,
        help="YAML detector parameters by source, dataset, feature variable, or residual model",
    )
    parser.add_argument(
        "--auto-calibrate-detectors", action="store_true",
        help="calibrate Page-Hinkley, ADWIN, and KSWIN per feature and model residual stream",
    )
    parser.add_argument(
        "--calibration-samples", type=int, default=512,
        help="maximum post-training forecast samples reserved for detector calibration",
    )
    parser.add_argument(
        "--extra-min-length",
        type=int,
        help="optional extra raw rows beyond context+horizon required for each selected window",
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
    if args.calibration_samples < 64:
        raise ValueError("--calibration-samples must be at least 64")
    if args.window_size // args.loss_block_size < 2:
        raise ValueError("each candidate window needs at least two loss blocks")
    if not strategies:
        raise ValueError("at least one non-linear strategy is required")
    if len(args.detectors) != len(set(args.detectors)):
        raise ValueError("--detectors must not contain duplicate names")


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


def detector_parameters(
    overrides: dict[str, Any], dataset_name: str, source: str, strategy: str,
    detector_name: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    dataset_overrides = overrides.get("datasets", {}).get(dataset_name, {})
    parameters = dict(overrides.get(source, {}).get(detector_name, {}))
    parameters.update(dataset_overrides.get(source, {}).get(detector_name, {}))
    if source == "residual":
        parameters.update(
            dataset_overrides.get("residual_models", {}).get(strategy, {}).get(detector_name, {})
        )
        variable_parameters = {
            variable: detectors[detector_name]
            for variable, detectors in dataset_overrides.get("residual_variables", {}).get(strategy, {}).items()
            if detector_name in detectors
        }
        return parameters, variable_parameters
    variable_parameters = {
        variable: detectors[detector_name]
        for variable, detectors in dataset_overrides.get("feature_variables", {}).items()
        if detector_name in detectors
    }
    return parameters, variable_parameters


def validate_detector_settings(settings: Any, detectors: Sequence[str], label: str) -> None:
    if not isinstance(settings, dict) or set(settings) - set(detectors):
        raise ValueError(f"{label} must map selected detector names to parameters")
    if any(not isinstance(parameters, dict) for parameters in settings.values()):
        raise ValueError(f"{label} detector parameters must be mappings")


def collect_alarm_indices(
    config_path: str | Path,
    detector_names: Sequence[str],
    strategy: str,
    run: OnlineRun,
    source: str,
    device: str | None,
    overrides: dict[str, Any],
    dataset: SlidingWindowDataset,
    cache_dir: Path,
    input_key: str,
) -> tuple[set[int], dict[str, int], list[dict[str, Any]], dict[str, bool]]:
    alarm_indices: set[int] = set()
    alarm_counts: dict[str, int] = {}
    alarm_events: list[dict[str, Any]] = []
    cache_hits: dict[str, bool] = {}
    for detector_name in detector_names:
        if source not in DETECTOR_SOURCES.get(detector_name, DRIFT_SOURCES):
            continue
        config = prepare_config(config_path, strategy, detector_name, device)
        config["data"]["feature_names"] = list(dataset.feature_names)
        config["data"]["target_names"] = list(dataset.target_names)
        config["drift"]["source"] = source
        parameters, variable_parameters = detector_parameters(
            overrides, config["data"]["name"], source, strategy, detector_name
        )
        config["drift"]["parameters"].update(parameters)
        config["drift"]["variable_parameters"] = variable_parameters
        key = cache_key({
            "version": 2, "input": input_key,
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


def detector_alarm_votes(start: int, end: int, votes_by_index: Sequence[int]) -> int:
    """Return the strongest same-sample detector agreement in a window."""

    return max(votes_by_index[start:end], default=0)


def build_alarm_vote_prefixes(
    alarm_events: Sequence[dict[str, Any]], detector_names: Sequence[str], dataset_length: int,
) -> tuple[dict[str, list[int]], list[int], dict[str, int]]:
    """Veto only simultaneous majority alarms from comparable detector streams."""

    sources = ("features", "residual")
    votes_to_veto = {
        source: sum(
            source in DETECTOR_SOURCES.get(name, DRIFT_SOURCES) for name in detector_names
        ) // 2 + 1
        for source in sources
    }
    votes_by_scope: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    for event in alarm_events:
        source = event["source"]
        scope = event["strategy"] if source == "residual" else event["variable_name"]
        votes_by_scope[(source, scope, event["sample_index"])].add(event["detector"])

    votes_by_index = [0] * dataset_length
    consensus_indices: dict[str, set[int]] = {source: set() for source in sources}
    for (source, _, index), voters in votes_by_scope.items():
        votes_by_index[index] = max(votes_by_index[index], len(voters))
        if len(voters) >= votes_to_veto[source]:
            consensus_indices[source].add(index)

    alarm_prefixes: dict[str, list[int]] = {}
    for source, indices in consensus_indices.items():
        prefix = [0] * (dataset_length + 1)
        for index in indices:
            prefix[index + 1] = 1
        for index in range(1, len(prefix)):
            prefix[index] += prefix[index - 1]
        alarm_prefixes[source] = prefix
    return alarm_prefixes, votes_by_index, votes_to_veto


def evaluate_window(
    sample_start: int,
    sample_end: int,
    *,
    dataset: SlidingWindowDataset,
    strategies: Sequence[str],
    losses: dict[str, dict[int, float]],
    loss_thresholds: dict[str, float],
    alarm_prefixes: dict[str, list[int]],
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
        detector_ok=all(
            prefix[sample_end] == prefix[sample_start]
            for prefix in alarm_prefixes.values()
        ),
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
        if not isinstance(detector_overrides, dict) or set(detector_overrides) - {
            "features", "residual", "datasets"
        }:
            raise ValueError("detector overrides must contain only features, residual, and datasets")
        for source in ("features", "residual"):
            if source in detector_overrides:
                validate_detector_settings(detector_overrides[source], args.detectors, source)

    first_config = prepare_config(args.config, strategies[0], args.detectors[0], args.device)
    data_config = first_config["data"]
    dataset = load_benchmark_dataset(
        data_config["name"],
        data_config["path"],
        context_length=data_config["context_length"],
        horizon=data_config["horizon"],
        stride=data_config.get("stride", 1),
    )
    datasets = detector_overrides.get("datasets", {})
    if not isinstance(datasets, dict):
        raise ValueError("detector overrides datasets must be a mapping")
    for dataset_name, settings in datasets.items():
        if not isinstance(settings, dict) or set(settings) - {
            "features", "residual", "feature_variables", "residual_models", "residual_variables"
        }:
            raise ValueError(f"datasets.{dataset_name} has invalid override sections")
        for source in ("features", "residual"):
            if source in settings:
                validate_detector_settings(settings[source], args.detectors, f"datasets.{dataset_name}.{source}")
        for section, known_names in (
            ("feature_variables", dataset.feature_names), ("residual_models", strategies)
        ):
            entries = settings.get(section, {})
            if not isinstance(entries, dict):
                raise ValueError(f"datasets.{dataset_name}.{section} must be a mapping")
            for name, parameters in entries.items():
                if dataset_name == data_config["name"] and name not in known_names:
                    raise ValueError(f"unknown {section} name {name!r} for {dataset_name}")
                validate_detector_settings(
                    parameters, args.detectors, f"datasets.{dataset_name}.{section}.{name}"
                )
        residual_variables = settings.get("residual_variables", {})
        if not isinstance(residual_variables, dict):
            raise ValueError(f"datasets.{dataset_name}.residual_variables must be a mapping")
        for strategy, variables in residual_variables.items():
            if dataset_name == data_config["name"] and strategy not in strategies:
                raise ValueError(f"unknown residual model {strategy!r} for {dataset_name}")
            if not isinstance(variables, dict):
                raise ValueError("residual_variables model entries must be mappings")
            for variable, parameters in variables.items():
                if dataset_name == data_config["name"] and variable not in dataset.target_names:
                    raise ValueError(f"unknown residual variable {variable!r} for {dataset_name}")
                validate_detector_settings(
                    parameters, args.detectors,
                    f"datasets.{dataset_name}.residual_variables.{strategy}.{variable}",
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

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_digest = file_digest(data_config["path"])
    runs: dict[str, OnlineRun] = {}
    losses: dict[str, dict[int, float]] = {}
    forecast_keys: dict[str, str] = {}
    residual_alarms: set[int] = set()
    alarm_counts: dict[str, dict[str, int]] = {}
    alarm_events: list[dict[str, Any]] = []
    alarm_indices_by_stream: dict[str, set[int]] = {}
    forecast_cache_hits: dict[str, bool] = {}
    detector_cache_hits: dict[str, dict[str, bool]] = {}
    for strategy in strategies:
        config = prepare_config(args.config, strategy, args.detectors[0], args.device)
        key = cache_key({
            "version": 2, "dataset": dataset_digest,
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
        forecast_keys[strategy] = key
        losses[strategy] = aggregate_event_losses(run.events)

    calibration_report: dict[str, Any] = {}
    calibratable_detectors = [
        name for name in args.detectors if name in {"page_hinkley", "adwin", "kswin"}
    ]
    if args.auto_calibrate_detectors and calibratable_detectors:
        calibration_start = eligible_start
        available = online_stop - calibration_start - args.window_size
        calibration_length = min(args.calibration_samples, available // 2)
        if calibration_length < 64:
            raise ValueError("not enough post-training data for calibration and a candidate window")
        calibration_end = calibration_start + calibration_length
        calibration_report = {
            "sample_start": calibration_start,
            "sample_end_exclusive": calibration_end,
            "streams": {"features": {}, "residual": {}},
        }
        for source in ("features", "residual"):
            source_strategies = strategies[:1] if source == "features" else strategies
            for strategy in source_strategies:
                stream_reports = (
                    calibration_report["streams"]["features"] if source == "features"
                    else calibration_report["streams"]["residual"].setdefault(strategy, {})
                )
                for detector_name in calibratable_detectors:
                    config = prepare_config(args.config, strategy, detector_name, args.device)
                    config["data"]["feature_names"] = list(dataset.feature_names)
                    config["data"]["target_names"] = list(dataset.target_names)
                    config["drift"]["source"] = source
                    parameters, variable_parameters = detector_parameters(
                        detector_overrides, data_config["name"], source, strategy, detector_name
                    )
                    config["drift"]["parameters"].update(parameters)
                    streams = calibration_streams(
                        config, runs[strategy], calibration_start, calibration_end
                    )
                    for variable, values in streams.items():
                        stream_config = {**config, "drift": {
                            **config["drift"], "parameters": {
                                **config["drift"]["parameters"], **variable_parameters.get(variable, {}),
                            },
                        }}
                        selected, report = calibrate_stream(stream_config, values)
                        stream_reports.setdefault(variable, {})[detector_name] = report
                        if source == "features":
                            target = detector_overrides.setdefault("datasets", {}).setdefault(
                                data_config["name"], {}
                            ).setdefault("feature_variables", {}).setdefault(variable, {})
                        else:
                            target = detector_overrides.setdefault("datasets", {}).setdefault(
                                data_config["name"], {}
                            ).setdefault("residual_variables", {}).setdefault(strategy, {}).setdefault(variable, {})
                        target.setdefault(detector_name, {}).update(selected)
        # Keep calibration targets/contexts out of the selected raw intervals.
        eligible_start = max(
            eligible_start, calibration_end + math.ceil((context + horizon) / stride)
        )

    candidate_starts = list(range(eligible_start, last_window_start + 1, args.window_step))
    if candidate_starts and candidate_starts[-1] != last_window_start:
        candidate_starts.append(last_window_start)
    candidate_ranges = [(start, start + args.window_size) for start in candidate_starts]
    if not candidate_ranges:
        raise ValueError("no complete candidate window remains after calibration and offline training")

    for strategy in strategies:
        alarms, counts, events, hits = collect_alarm_indices(
            args.config, args.detectors, strategy, runs[strategy], "residual", args.device,
            detector_overrides, dataset, args.cache_dir, forecast_keys[strategy],
        )
        detector_cache_hits[f"residual:{strategy}"] = hits
        residual_alarms.update(alarms)
        alarm_events.extend(events)
        alarm_counts[f"residual:{strategy}"] = counts
        for detector_name in args.detectors:
            if "residual" not in DETECTOR_SOURCES.get(detector_name, DRIFT_SOURCES):
                continue
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
        dataset,
        args.cache_dir,
        feature_input_key,
    )
    detector_cache_hits["features"] = hits
    alarm_events.extend(events)
    alarm_counts["features"] = counts
    for detector_name in args.detectors:
        if "features" not in DETECTOR_SOURCES.get(detector_name, DRIFT_SOURCES):
            continue
        alarm_indices_by_stream[f"features:{detector_name}"] = {
            event["sample_index"] for event in events if event["detector"] == detector_name
        }
    alarm_prefixes, votes_by_index, votes_to_veto = build_alarm_vote_prefixes(
        alarm_events, args.detectors, len(dataset)
    )

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
                alarm_prefixes=alarm_prefixes,
                args=args,
                distributions=distribution_by_range,
                model_losses=loss_by_range,
            )
        )

    minimum_raw_length = (
        context + horizon + args.extra_min_length if args.extra_min_length is not None else 0
    )
    intervals: list[StableInterval] = []
    interval_candidates: list[dict[str, Any]] = []
    for window in windows:
        if not window.stable:
            continue
        # The context is used to predict, but only the target rows are selected.
        raw_start = window.sample_start * stride + context
        raw_end = (window.sample_end - 1) * stride + context + horizon
        length_ok = raw_end - raw_start > minimum_raw_length
        selected = length_ok and (not intervals or raw_start >= intervals[-1].raw_end)
        distribution = distribution_by_range[(window.sample_start, window.sample_end)]
        candidate: dict[str, Any] = {
            "sample_start": window.sample_start,
            "sample_end": window.sample_end,
            "raw_start": raw_start,
            "raw_end_exclusive": raw_end,
            "raw_length": raw_end - raw_start,
            "window_count": 1,
            "length_ok": length_ok,
            "selected": selected,
            "detector_alarm_votes": detector_alarm_votes(
                window.sample_start, window.sample_end, votes_by_index
            ),
            "detector_ok": window.detector_ok,
            "mean_change": distribution.mean_change,
            "std_change": distribution.std_change,
            "quantile_change": distribution.quantile_change,
            "distribution_ok": window.distribution_ok,
            "loss_ok": window.loss_ok,
            "stable": window.stable,
        }
        for strategy in strategies:
            metrics = loss_by_range[(window.sample_start, window.sample_end)][strategy]
            candidate[f"{strategy}_loss_mean"] = metrics.mean
            candidate[f"{strategy}_loss_mean_ok"] = metrics.mean <= loss_thresholds[strategy]
            candidate[f"{strategy}_loss_cv"] = metrics.coefficient_of_variation
            candidate[f"{strategy}_loss_cv_ok"] = (
                metrics.coefficient_of_variation <= args.loss_cv_threshold
            )
        interval_candidates.append(candidate)
        if selected:
            intervals.append(
                StableInterval(window.sample_start, window.sample_end, raw_start, raw_end, 1)
            )
    analyzed_raw_start = windows[0].sample_start * stride + context
    analyzed_raw_end = (windows[-1].sample_end - 1) * stride + context + horizon
    retained_rows = covered_raw_length(intervals)
    analyzed_rows = analyzed_raw_end - analyzed_raw_start
    timestamps = read_timestamps(data_config["path"], "date")
    effective_detector_parameters: dict[str, dict[str, dict[str, Any]]] = {}
    for source in ("features", "residual"):
        effective_detector_parameters[source] = {}
        for detector_name in args.detectors:
            if source not in DETECTOR_SOURCES.get(detector_name, DRIFT_SOURCES):
                continue
            parameters, _ = detector_parameters(
                detector_overrides, data_config["name"], source, "", detector_name
            )
            effective_detector_parameters[source][detector_name] = {
                **prepare_config(args.config, strategies[0], detector_name, args.device)["drift"]["parameters"],
                **parameters,
            }
    effective_stream_parameters: dict[str, Any] = {"features": {}, "residual": {}}
    for variable in dataset.feature_names:
        effective_stream_parameters["features"][variable] = {}
        for detector_name in args.detectors:
            if "features" not in DETECTOR_SOURCES.get(detector_name, DRIFT_SOURCES) or detector_name == "abcd":
                continue
            _, per_variable = detector_parameters(
                detector_overrides, data_config["name"], "features", strategies[0], detector_name
            )
            effective_stream_parameters["features"][variable][detector_name] = {
                **effective_detector_parameters["features"][detector_name],
                **per_variable.get(variable, {}),
            }
    if "abcd" in args.detectors:
        effective_stream_parameters["features"]["all_features"] = {
            "abcd": effective_detector_parameters["features"]["abcd"]
        }
    for strategy in strategies:
        effective_stream_parameters["residual"][strategy] = {}
        for detector_name in args.detectors:
            if "residual" not in DETECTOR_SOURCES.get(detector_name, DRIFT_SOURCES):
                continue
            parameters, per_variable = detector_parameters(
                detector_overrides, data_config["name"], "residual", strategy, detector_name
            )
            baseline = {
                **prepare_config(args.config, strategy, detector_name, args.device)["drift"]["parameters"],
                **parameters,
            }
            if per_variable:
                effective_stream_parameters["residual"][strategy][detector_name] = {
                    variable: {**baseline, **per_variable.get(variable, {})}
                    for variable in dataset.target_names
                }
            else:
                effective_stream_parameters["residual"][strategy][detector_name] = baseline

    destination = output_directory(args, data_config["name"])
    destination.mkdir(parents=True, exist_ok=False)
    if calibration_report:
        with (destination / "detector_calibration.json").open("w", encoding="utf-8") as handle:
            json.dump(calibration_report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    window_fields = [
        "sample_start", "sample_end", "raw_start", "raw_end",
        "feature_detector_ok", "residual_detector_ok", "detector_alarm_votes", "detector_ok",
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
                "feature_detector_ok": (
                    alarm_prefixes["features"][window.sample_end]
                    == alarm_prefixes["features"][window.sample_start]
                ),
                "residual_detector_ok": (
                    alarm_prefixes["residual"][window.sample_end]
                    == alarm_prefixes["residual"][window.sample_start]
                ),
                "feature_alarm_indices": sum(
                    window.sample_start <= index < window.sample_end
                    for index in feature_alarms
                ),
                "residual_alarm_indices": sum(
                    window.sample_start <= index < window.sample_end
                    for index in residual_alarms
                ),
                "detector_alarm_votes": detector_alarm_votes(
                    window.sample_start, window.sample_end, votes_by_index
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
        "window_count", "length_ok", "selected", "detector_alarm_votes", "detector_ok", "mean_change", "std_change",
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
        "detector_voting": {
            "rule": "simultaneous_majority_alarm_veto",
            "voters": list(args.detectors),
            "votes_to_veto": votes_to_veto,
            "scope": "same sample index and source; feature alarms share a variable, residual alarms share a strategy",
        },
        "offline_train_ratio": first_config["offline"]["train_ratio"],
        "offline_training_windows": train_size,
        "offline_raw_end_exclusive": offline_raw_end,
        "eligible_sample_start": eligible_start,
        "online_stop": online_stop,
        "candidate_window_size": args.window_size,
        "candidate_window_step": args.window_step,
        "minimum_raw_length_strictly_greater_than": minimum_raw_length,
        "passing_window_count": len(interval_candidates),
        "windows_rejected_by_length": sum(not candidate["length_ok"] for candidate in interval_candidates),
        "windows_skipped_due_to_overlap": sum(
            candidate["length_ok"] and not candidate["selected"] for candidate in interval_candidates
        ),
        "maximum_selected_raw_length": max((interval.raw_length for interval in intervals), default=0),
        "condition_pass_counts": condition_counts(windows),
        "subcondition_pass_counts": {
            "feature_detector": sum(
                alarm_prefixes["features"][window.sample_end]
                == alarm_prefixes["features"][window.sample_start]
                for window in windows
            ),
            "residual_detector": sum(
                alarm_prefixes["residual"][window.sample_end]
                == alarm_prefixes["residual"][window.sample_start]
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
        "effective_detector_parameters": effective_detector_parameters,
        "effective_detector_stream_parameters": effective_stream_parameters,
        "detector_calibration": {
            "enabled": bool(calibration_report),
            "not_calibrated": [name for name in args.detectors if name not in calibratable_detectors],
            "sample_start": calibration_report.get("sample_start"),
            "sample_end_exclusive": calibration_report.get("sample_end_exclusive"),
            "report": "detector_calibration.json" if calibration_report else None,
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
            "auto_calibrate_detectors": args.auto_calibrate_detectors,
            "calibration_samples": args.calibration_samples,
            "extra_min_length": args.extra_min_length,
        },
    }
    with (destination / "summary.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output={destination.resolve()}")


if __name__ == "__main__":
    main()

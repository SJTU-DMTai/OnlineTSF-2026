# -*- coding: utf-8 -*-
"""Command-line entry point for a configured online forecasting experiment."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from time import perf_counter
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from .config import BINARY_RESIDUAL_DETECTORS, load_config
from .data import load_benchmark_dataset
from .documentation import write_experiment_documents
from .drift import (
    ADWINDetector, CapyMOADetector, DriftObservation, DriftRecord,
    HDDMWDetector, KSWINDetector, PageHinkleyDetector,
)
from .factory import build_backbone, build_method
from .methods import DSOFMethod, FSNetMethod, OGDMethod, OneNetMethod, UnderCaliMethod
from .online import FeedbackEvent, OnlineExecutor, OnlineRun


def _run_offline_training(
    dataset, method: OGDMethod | FSNetMethod | OneNetMethod | DSOFMethod | UnderCaliMethod, offline: dict[str, Any]
) -> int:
    """Train the model on an initial window prefix before online evaluation."""

    train_size = int(len(dataset) * offline["train_ratio"])
    if train_size == 0:
        if offline["train_ratio"] == 0.0:
            return 0
        raise ValueError("config.offline.train_ratio selects no training windows")

    loader = DataLoader(
        Subset(dataset, range(train_size)),
        batch_size=offline["batch_size"],
        shuffle=True,
    )
    if isinstance(method, DSOFMethod):
        for _ in range(offline["epochs"]):
            for context, target in loader:
                method.train_batch(context, target)
        method.reset_online_state()
        return train_size

    if isinstance(method, OneNetMethod):
        for _ in range(offline["epochs"]):
            for context, target in loader:
                method.train_batch(context, target)
        method.reset_online_state()
        return train_size

    optimizer = method.optimizer
    loss_fn = method.loss_fn
    if optimizer is None or loss_fn is None:
        raise ValueError("offline training requires config.method.learning_rate")

    method.model.train()
    for _ in range(offline["epochs"]):
        for context, target in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = method.model(context.to(method.device))
            loss = loss_fn(prediction, target.to(method.device))
            loss.backward()
            optimizer.step()
            if isinstance(method, FSNetMethod):
                method.model.record_gradients()
    return train_size


def _build_detector(config: dict[str, Any], parameters: dict[str, Any] | None = None):
    drift = config["drift"]
    options = drift["parameters"] if parameters is None else parameters
    if drift["name"] == "none":
        return None
    if drift["name"] == "page_hinkley":
        return PageHinkleyDetector(**options)
    if drift["name"] == "adwin":
        return ADWINDetector(**options)
    if drift["name"] == "kswin":
        return KSWINDetector(**options)
    if drift["name"] == "hddmw":
        return HDDMWDetector(source=drift["source"], **options)
    return CapyMOADetector(drift["name"], **options)


def run_forecast(config: dict[str, Any]) -> OnlineRun:
    """Run one configured prequential forecasting experiment."""

    experiment_started = perf_counter()
    seed = config.get("seed")
    if seed is not None:
        torch.manual_seed(seed)

    data = config["data"]
    dataset = load_benchmark_dataset(
        data["name"],
        data["path"],
        context_length=data["context_length"],
        horizon=data["horizon"],
        stride=data.get("stride", 1),
        time_column=data.get("time_column"),
        target_columns=data.get("target_columns"),
        feature_columns=data.get("feature_columns"),
        include_time_features=data.get("include_time_features", config["forecasting"]["backbone"] in {
            "tcn", "fsnet_tcn", "onenet_tcn"
        }),
        keep_decision_inputs=config.get("decision") is not None,
    )
    train_windows = int(len(dataset) * config["offline"]["train_ratio"])
    dataset.standardize(train_windows)
    data["target_names"] = list(dataset.target_names)
    data["feature_names"] = list(dataset.feature_names)
    data["num_features"] = dataset.num_features
    model = build_backbone(config, dataset.num_features, dataset.num_targets, dataset.target_indices)
    method = build_method(config, model)
    decision = config.get("decision")
    adaptive = None
    if decision is not None:
        from .decision.runtime import AdaptiveMethod

        adaptive = AdaptiveMethod.from_file(method, decision["selector"], decision.get("controller", "policy_full"))
    setup_seconds = perf_counter() - experiment_started
    offline_started = perf_counter()
    offline_stop = _run_offline_training(dataset, method, config["offline"])
    if isinstance(method, UnderCaliMethod):
        method.pretrain_uncertainty(dataset, offline_stop, config["offline"]["batch_size"])
    offline_training_seconds = perf_counter() - offline_started
    online = config["online"]
    if adaptive is not None:
        method = adaptive
    executor = OnlineExecutor(
        method,
        feedback_delay=online["feedback_delay"],
        keep_predictions=(
            online.get("keep_predictions", False)
            or config["output"]["write_per_value_errors"]
            or config["drift"]["name"] != "none"
        ),
    )
    online_started = perf_counter()
    # The final offline window's target must end before the first evaluated target begins.
    online_start = offline_stop
    if offline_stop:
        online_start += (data["horizon"] - 1) // data.get("stride", 1)
    run = executor.run_dataset(
        dataset, start=max(online_start, online.get("start", 0)), stop=online.get("stop"),
    )
    online_evaluation_seconds = perf_counter() - online_started
    metrics = replace(
        run.metrics,
        setup_seconds=setup_seconds,
        offline_training_seconds=offline_training_seconds,
        online_evaluation_seconds=online_evaluation_seconds,
        total_seconds=perf_counter() - experiment_started,
    )
    if decision is not None:
        metrics = replace(metrics, gradient_steps=method.gradient_steps, formula_steps=method.egd_steps,
                          gradient_update_clocks=method.gradient_update_clocks, decision_count=method.decision_count,
                          decision_seconds=method.decision_seconds, update_seconds=method.update_seconds)
    return replace(run, metrics=metrics)


def _drift_values(config: dict[str, Any], event: FeedbackEvent) -> list[DriftObservation]:
    source = config["drift"]["source"]
    if source == "features":
        if event.features is None:
            raise ValueError("feature drift detection requires retained feature values")
        feature_names = config["data"].get(
            "feature_names", [f"feature_{index}" for index in range(event.features.numel())]
        )
        return [
            DriftObservation(name, index, None, value, event.index)
            for index, (name, value) in enumerate(
                zip(feature_names, event.features[:len(feature_names)].tolist(), strict=True)
            )
        ]

    if event.target is None or event.observed_mask is None:
        raise ValueError("target and residual drift detection require retained forecast values")
    values: list[DriftObservation] = []
    target_names = config["data"].get(
        "target_names", [f"target_{index}" for index in range(event.target.shape[1])]
    )
    for horizon_index in range(event.target.shape[0]):
        for target_index in range(event.target.shape[1]):
            if not event.observed_mask[horizon_index, target_index].item():
                continue
            value = event.target[horizon_index, target_index].item()
            if source == "residual":
                if event.prediction is None:
                    raise ValueError("residual drift detection requires retained forecast values")
                value = event.prediction[horizon_index, target_index].item() - value
                if config["drift"]["name"] in BINARY_RESIDUAL_DETECTORS:
                    error = abs(value) > config["drift"]["error_threshold"]
                    value = float(error if config["drift"]["name"] == "hddmw" else not error)
                elif config["drift"]["name"] == "seed":
                    value = abs(value)
            values.append(
                DriftObservation(
                    target_names[target_index],
                    target_index,
                    horizon_index + 1,
                    value,
                    event.available_at,
                )
            )
    return values


def raw_signal_index(
    data: dict[str, Any], forecast_index: int, source: str, horizon_step: int | None
) -> int:
    """Map a detector input to its zero-based row in the generated CSV."""

    context_length = data["context_length"]
    raw_context_start = forecast_index * data.get("stride", 1)
    if source == "features":
        return raw_context_start + context_length - 1
    if horizon_step is None:
        raise ValueError("target and residual observations require a horizon step")
    return raw_context_start + context_length + horizon_step - 1


def collect_drift_records(config: dict[str, Any], run: OnlineRun) -> list[DriftRecord]:
    """Apply independent detectors to the configured source variables."""

    records: list[DriftRecord] = []
    detectors: dict[str, Any] = {}
    processed_feature_indices: set[int] = set()
    processed_target_positions: set[tuple[int, int]] = set()
    source = config["drift"]["source"]
    if config["drift"]["name"] == "abcd":
        detector = _build_detector(config)
        for event in run.events:
            if event.index in processed_feature_indices:
                continue
            processed_feature_indices.add(event.index)
            if event.features is None:
                raise ValueError("ABCD requires retained feature values")
            numeric_features = config["data"].get("feature_names", event.features)
            update = detector.update(event.features[:len(numeric_features)])
            records.append(DriftRecord(
                detector="abcd", source="features", index=event.index,
                available_at=event.index,
                raw_signal_index=raw_signal_index(config["data"], event.index, "features", None),
                variable_name="all_features",
                variable_index=-1, horizon_step=None, value=update.value,
                detected=update.detected,
            ))
        return records
    if config["drift"]["name"] != "none":
        for event in run.events:
            if source == "features" and event.index in processed_feature_indices:
                continue
            if source == "features":
                processed_feature_indices.add(event.index)
            for observation in _drift_values(config, event):
                if source == "target":
                    target_position = (
                        event.index * config["data"].get("stride", 1)
                        + config["data"]["context_length"]
                        + observation.horizon_step
                        - 1
                    )
                    target_key = (target_position, observation.variable_index)
                    if target_key in processed_target_positions:
                        continue
                    processed_target_positions.add(target_key)
                detector = detectors.get(observation.variable_name)
                if detector is None:
                    parameters = {
                        **config["drift"]["parameters"],
                        **config["drift"].get("variable_parameters", {}).get(observation.variable_name, {}),
                    }
                    detector = _build_detector(config, parameters)
                    detectors[observation.variable_name] = detector
                update = detector.update(observation.value)
                records.append(
                    DriftRecord(
                        detector=config["drift"]["name"],
                        source=source,
                        index=event.index,
                        available_at=observation.available_at,
                        raw_signal_index=raw_signal_index(
                            config["data"], event.index, source, observation.horizon_step
                        ),
                        variable_name=observation.variable_name,
                        variable_index=observation.variable_index,
                        horizon_step=observation.horizon_step,
                        value=update.value,
                        detected=update.detected,
                    )
                )
    return records


def run_experiment_with_drift(
    config: dict[str, Any],
) -> tuple[OnlineRun, list[int], list[DriftRecord]]:
    """Run one experiment and preserve every detector update for documentation."""

    run = run_forecast(config)
    detection_started = perf_counter()
    records = collect_drift_records(config, run)
    drift_detection_seconds = perf_counter() - detection_started
    metrics = replace(
        run.metrics,
        drift_detection_seconds=drift_detection_seconds,
        total_seconds=(run.metrics.total_seconds or 0.0) + drift_detection_seconds,
    )
    run = replace(run, metrics=metrics)
    return run, [record.index for record in records if record.detected], records


def run_experiment(config: dict[str, Any]) -> tuple[OnlineRun, list[int]]:
    """Run one configured prequential experiment and return detected drift indices."""

    run, drift_indices, _ = run_experiment_with_drift(config)
    return run, drift_indices


def main(argv: Sequence[str] | None = None) -> None:
    """Load a YAML configuration, execute it, and print concise metrics."""

    parser = argparse.ArgumentParser(description="Run an online time-series forecasting experiment")
    parser.add_argument("--config", default="config.yaml", help="path to the YAML configuration")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    run, drift_indices, drift_records = run_experiment_with_drift(config)
    document_directory = write_experiment_documents(
        config["output"]["directory"],
        config["output"].get("run_name"),
        config,
        run,
        drift_indices,
        drift_records,
    )
    print(
        f"forecasts={run.metrics.forecasts_emitted} "
        f"feedback={run.metrics.feedback_received} "
        f"mae={run.metrics.mae} mse={run.metrics.mse}"
    )
    if drift_indices:
        print(f"drift_indices={drift_indices}")
    print(f"documents={document_directory}")


if __name__ == "__main__":
    main()

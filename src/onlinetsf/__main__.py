# -*- coding: utf-8 -*-
"""Command-line entry point for a configured online forecasting experiment."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from time import perf_counter
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .config import load_config
from .data import load_benchmark_dataset
from .documentation import write_experiment_documents
from .drift import ADWINDetector, DriftRecord, KSWINDetector, PageHinkleyDetector
from .forecasting import (
    LinearForecastBackbone,
    LSTMForecastBackbone,
    PatchTSTForecastBackbone,
    TCNForecastBackbone,
    TimeTCNForecastBackbone,
)
from .methods import FSNetMethod, FSNetTCN, OGDMethod, OneNetEnsemble, OneNetMethod
from .online import OnlineExecutor, OnlineRun


def _build_backbone(config: dict[str, Any], num_features: int, num_targets: int, target_indices: Sequence[int]):
    data = config["data"]
    forecasting = config["forecasting"]
    options = forecasting["parameters"]
    dimensions = {
        "context_length": data["context_length"],
        "num_features": num_features,
        "horizon": data["horizon"],
        "num_targets": num_targets,
    }
    if forecasting["backbone"] == "linear":
        return LinearForecastBackbone(**dimensions, **options)
    if forecasting["backbone"] == "lstm":
        return LSTMForecastBackbone(**dimensions, **options)
    if forecasting["backbone"] == "tcn":
        return TCNForecastBackbone(**dimensions, **options)
    if forecasting["backbone"] == "patchtst":
        return PatchTSTForecastBackbone(**dimensions, target_indices=target_indices, **options)
    if forecasting["backbone"] == "onenet_tcn":
        cross_time = TimeTCNForecastBackbone(
            **dimensions, target_indices=target_indices, **options
        )
        cross_variable = TCNForecastBackbone(**dimensions, **options)
        return OneNetEnsemble(
            cross_time,
            cross_variable,
            horizon=data["horizon"],
            num_targets=num_targets,
            decision_hidden=config["method"].get("decision_hidden", 32),
            decision_dropout=config["method"].get("decision_dropout", 0.1),
        )
    return FSNetTCN(**dimensions, **options)


def _build_method(config: dict[str, Any], model: nn.Module):
    method = config["method"]
    device = config["online"].get("device")
    learning_rate = method.get("learning_rate")
    if method["name"] == "onenet":
        assert isinstance(model, OneNetEnsemble)
        return OneNetMethod(
            model,
            learning_rate=learning_rate,
            weight_learning_rate=method.get("weight_learning_rate", 1e-3),
            decision_learning_rate=method.get("decision_learning_rate", 1e-3),
            n_inner=method.get("n_inner", 1),
            device=device,
        )
    if method["name"] == "fsnet":
        assert isinstance(model, FSNetTCN)
        return FSNetMethod(
            model,
            learning_rate=learning_rate,
            n_inner=method.get("n_inner", 1),
            device=device,
        )

    if learning_rate is None:
        return OGDMethod(model, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return OGDMethod(
        model,
        optimizer=optimizer,
        loss_fn=nn.MSELoss(),
        update_steps=method.get("update_steps", 1),
        device=device,
    )


def _run_offline_training(
    dataset, method: OGDMethod | FSNetMethod | OneNetMethod, offline: dict[str, Any]
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


def _build_detector(config: dict[str, Any]):
    drift = config["drift"]
    options = drift["parameters"]
    if drift["name"] == "none":
        return None
    if drift["name"] == "page_hinkley":
        return PageHinkleyDetector(**options)
    if drift["name"] == "adwin":
        return ADWINDetector(**options)
    return KSWINDetector(**options)


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
    )
    data["target_names"] = list(dataset.target_names)
    model = _build_backbone(config, dataset.num_features, dataset.num_targets, dataset.target_indices)
    method = _build_method(config, model)
    setup_seconds = perf_counter() - experiment_started
    offline_started = perf_counter()
    offline_stop = _run_offline_training(dataset, method, config["offline"])
    offline_training_seconds = perf_counter() - offline_started
    online = config["online"]
    executor = OnlineExecutor(
        method,
        feedback_delay=online["feedback_delay"],
        keep_predictions=(
            online.get("keep_predictions", False)
            or config["output"]["write_per_value_errors"]
        ),
    )
    online_started = perf_counter()
    run = executor.run_dataset(
        dataset,
        start=max(offline_stop, online.get("start", 0)),
        stop=online.get("stop"),
    )
    online_evaluation_seconds = perf_counter() - online_started
    metrics = replace(
        run.metrics,
        setup_seconds=setup_seconds,
        offline_training_seconds=offline_training_seconds,
        online_evaluation_seconds=online_evaluation_seconds,
        total_seconds=perf_counter() - experiment_started,
    )
    return replace(run, metrics=metrics)


def collect_drift_records(config: dict[str, Any], run: OnlineRun) -> list[DriftRecord]:
    """Apply the selected detector to every recorded forecasting feedback event."""

    detector = _build_detector(config)
    records: list[DriftRecord] = []
    if detector is not None:
        signal_name = config["drift"].get("signal", "mae")
        for event in run.events:
            signal = getattr(event, signal_name)
            if config["drift"]["name"] == "adwin":
                signal = min(signal / config["drift"]["scale"], 1.0)
            update = detector.update(signal)
            records.append(
                DriftRecord(
                    detector=config["drift"]["name"],
                    signal=signal_name,
                    index=event.index,
                    available_at=event.available_at,
                    value=update.value,
                    mean=update.mean,
                    score=update.score,
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

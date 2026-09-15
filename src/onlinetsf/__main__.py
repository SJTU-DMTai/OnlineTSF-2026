# -*- coding: utf-8 -*-
"""Command-line entry point for a configured online forecasting experiment."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from .config import load_config
from .data import load_benchmark_dataset
from .documentation import write_experiment_documents
from .drift import ADWINDetector, KSWINDetector, PageHinkleyDetector
from .forecasting import LinearForecastBackbone, PatchTSTForecastBackbone, TCNForecastBackbone
from .methods import FSNetMethod, FSNetTCN, OGDMethod
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
    if forecasting["backbone"] == "tcn":
        return TCNForecastBackbone(**dimensions, **options)
    if forecasting["backbone"] == "patchtst":
        return PatchTSTForecastBackbone(**dimensions, target_indices=target_indices, **options)
    return FSNetTCN(**dimensions, **options)


def _build_method(config: dict[str, Any], model: nn.Module):
    method = config["method"]
    device = config["online"].get("device")
    learning_rate = method.get("learning_rate")
    if method["name"] == "fsnet":
        assert isinstance(model, FSNetTCN)
        return FSNetMethod(model, learning_rate=learning_rate, device=device)

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


def run_experiment(config: dict[str, Any]) -> tuple[OnlineRun, list[int]]:
    """Run one configured prequential experiment and return detected drift indices."""

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
    model = _build_backbone(config, dataset.num_features, dataset.num_targets, dataset.target_indices)
    method = _build_method(config, model)
    online = config["online"]
    executor = OnlineExecutor(
        method,
        feedback_delay=online["feedback_delay"],
        keep_predictions=online.get("keep_predictions", False),
    )
    run = executor.run_dataset(dataset, start=online.get("start", 0), stop=online.get("stop"))

    detector = _build_detector(config)
    drift_indices: list[int] = []
    if detector is not None:
        signal_name = config["drift"].get("signal", "mae")
        for event in run.events:
            signal = getattr(event, signal_name)
            if config["drift"]["name"] == "adwin":
                signal = min(signal / config["drift"]["scale"], 1.0)
            if detector.update(signal).detected:
                drift_indices.append(event.index)
    return run, drift_indices


def main(argv: Sequence[str] | None = None) -> None:
    """Load a YAML configuration, execute it, and print concise metrics."""

    parser = argparse.ArgumentParser(description="Run an online time-series forecasting experiment")
    parser.add_argument("--config", default="config.yaml", help="path to the YAML configuration")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    run, drift_indices = run_experiment(config)
    document_directory = write_experiment_documents(
        config["output"]["directory"],
        config["output"].get("run_name"),
        config,
        run,
        drift_indices,
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

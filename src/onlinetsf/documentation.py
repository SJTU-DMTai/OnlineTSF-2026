# -*- coding: utf-8 -*-
"""Writing experiment configuration, online-training logs, and result reports."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .online import OnlineRun


def _run_directory(output_directory: str | Path, run_name: str | None) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    prefix = "run" if run_name is None else "".join(
        character if character.isalnum() or character in "-_" else "-" for character in run_name
    ).strip("-") or "run"
    directory = Path(output_directory) / f"{prefix}-{timestamp}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _format_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.6f}"


def write_experiment_documents(
    output_directory: str | Path,
    run_name: str | None,
    config: dict[str, Any],
    run: OnlineRun,
    drift_indices: list[int],
) -> Path:
    """Write the resolved config, online feedback log, and English result report."""

    directory = _run_directory(output_directory, run_name)
    with (directory / "config.txt").open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    with (directory / "online_training.log").open("w", encoding="utf-8", newline="\n") as handle:
        for event in run.events:
            loss = "N/A" if event.adaptation_loss is None else f"{event.adaptation_loss:.6f}"
            handle.write(
                f"feedback index={event.index} available_at={event.available_at} "
                f"observed_values={event.observed_values} mae={event.mae:.6f} "
                f"mse={event.mse:.6f} adaptation_loss={loss}\n"
            )
        handle.write(
            f"summary forecasts={run.metrics.forecasts_emitted} "
            f"feedback={run.metrics.feedback_received} "
            f"mae={_format_metric(run.metrics.mae)} "
            f"mse={_format_metric(run.metrics.mse)}\n"
        )

    data = config["data"]
    forecasting = config["forecasting"]
    method = config["method"]
    drift = config["drift"]
    with (directory / "results.txt").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("ONLINE TIME-SERIES FORECASTING RESULT\n")
        handle.write("=====================================\n\n")
        handle.write("Experiment\n")
        handle.write(f"Dataset: {data['name']}\n")
        handle.write(f"Backbone: {forecasting['backbone']}\n")
        handle.write(f"Method: {method['name']}\n")
        handle.write(f"Drift detector: {drift['name']}\n")
        handle.write(f"Feedback delay: {config['online']['feedback_delay']}\n\n")
        handle.write("Metrics\n")
        handle.write(f"Forecasts emitted: {run.metrics.forecasts_emitted}\n")
        handle.write(f"Feedback events: {run.metrics.feedback_received}\n")
        handle.write(f"Target values scored: {run.metrics.values_scored}\n")
        handle.write(f"Prequential MAE: {_format_metric(run.metrics.mae)}\n")
        handle.write(f"Prequential MSE: {_format_metric(run.metrics.mse)}\n")
        handle.write(f"Adaptation steps: {run.metrics.adaptation_steps}\n")
        handle.write(f"Mean adaptation loss: {_format_metric(run.metrics.mean_adaptation_loss)}\n\n")
        handle.write("Drift Events\n")
        if drift_indices:
            for index in drift_indices:
                handle.write(f"Detected at forecast index {index}.\n")
        else:
            handle.write("No drift events were detected.\n")
        handle.write("\nThe complete resolved configuration is saved in config.txt; "
                     "per-feedback online updates are saved in online_training.log.\n")
    return directory

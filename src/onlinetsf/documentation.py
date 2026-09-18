# -*- coding: utf-8 -*-
"""Writing experiment configuration, online-training logs, and result reports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import yaml

from .drift import DriftRecord
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
    drift_records: Sequence[DriftRecord] = (),
) -> Path:
    """Write the resolved config, per-step errors, detector trace, and result report."""

    directory = _run_directory(output_directory, run_name)
    with (directory / "config.txt").open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    with (directory / "online_training.log").open("w", encoding="utf-8", newline="\n") as handle:
        for event in run.events:
            loss = "N/A" if event.adaptation_loss is None else f"{event.adaptation_loss:.6f}"
            handle.write(
                f"feedback index={event.index} available_at={event.available_at} "
                f"observed_values={event.observed_values} mae={event.mae:.6f} "
                f"mse={event.mse:.6f} cumulative_mae={event.cumulative_mae:.6f} "
                f"cumulative_mse={event.cumulative_mse:.6f} adaptation_loss={loss}\n"
            )
        handle.write(
            f"summary forecasts={run.metrics.forecasts_emitted} "
            f"feedback={run.metrics.feedback_received} "
            f"mae={_format_metric(run.metrics.mae)} "
            f"mse={_format_metric(run.metrics.mse)}\n"
        )

    with (directory / "forecast_values.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        target_names = config["data"].get("target_names", ())
        writer.writerow(
            (
                "forecast_index",
                "feedback_available_at",
                "horizon_step",
                "target_position",
                "target_name",
                "prediction",
                "target",
                "absolute_error",
                "squared_error",
            )
        )
        for event in run.events:
            if event.prediction is None or event.target is None or event.observed_mask is None:
                continue
            for horizon_index in range(event.prediction.shape[0]):
                for target_index in range(event.prediction.shape[1]):
                    if not event.observed_mask[horizon_index, target_index].item():
                        continue
                    prediction = event.prediction[horizon_index, target_index].item()
                    target = event.target[horizon_index, target_index].item()
                    error = prediction - target
                    writer.writerow(
                        (
                            event.index,
                            event.available_at,
                            horizon_index + 1,
                            target_index,
                            target_names[target_index],
                            prediction,
                            target,
                            abs(error),
                            error * error,
                        )
                    )

    with (directory / "drift_trace.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "detector",
                "source",
                "forecast_index",
                "available_at",
                "variable_name",
                "variable_index",
                "horizon_step",
                "value",
                "detected",
            )
        )
        for record in drift_records:
            writer.writerow(
                (
                    record.detector,
                    record.source,
                    record.index,
                    record.available_at,
                    record.variable_name,
                    record.variable_index,
                    record.horizon_step,
                    record.value,
                    record.detected,
                )
            )

    data = config["data"]
    forecasting = config["forecasting"]
    method = config["method"]
    drift = config["drift"]
    detector_label = ", ".join(config.get("batch_detectors", (drift["name"],)))
    with (directory / "results.txt").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("ONLINE TIME-SERIES FORECASTING RESULT\n")
        handle.write("=====================================\n\n")
        handle.write("Experiment\n")
        handle.write(f"Dataset: {data['name']}\n")
        handle.write(f"Backbone: {forecasting['backbone']}\n")
        handle.write(f"Method: {method['name']}\n")
        handle.write(f"Drift detector: {detector_label}\n")
        handle.write(f"Feedback delay: {config['online']['feedback_delay']}\n\n")
        handle.write("Metrics\n")
        handle.write(f"Forecasts emitted: {run.metrics.forecasts_emitted}\n")
        handle.write(f"Feedback events: {run.metrics.feedback_received}\n")
        handle.write(f"Target values scored: {run.metrics.values_scored}\n")
        handle.write(f"Prequential MAE: {_format_metric(run.metrics.mae)}\n")
        handle.write(f"Prequential MSE: {_format_metric(run.metrics.mse)}\n")
        handle.write(f"Adaptation steps: {run.metrics.adaptation_steps}\n")
        handle.write(f"Mean adaptation loss: {_format_metric(run.metrics.mean_adaptation_loss)}\n\n")
        handle.write("Runtime\n")
        handle.write(f"Setup: {_format_metric(run.metrics.setup_seconds)} seconds\n")
        handle.write(f"Offline training: {_format_metric(run.metrics.offline_training_seconds)} seconds\n")
        handle.write(f"Online evaluation: {_format_metric(run.metrics.online_evaluation_seconds)} seconds\n")
        handle.write(f"Drift detection: {_format_metric(run.metrics.drift_detection_seconds)} seconds\n")
        handle.write(f"Total execution: {_format_metric(run.metrics.total_seconds)} seconds\n\n")
        handle.write("Drift Events\n")
        detected_records = [record for record in drift_records if record.detected]
        if detected_records:
            for record in detected_records:
                handle.write(
                    f"{record.detector} detected at forecast index {record.index} "
                    f"(feedback available at {record.available_at}).\n"
                )
        elif drift_indices:
            for index in drift_indices:
                handle.write(f"Detected at forecast index {index}.\n")
        else:
            handle.write("No drift events were detected.\n")
        handle.write("\nThe complete resolved configuration is saved in config.txt; "
                     "per-feedback online updates are saved in online_training.log.\n")
        handle.write("forecast_values.csv contains one row per observed horizon and target value; "
                     "drift_trace.csv contains every detector update.\n")
    return directory

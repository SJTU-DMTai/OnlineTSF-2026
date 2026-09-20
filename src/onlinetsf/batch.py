# -*- coding: utf-8 -*-
"""Run a strategy and detector comparison while retaining per-step results."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Sequence

import yaml

from .__main__ import collect_drift_records, run_forecast
from .config import load_config
from .documentation import write_experiment_documents


def main(argv: Sequence[str] | None = None) -> None:
    """Run every requested strategy, seed, and detector combination."""

    parser = argparse.ArgumentParser(description="Run a batch of online forecasting experiments")
    parser.add_argument("--config", default="config.yaml", help="base YAML configuration path")
    parser.add_argument("--strategies", nargs="+", required=True, help="strategy profile names")
    parser.add_argument("--detectors", nargs="+", required=True, help="detector profile names")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0], help="random seeds")
    parser.add_argument("--output", help="directory for the batch; defaults to output.directory")
    args = parser.parse_args(argv)

    base_config = load_config(args.config)
    output_directory = Path(args.output) if args.output else Path(base_config["output"]["directory"])
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    batch_directory = output_directory / f"batch-{timestamp}"
    batch_directory.mkdir(parents=True, exist_ok=False)

    manifest = {
        "base_config": str(Path(args.config).resolve()),
        "strategies": args.strategies,
        "detectors": args.detectors,
        "seeds": args.seeds,
    }
    with (batch_directory / "manifest.yaml").open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False)

    with (
        (batch_directory / "forecast_steps.csv").open("w", encoding="utf-8", newline="") as step_handle,
        (batch_directory / "forecast_values.csv").open("w", encoding="utf-8", newline="") as value_handle,
        (batch_directory / "drift_trace.csv").open("w", encoding="utf-8", newline="") as trace_handle,
        (batch_directory / "drift_events.csv").open("w", encoding="utf-8", newline="") as event_handle,
        (batch_directory / "summary.csv").open("w", encoding="utf-8", newline="") as summary_handle,
    ):
        step_writer = csv.writer(step_handle)
        value_writer = csv.writer(value_handle)
        trace_writer = csv.writer(trace_handle)
        event_writer = csv.writer(event_handle)
        summary_writer = csv.writer(summary_handle)
        step_writer.writerow(
            (
                "strategy",
                "seed",
                "forecast_index",
                "feedback_available_at",
                "raw_context_start_index",
                "raw_context_end_index",
                "raw_target_start_index",
                "raw_target_end_exclusive",
                "observed_values",
                "step_mae",
                "step_mse",
                "cumulative_mae",
                "cumulative_mse",
                "adaptation_loss",
            )
        )
        value_writer.writerow(
            (
                "strategy",
                "seed",
                "forecast_index",
                "feedback_available_at",
                "raw_context_start_index",
                "raw_context_end_index",
                "raw_target_index",
                "horizon_step",
                "target_position",
                "target_name",
                "prediction",
                "target",
                "absolute_error",
                "squared_error",
            )
        )
        trace_writer.writerow(
            (
                "strategy",
                "seed",
                "detector",
                "source",
                "forecast_index",
                "available_at",
                "raw_signal_index",
                "variable_name",
                "variable_index",
                "horizon_step",
                "value",
                "detected",
            )
        )
        event_writer.writerow(
            (
                "strategy",
                "seed",
                "detector",
                "source",
                "forecast_index",
                "available_at",
                "raw_signal_index",
                "variable_name",
                "variable_index",
                "horizon_step",
                "value",
            )
        )
        summary_writer.writerow(
            (
                "strategy",
                "seed",
                "forecasts_emitted",
                "feedback_received",
                "values_scored",
                "mae",
                "mse",
                "adaptation_steps",
                "mean_adaptation_loss",
                "setup_seconds",
                "offline_training_seconds",
                "online_evaluation_seconds",
                "drift_detection_seconds",
                "total_seconds",
                "documents",
            )
        )

        for strategy in args.strategies:
            for seed in args.seeds:
                config = load_config(
                    args.config,
                    strategy_name=strategy,
                    detector_name=args.detectors[0],
                )
                config["seed"] = seed
                config["batch_detectors"] = args.detectors
                config["output"]["directory"] = str(batch_directory / "jobs")
                config["output"]["run_name"] = f"{strategy}-seed-{seed}"
                run = run_forecast(config)

                detection_started = perf_counter()
                drift_records = []
                for detector_name in args.detectors:
                    detector_config = load_config(
                        args.config,
                        strategy_name=strategy,
                        detector_name=detector_name,
                    )
                    detector_config["data"]["feature_names"] = config["data"]["feature_names"]
                    detector_config["data"]["target_names"] = config["data"]["target_names"]
                    drift_records.extend(collect_drift_records(detector_config, run))
                drift_detection_seconds = perf_counter() - detection_started
                run = replace(
                    run,
                    metrics=replace(
                        run.metrics,
                        drift_detection_seconds=drift_detection_seconds,
                        total_seconds=(run.metrics.total_seconds or 0.0) + drift_detection_seconds,
                    ),
                )
                drift_indices = [record.index for record in drift_records if record.detected]
                document_directory = write_experiment_documents(
                    config["output"]["directory"],
                    config["output"]["run_name"],
                    config,
                    run,
                    drift_indices,
                    drift_records,
                )

                for feedback in run.events:
                    raw_context_start = feedback.index * config["data"].get("stride", 1)
                    raw_target_start = raw_context_start + config["data"]["context_length"]
                    step_writer.writerow(
                        (
                            strategy,
                            seed,
                            feedback.index,
                            feedback.available_at,
                            raw_context_start,
                            raw_context_start + config["data"]["context_length"] - 1,
                            raw_target_start,
                            raw_target_start + config["data"]["horizon"],
                            feedback.observed_values,
                            feedback.mae,
                            feedback.mse,
                            feedback.cumulative_mae,
                            feedback.cumulative_mse,
                            feedback.adaptation_loss,
                        )
                    )
                with (document_directory / "forecast_values.csv").open(
                    "r", encoding="utf-8", newline=""
                ) as handle:
                    for row in csv.reader(handle):
                        if row[0] == "forecast_index":
                            continue
                        value_writer.writerow((strategy, seed, *row))
                for record in drift_records:
                    trace_writer.writerow(
                        (
                            strategy,
                            seed,
                            record.detector,
                            record.source,
                            record.index,
                            record.available_at,
                            record.raw_signal_index,
                            record.variable_name,
                            record.variable_index,
                            record.horizon_step,
                            record.value,
                            record.detected,
                        )
                    )
                    if record.detected:
                        event_writer.writerow(
                            (
                                strategy,
                                seed,
                                record.detector,
                                record.source,
                                record.index,
                                record.available_at,
                                record.raw_signal_index,
                                record.variable_name,
                                record.variable_index,
                                record.horizon_step,
                                record.value,
                            )
                        )
                metrics = run.metrics
                summary_writer.writerow(
                    (
                        strategy,
                        seed,
                        metrics.forecasts_emitted,
                        metrics.feedback_received,
                        metrics.values_scored,
                        metrics.mae,
                        metrics.mse,
                        metrics.adaptation_steps,
                        metrics.mean_adaptation_loss,
                        metrics.setup_seconds,
                        metrics.offline_training_seconds,
                        metrics.online_evaluation_seconds,
                        metrics.drift_detection_seconds,
                        metrics.total_seconds,
                        document_directory,
                    )
                )
                print(
                    f"strategy={strategy} seed={seed} "
                    f"mae={metrics.mae} mse={metrics.mse} documents={document_directory}"
                )

    print(f"batch_documents={batch_directory}")


if __name__ == "__main__":
    main()

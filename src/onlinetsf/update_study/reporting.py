"""Streaming forecast logs and target-aligned phase/horizon summaries."""

from __future__ import annotations

import csv
from pathlib import Path

import torch

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class Reporter:
    """Errors use immutable emitted forecasts; phase attribution uses raw target rows."""

    def __init__(self, directory: Path, stream, first_origin: int, save_predictions: bool = False):
        directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory
        self.stream = stream
        self.first_origin = first_origin
        self.totals = {}
        self.forecasts = 0
        self.save_predictions = save_predictions
        self.prediction_buffer = []
        self.prediction_chunk = 0
        self.forecast_file = (directory / "forecasts.csv").open("w", encoding="utf-8", newline="")
        self.update_file = (directory / "updates.csv").open("w", encoding="utf-8", newline="")
        self.forecast_writer = csv.DictWriter(self.forecast_file, fieldnames=[
            "forecast_index", "origin_raw_index", "target_start_index", "target_end_index_exclusive",
            "feedback_raw_index", "mae", "mse", "time_expert_mae", "variable_expert_mae",
        ])
        self.update_writer = csv.DictWriter(self.update_file, fieldnames=[
            "forecast_index", "available_at", "feedback_raw_index", "feedback_phase", "groups",
            "gradient_steps", "egd_steps", "adaptation_loss", "seconds",
        ])
        self.forecast_writer.writeheader()
        self.update_writer.writeheader()

    def add(self, records: list[dict]) -> None:
        length = self.stream.dataset.context_length
        horizon = self.stream.dataset.horizon
        for record in records:
            # Old pending forecasts still train the fork but are never scored as its outcomes.
            if record["update"] is not None:
                update = dict(record["update"])
                clock = update["available_at"] + length - 1
                update.update(feedback_raw_index=clock, feedback_phase=self.stream.phases[clock])
                self.update_writer.writerow(update)
            if record["index"] < self.first_origin:
                continue
            raw = record["index"] + length
            self.forecast_writer.writerow({"forecast_index": record["index"], "origin_raw_index": raw - 1,
                                           "target_start_index": raw, "target_end_index_exclusive": raw + horizon,
                                           "feedback_raw_index": record["available_at"] + length - 1,
                                           "mae": record["mae"], "mse": record["mse"],
                                           "time_expert_mae": record["expert_errors"][0] if record["expert_errors"] else None,
                                           "variable_expert_mae": record["expert_errors"][1] if record["expert_errors"] else None})
            self.forecasts += 1
            if self.save_predictions:
                self.prediction_buffer.append((record["index"], record["prediction"], record["target"].clone()))
                if len(self.prediction_buffer) == 256:
                    self.flush_predictions()
            for step, error in enumerate(record["error"]):
                phase = self.stream.phases[raw + step]
                absolute, squared, count = error.abs().sum().item(), error.square().sum().item(), error.numel()
                for key in (("all", 0), (phase, 0), ("all", step + 1), (phase, step + 1)):
                    totals = self.totals.setdefault(key, [0.0, 0.0, 0])
                    totals[0] += absolute
                    totals[1] += squared
                    totals[2] += count

    def flush_predictions(self) -> None:
        if self.prediction_buffer:
            torch.save({"indices": [row[0] for row in self.prediction_buffer],
                        "predictions": torch.stack([row[1] for row in self.prediction_buffer]),
                        "targets": torch.stack([row[2] for row in self.prediction_buffer])},
                       self.directory / f"predictions_{self.prediction_chunk:05d}.pt")
            self.prediction_buffer.clear()
            self.prediction_chunk += 1

    def close(self, controlled, seconds: float) -> dict:
        self.forecast_file.close()
        self.update_file.close()
        self.flush_predictions()
        rows = [{"phase": phase, "horizon_step": step, "values": count,
                 "mae": absolute / count, "mse": squared / count}
                for (phase, step), (absolute, squared, count) in sorted(self.totals.items()) if count]
        write_csv(self.directory / "phase_metrics.csv", rows)
        absolute, squared, count = self.totals.get(("all", 0), (0.0, 0.0, 0))
        return {"forecasts": self.forecasts, "values": count, "mae": absolute / count if count else None,
                "mse": squared / count if count else None, "gradient_steps": controlled.gradient_steps,
                "egd_steps": controlled.egd_steps, "seconds": seconds}

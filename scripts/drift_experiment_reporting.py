# -*- coding: utf-8 -*-
"""Retrospective scoring and artifacts; never used to choose online actions."""

from __future__ import annotations

import json
import csv
from statistics import mean
from time import perf_counter

import torch

from onlinetsf.online import OnlineExecutor

from drift_experiment_data import aggregate_metrics, phase_at, write_json, write_rows
from drift_experiment_runtime import method_state


class ReplayReport:
    """Keep one run's compact traces and accumulate target-time phase statistics."""

    def __init__(self, stream, dataset, args):
        self.stream, self.dataset, self.args = stream, dataset, args
        self.steps, self.updates, self.states, self.values, self.components = [], [], [], [], []
        self.phase_sums = {}
        self.first_horizon = {}

    def emission(self, index, prediction, target, controlled):
        error = prediction.cpu() - target.cpu()
        if not torch.isfinite(error).all():
            raise FloatingPointError(f"nonfinite prediction error at origin {index}")
        first = index + self.dataset.context_length
        last = first + self.dataset.horizon - 1
        labels = self.stream["labels"]
        first_phase = phase_at(first, labels, self.args.pre_rows, self.args.post_rows)
        last_phase = phase_at(last, labels, self.args.pre_rows, self.args.post_rows)
        self.steps.append({"forecast_index": index, "raw_origin": first - 1,
                           "raw_target_start": first, "raw_target_end": last,
                           **first_phase, "last_target_phase": last_phase["phase"],
                           "last_target_drift_type": last_phase["drift_type"],
                           "context_crosses_drift": any(index < label["start_index"] < first for label in labels),
                           "target_crosses_drift": any(first < label["start_index"] <= last for label in labels),
                           "mae": error.abs().mean().item(), "mse": error.square().mean().item()})
        self.first_horizon[first] = error[0].abs().mean().item()
        self.states.append({"forecast_index": index, "raw_origin": first - 1,
                            "state": json.dumps(method_state(controlled.method))})
        for name, component in controlled.last_components.items():
            difference = component.cpu() - target.cpu()
            self.components.append({"forecast_index": index, "component": name,
                                    "mae": difference.abs().mean().item(), "mse": difference.square().mean().item()})
        for h in range(self.dataset.horizon):
            raw_index = first + h
            phase = phase_at(raw_index, labels, self.args.pre_rows, self.args.post_rows)
            for c, name in enumerate(self.dataset.target_names):
                absolute, squared = abs(error[h, c].item()), error[h, c].item() ** 2
                base = (phase["phase"], phase["mechanism"], phase["transition"], phase["drift_type"])
                for horizon_step, target_name in ((str(h + 1), name), ("all", "all")):
                    sums = self.phase_sums.setdefault(base + (horizon_step, target_name), [0, 0.0, 0.0])
                    sums[0] += 1
                    sums[1] += absolute
                    sums[2] += squared
                if self.args.write_values:
                    self.values.append({"forecast_index": index, "raw_target_index": raw_index,
                                        "horizon_step": h + 1, "target": name, **phase,
                                        "prediction": prediction[h, c].item(), "truth": target[h, c].item(),
                                        "absolute_error": absolute, "squared_error": squared})

    def feedback(self, controlled):
        for row in controlled.records:
            if row["eligible"] or row["optimizer_steps"] or row["terminal"]:
                phase = phase_at(row["raw_available_index"], self.stream["labels"], self.args.pre_rows, self.args.post_rows)
                self.updates.append({**row, **phase})
        controlled.records.clear()

    def phase_metrics(self):
        result = []
        for key, (count, absolute, squared) in sorted(self.phase_sums.items()):
            row = dict(zip(("phase", "mechanism", "transition", "drift_type", "horizon_step", "target"), key))
            row.update(values=count, mae=absolute / count, mse=squared / count)
            result.append(row)
        return result

    def event_metrics(self):
        """Recovery uses h=1 errors, a rolling 8-point mean held for 8 windows."""
        result = []
        errors = self.first_horizon
        if not errors:
            return result
        first, last = min(errors), max(errors)
        labels = self.stream["labels"]
        for position, label in enumerate(labels):
            onset, full = label["start_index"], label["new_concept_full_from"]
            next_start = labels[position + 1]["start_index"] if position + 1 < len(labels) else self.stream["manifest"]["rows"]
            previous_end = (labels[position - 1]["new_concept_full_from"] or labels[position - 1]["end_index_exclusive"]) if position else first
            pre_start = max(first, previous_end, onset - self.args.pre_rows)
            before = [errors[t] for t in range(pre_start, onset) if t in errors]
            pre_mae = mean(before) if before else None
            analysis_end = min(next_start, (full if full is not None else onset) + self.args.post_rows, last + 1)
            after = [errors[t] for t in range(max(first, onset), analysis_end) if t in errors]
            post = [(t, errors[t]) for t in range(max(first, full), analysis_end) if t in errors] if full is not None else []
            clean_pre = len(before) == self.args.pre_rows
            initially_degraded = None
            delay = None
            if clean_pre and len(post) >= 8 and full is not None and full >= first:
                initially_degraded = mean(value for _, value in post[:8]) > 1.2 * max(pre_mae, 1e-8)
                if initially_degraded:
                    streak = 0
                    for offset in range(len(post) - 7):
                        rolling = mean(value for _, value in post[offset:offset + 8])
                        streak = streak + 1 if rolling <= 1.1 * max(pre_mae, 1e-8) else 0
                        if streak == 8:
                            delay = post[offset - 7][0] - full
                            break
            opportunities = [row for row in self.updates if onset <= row["raw_available_index"] < analysis_end and not row["terminal"]]
            result.append({"event_id": label["event_id"], "mechanism": label["method"],
                           "transition": label["transition"], "drift_type": label["drift_type"],
                           "start_index": onset, "new_concept_full_from": full,
                           "pre_count": len(before), "post_count": len(post), "pre_mae": pre_mae,
                           "post_mae": mean(value for _, value in post) if post else None,
                           "event_cumulative_absolute_error": sum(after) if after else None,
                           "clean_pre_window": clean_pre, "initially_degraded": initially_degraded,
                           "recovery_delay": delay, "recovered": delay is not None if initially_degraded else None,
                           "unrecovered_at_end": bool(initially_degraded and delay is None),
                           "right_censored": full is not None and analysis_end < full + self.args.post_rows,
                           "recovery_applicable": full is not None and full < self.stream["manifest"]["rows"] and clean_pre and full >= first,
                           "optimizer_steps": sum(row["optimizer_steps"] for row in opportunities),
                           "selected_opportunities": sum(row["selected"] for row in opportunities)})
        return result

    def update_metrics(self):
        """Group optimization by its actual execution clock, separately from target errors."""
        groups = {}
        for state in self.states:
            phase = phase_at(state["raw_origin"], self.stream["labels"], self.args.pre_rows, self.args.post_rows)
            key = tuple(phase[name] for name in ("phase", "mechanism", "transition", "drift_type"))
            if key not in groups:
                groups[key] = {"ticks": 0, "opportunities": 0, "selected_opportunities": 0,
                               "gradient_events": 0, "optimizer_steps": 0, "optimizer_sample_uses": 0,
                               "feedback_seconds": 0.0, "updated_times": set()}
            groups[key]["ticks"] += 1
        for update in self.updates:
            if update["terminal"]:
                continue
            key = tuple(update[name] for name in ("phase", "mechanism", "transition", "drift_type"))
            record = groups[key]
            record["opportunities"] += update["eligible"]
            record["selected_opportunities"] += update["selected"]
            record["gradient_events"] += update["optimizer_steps"] > 0
            for name in ("optimizer_steps", "optimizer_sample_uses", "feedback_seconds"):
                record[name] += update[name]
            if update["optimizer_steps"]:
                record["updated_times"].add(update["available_at"])
        result = []
        for key, counts in sorted(groups.items()):
            record = dict(zip(("phase", "mechanism", "transition", "drift_type"), key))
            updated_times = counts.pop("updated_times")
            record.update(counts, updated_ticks=len(updated_times),
                          updated_ticks_per_1000=1000 * len(updated_times) / counts["ticks"],
                          optimizer_steps_per_1000=1000 * counts["optimizer_steps"] / counts["ticks"])
            result.append(record)
        return result


def run_replay(controlled, dataset, start, stop, report, executor=None):
    if executor is None:
        from onlinetsf.methods import DSOFMethod
        delay = list(range(1, dataset.horizon + 1)) if isinstance(controlled.method, DSOFMethod) else dataset.horizon
        executor = OnlineExecutor(controlled, feedback_delay=delay, keep_predictions=False)
    if controlled.method.device.type == "cuda":
        torch.cuda.synchronize(controlled.method.device)
    began = perf_counter()
    for index in range(start, stop):
        context, target = dataset[index]
        step = executor.step(index, context, target)
        report.emission(index, step.emission.prediction, target, controlled)
        report.feedback(controlled)
        # Counts are retained by the executor; past tensor events are not needed by later steps.
        executor._events.clear()
    controlled.training_enabled = False
    executor.flush()
    report.feedback(controlled)
    executor._events.clear()
    if controlled.method.device.type == "cuda":
        torch.cuda.synchronize(controlled.method.device)
    return {"forecasts": executor.metrics.forecasts_emitted, "mae": executor.metrics.mae, "mse": executor.metrics.mse,
            "opportunities": controlled.opportunity_count, "selected_opportunities": controlled.selected_count,
            "optimizer_steps": sum(row["optimizer_steps"] for row in report.updates),
            "optimizer_sample_uses": sum(row["optimizer_sample_uses"] for row in report.updates),
            "feedback_seconds": sum(row["feedback_seconds"] for row in report.updates),
            "replay_seconds_including_diagnostics": perf_counter() - began}


def save_run(directory, metadata, policy_name, report, summary):
    directory.mkdir(parents=True, exist_ok=False)
    for filename, rows in (("forecast_steps.csv", report.steps), ("update_trace.csv", report.updates),
                           ("method_state.csv", report.states), ("forecast_values.csv", report.values),
                           ("component_errors.csv", report.components)):
        write_rows(directory / filename, rows)
    event_rows = [{**metadata, "policy": policy_name, **row} for row in report.event_metrics()]
    write_rows(directory / "event_metrics.csv", event_rows)
    phase_rows = [{**metadata, "policy": policy_name, **row} for row in report.phase_metrics()]
    update_rows = [{**metadata, "policy": policy_name, **row} for row in report.update_metrics()]
    write_rows(directory / "phase_metrics.csv", phase_rows)
    write_rows(directory / "update_metrics.csv", update_rows)
    summary = {**metadata, "policy": policy_name, **summary}
    write_json(directory / "summary.json", summary)
    return summary, phase_rows


def finish_experiment(output, summaries, phase_rows):
    write_rows(output / "run_summary.csv", summaries)
    write_rows(output / "phase_metrics.csv", phase_rows)
    aggregate_metrics(summaries, output / "summary_by_family.csv", ("family", "strategy", "horizon", "policy"),
                      ("mae", "mse", "optimizer_steps", "selected_opportunities", "feedback_seconds"))
    aggregate_metrics(phase_rows, output / "summary_by_drift_type_phase.csv",
                      ("family", "strategy", "horizon", "policy", "drift_type", "phase", "horizon_step", "target"), ("mae", "mse"))
    update_rows = []
    update_fields = ("ticks", "opportunities", "selected_opportunities", "gradient_events", "optimizer_steps",
                     "optimizer_sample_uses", "feedback_seconds", "updated_ticks", "updated_ticks_per_1000", "optimizer_steps_per_1000")
    for path in sorted(output.glob("stream-*/*/*/*/update_metrics.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                for name in update_fields:
                    row[name] = float(row[name])
                update_rows.append(row)
    write_rows(output / "update_metrics.csv", update_rows)
    aggregate_metrics(update_rows, output / "updates_by_drift_type_phase.csv",
                      ("family", "strategy", "horizon", "policy", "drift_type", "phase"), update_fields)
    event_rows = []
    for path in sorted(output.glob("stream-*/*/*/*/event_metrics.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                for name in ("pre_mae", "post_mae", "event_cumulative_absolute_error", "recovery_delay"):
                    row[name] = float(row[name]) if row[name] else None
                for name in ("recovery_applicable", "initially_degraded", "recovered", "unrecovered_at_end", "right_censored"):
                    row[name] = row[name] == "True" if row[name] else None
                event_rows.append(row)
    write_rows(output / "event_metrics.csv", event_rows)
    eligible_recovery = [row for row in event_rows if row["recovery_applicable"] and row["initially_degraded"]]
    aggregate_metrics(eligible_recovery, output / "recovery_by_drift_type.csv",
                      ("family", "strategy", "horizon", "policy", "drift_type"),
                      ("recovered", "unrecovered_at_end", "right_censored", "recovery_delay"))
    print(f"output={output.resolve()}", flush=True)

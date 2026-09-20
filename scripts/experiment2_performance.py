# -*- coding: utf-8 -*-
"""Relate labeled drifts and alarms to changes in forecast error.

Run experiment1_detection.py first. Error before each drift is the reference.
Example: python scripts/experiment2_performance.py --detection runs/experiment1 --output runs/experiment2
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detection", type=Path, required=True, help="experiment 1 output directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pre-rows", type=int, default=32)
    parser.add_argument("--post-rows", type=int, default=32, help="rows after drift transition")
    parser.add_argument("--block-rows", type=int, default=16)
    parser.add_argument("--min-rows", type=int, default=8)
    parser.add_argument("--drop-ratio", type=float, default=0.20)
    parser.add_argument("--alarm-delay", type=int, default=0, help="extra rows allowed after a drop block")
    args = parser.parse_args()
    if min(args.pre_rows, args.post_rows, args.block_rows, args.min_rows) < 1 or args.drop_ratio < 0 or args.alarm_delay < 0:
        parser.error("row counts must be positive and drop-ratio must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)

    forecasts = read_rows(args.detection / "forecast_steps.csv")
    matches = read_rows(args.detection / "event_matches.csv")
    alarm_events = read_rows(args.detection / "alarm_events.csv")
    detection_summaries = read_rows(args.detection / "detection_summary.csv")
    steps_by_run: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in forecasts:
        steps_by_run[(row["stream_id"], row["strategy"])].append({
            "raw": int(row["target_raw_index"]), "mae": float(row["mae"]),
        })
    for steps in steps_by_run.values():
        steps.sort(key=lambda row: row["raw"])

    labels_by_stream: dict[str, dict[int, dict]] = defaultdict(dict)
    for row in matches:
        labels_by_stream[row["stream_id"]][int(row["event_id"])] = row
    event_performance: list[dict] = []
    aligned_errors: list[dict] = []
    performance_by_key: dict[tuple[str, str, int], dict] = {}
    for (stream_id, strategy), steps in steps_by_run.items():
        dataset_name = stream_id.split("/", 1)[0].removesuffix("-batch")
        labels = sorted(labels_by_stream[stream_id].values(), key=lambda row: int(row["start_index"]))
        first_raw = steps[0]["raw"]
        last_raw = steps[-1]["raw"]
        for position, label in enumerate(labels):
            event_id = int(label["event_id"])
            start = int(label["start_index"])
            end = int(label["end_index_exclusive"])
            previous_end = int(labels[position - 1]["end_index_exclusive"]) if position else first_raw
            next_start = int(labels[position + 1]["start_index"]) if position + 1 < len(labels) else last_raw + 1
            pre_start = max(start - args.pre_rows, previous_end, first_raw)
            post_end = min(end + args.post_rows, next_start, last_raw + 1)
            pre = [row["mae"] for row in steps if pre_start <= row["raw"] < start]
            post = [row["mae"] for row in steps if start <= row["raw"] < post_end]
            eligible = len(pre) >= args.min_rows and len(post) >= args.min_rows
            pre_mae = mean(pre) if pre else None
            post_mae = mean(post) if post else None
            increase = post_mae - pre_mae if eligible else None
            increase_ratio = increase / max(pre_mae, 1e-8) if eligible else None
            drop = increase_ratio > args.drop_ratio if eligible else None
            record = {
                "stream_id": stream_id, "dataset": dataset_name, "strategy": strategy,
                "event_id": event_id, "method": label["method"], "transition": label["transition"],
                "start_index": start, "end_index_exclusive": end,
                "target_affected": "OT" in json.loads(label["affected_variables"]),
                "pre_start": pre_start, "post_end_exclusive": post_end,
                "pre_count": len(pre), "post_count": len(post), "eligible": eligible,
                "pre_mae": pre_mae, "post_mae": post_mae,
                "mae_increase": increase, "increase_ratio": increase_ratio, "performance_drop": drop,
            }
            event_performance.append(record)
            performance_by_key[(stream_id, strategy, event_id)] = record
            for step in steps:
                if pre_start <= step["raw"] < post_end:
                    aligned_errors.append({
                        "stream_id": stream_id, "dataset": dataset_name, "strategy": strategy,
                        "event_id": event_id, "method": label["method"],
                        "raw_index": step["raw"], "relative_index": step["raw"] - start,
                        "phase": "pre" if step["raw"] < start else (
                            "transition" if step["raw"] < end else "after"
                        ),
                        "mae": step["mae"], "pre_mae": pre_mae,
                    })

    relationship_rows: list[dict] = []
    for match in matches:
        stream_id = match["stream_id"]
        strategies = [match["strategy"]] if match["strategy"] != "shared" else [
            strategy for candidate_stream, strategy in steps_by_run if candidate_stream == stream_id
        ]
        for strategy in strategies:
            performance = performance_by_key[(stream_id, strategy, int(match["event_id"]))]
            relationship_rows.append({
                "stream_id": stream_id, "dataset": match["dataset"], "strategy": strategy,
                "detector": match["detector"], "source": match["source"],
                "event_id": match["event_id"], "method": match["method"],
                "transition": match["transition"], "target_affected": performance["target_affected"],
                "performance_eligible": performance["eligible"],
                "detection_status": match["status"],
                "performance_drop": performance["performance_drop"],
                "alarm": match["status"] == "hit",
                "pre_mae": performance["pre_mae"], "post_mae": performance["post_mae"],
                "increase_ratio": performance["increase_ratio"],
                "alarm_delay": match["delay_from_start"],
            })

    relationship_summary: list[dict] = []
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in relationship_rows:
        if row["performance_eligible"] and row["detection_status"] in {"hit", "miss"}:
            groups[(row["dataset"], row["strategy"], row["detector"], row["source"], row["method"])].append(row)
    for (dataset_name, strategy, detector, source, method), rows in sorted(groups.items()):
        drops = sum(row["performance_drop"] for row in rows)
        alarms = sum(row["alarm"] for row in rows)
        both = sum(row["performance_drop"] and row["alarm"] for row in rows)
        relationship_summary.append({
            "dataset": dataset_name, "strategy": strategy, "detector": detector,
            "source": source, "method": method, "events": len(rows),
            "performance_drops": drops, "detected_events": alarms, "drop_and_alarm": both,
            "drop_given_drift": drops / len(rows),
            "alarm_given_drop": both / drops if drops else None,
            "drop_given_alarm": both / alarms if alarms else None,
            "alarm_without_drop": alarms - both,
            "drop_without_alarm": drops - both,
        })

    collapsed_alarms: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    for row in alarm_events:
        if row["collapsed_alarm"] == "True":
            collapsed_alarms[(row["stream_id"], row["strategy"], row["detector"])].add(
                int(row["raw_signal_index"])
            )
    first_evaluable = {
        (row["stream_id"], row["strategy"], row["detector"]): int(row["first_evaluable_raw_index"])
        for row in detection_summaries
    }
    block_rows: list[dict] = []
    for (stream_id, strategy), steps in steps_by_run.items():
        dataset_name = stream_id.split("/", 1)[0].removesuffix("-batch")
        labels = list(labels_by_stream[stream_id].values())
        detector_scopes = {
            (row["detector"], row["strategy"])
            for row in matches if row["stream_id"] == stream_id
            and row["strategy"] in {strategy, "shared"}
        }
        for offset in range(args.block_rows, len(steps) - args.block_rows + 1, args.block_rows):
            previous = steps[offset - args.block_rows:offset]
            current = steps[offset:offset + args.block_rows]
            pre_mae = mean(row["mae"] for row in previous)
            current_mae = mean(row["mae"] for row in current)
            start = current[0]["raw"]
            end = current[-1]["raw"] + 1
            transition_overlap = any(
                int(label["start_index"]) < end and int(label["end_index_exclusive"]) > start
                for label in labels
            )
            for detector, scope in sorted(detector_scopes):
                if start < first_evaluable[(stream_id, scope, detector)]:
                    continue
                alarm = any(
                    start <= value < end + args.alarm_delay
                    for value in collapsed_alarms[(stream_id, scope, detector)]
                )
                block_rows.append({
                    "stream_id": stream_id, "dataset": dataset_name, "strategy": strategy,
                    "detector": detector, "block_start": start, "block_end_exclusive": end,
                    "previous_mae": pre_mae, "current_mae": current_mae,
                    "increase_ratio": (current_mae - pre_mae) / max(pre_mae, 1e-8),
                    "performance_drop": current_mae > pre_mae * (1 + args.drop_ratio),
                    "transition_overlap": transition_overlap, "alarm": alarm,
                })

    block_summary: list[dict] = []
    block_groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in block_rows:
        block_groups[(row["dataset"], row["strategy"], row["detector"])].append(row)
    for (dataset_name, strategy, detector), rows in sorted(block_groups.items()):
        drops = sum(row["performance_drop"] for row in rows)
        alarms = sum(row["alarm"] for row in rows)
        both = sum(row["performance_drop"] and row["alarm"] for row in rows)
        block_summary.append({
            "dataset": dataset_name, "strategy": strategy, "detector": detector,
            "blocks": len(rows), "drop_blocks": drops, "alarm_blocks": alarms,
            "drop_and_alarm": both,
            "alarm_given_drop": both / drops if drops else None,
            "drop_given_alarm": both / alarms if alarms else None,
        })

    write_rows(args.output / "event_performance.csv", [
        "stream_id", "dataset", "strategy", "event_id", "method", "transition",
        "start_index", "end_index_exclusive", "target_affected", "pre_start", "post_end_exclusive",
        "pre_count", "post_count", "eligible", "pre_mae", "post_mae", "mae_increase",
        "increase_ratio", "performance_drop",
    ], event_performance)
    write_rows(args.output / "aligned_errors.csv", [
        "stream_id", "dataset", "strategy", "event_id", "method", "raw_index",
        "relative_index", "phase", "mae", "pre_mae",
    ], aligned_errors)
    write_rows(args.output / "relationship_events.csv", [
        "stream_id", "dataset", "strategy", "detector", "source", "event_id", "method",
        "transition", "target_affected", "performance_eligible", "detection_status",
        "performance_drop", "alarm", "pre_mae", "post_mae", "increase_ratio", "alarm_delay",
    ], relationship_rows)
    write_rows(args.output / "relationship_summary.csv", [
        "dataset", "strategy", "detector", "source", "method", "events", "performance_drops",
        "detected_events", "drop_and_alarm", "drop_given_drift", "alarm_given_drop",
        "drop_given_alarm", "alarm_without_drop", "drop_without_alarm",
    ], relationship_summary)
    write_rows(args.output / "performance_blocks.csv", [
        "stream_id", "dataset", "strategy", "detector", "block_start", "block_end_exclusive",
        "previous_mae", "current_mae", "increase_ratio", "performance_drop", "transition_overlap", "alarm",
    ], block_rows)
    write_rows(args.output / "block_summary.csv", [
        "dataset", "strategy", "detector", "blocks", "drop_blocks", "alarm_blocks",
        "drop_and_alarm", "alarm_given_drop", "drop_given_alarm",
    ], block_summary)
    (args.output / "settings.json").write_text(json.dumps({
        "detection": str(args.detection.resolve()), "pre_rows": args.pre_rows,
        "post_rows": args.post_rows, "block_rows": args.block_rows,
        "min_rows": args.min_rows, "drop_ratio": args.drop_ratio,
        "alarm_delay": args.alarm_delay,
        "note": "Performance drop compares post-drift error with the same model's pre-drift error.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()

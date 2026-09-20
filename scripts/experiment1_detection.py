# -*- coding: utf-8 -*-
"""Compare detector alarms with the labeled events in data/generated.

Example: python scripts/experiment1_detection.py --output runs/experiment1
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from statistics import mean

from onlinetsf.__main__ import collect_drift_records, run_forecast
from onlinetsf.config import load_config
from onlinetsf.data import BENCHMARK_SPECS


def write_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def match_events(labels: list[dict], alarms: list[int], first: int, last: int, delay: int) -> tuple[list[dict], int, int]:
    """Match at most one alarm to each event; exclude alarms in censored windows."""

    matches: list[dict] = []
    used: set[int] = set()
    for position, label in enumerate(labels):
        start = int(label["start_index"])
        end = int(label["end_index_exclusive"])
        next_start = int(labels[position + 1]["start_index"]) if position + 1 < len(labels) else last + 1
        window_end = min(end + delay, next_start, last + 1)
        status = "miss"
        alarm_row = None
        if start < first:
            status = "excluded_warmup"
        elif end > last + 1:
            status = "excluded_tail"
        else:
            for alarm in alarms:
                if alarm in used:
                    continue
                if start <= alarm < window_end:
                    alarm_row = alarm
                    used.add(alarm)
                    status = "hit"
                    break
        matches.append({
            "event_id": label["event_id"],
            "method": label["method"],
            "transition": label["transition"],
            "start_index": start,
            "end_index_exclusive": end,
            "match_window_end_exclusive": window_end,
            "status": status,
            "alarm_raw_index": alarm_row,
            "delay_from_start": None if alarm_row is None else alarm_row - start,
            "affected_variables": label["affected_variables"],
        })
    excluded_alarms = {
        alarm for alarm in alarms
        if any(
            row["status"].startswith("excluded")
            and row["start_index"] <= alarm < row["match_window_end_exclusive"]
            for row in matches
        )
    }
    return matches, len(alarms) - len(used) - len(excluded_alarms), len(excluded_alarms)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, default=Path("data/generated"))
    parser.add_argument("--config-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strategies", nargs="+", default=["tcn_ogd", "fsnet", "onenet"])
    parser.add_argument(
        "--detectors", nargs="+",
        default=["page_hinkley", "adwin", "kswin", "seed", "stepd", "hddmw", "abcd"],
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-delay", type=int, default=32)
    parser.add_argument("--warmup-rows", type=int, default=100)
    parser.add_argument("--cooldown-rows", type=int, default=8)
    parser.add_argument("--max-streams", type=int, help="limit streams for a quick check")
    args = parser.parse_args()
    if args.max_delay < 0 or args.warmup_rows < 1 or args.cooldown_rows < 0:
        parser.error("max-delay and cooldown-rows must be nonnegative; warmup-rows must be positive")
    if args.max_streams is not None and args.max_streams < 1:
        parser.error("max-streams must be positive")
    if {"seed", "stepd", "abcd"}.intersection(args.detectors) and importlib.util.find_spec("capymoa") is None:
        parser.error("SEED, STEPD, and ABCD require the optional capymoa package")

    manifest_paths = sorted(args.generated.rglob("manifest.json"))
    if args.max_streams is not None:
        manifest_paths = manifest_paths[:args.max_streams]
    if not manifest_paths:
        parser.error(f"no generated manifests found under {args.generated}")
    args.output.mkdir(parents=True, exist_ok=False)

    forecast_rows: list[dict] = []
    alarm_rows: list[dict] = []
    match_rows: list[dict] = []
    summary_rows: list[dict] = []
    run_rows: list[dict] = []
    for manifest_path in manifest_paths:
        dataset_name = manifest_path.parent.parent.name.removesuffix("-batch")
        stream_id = f"{dataset_name}-batch/{manifest_path.parent.name}"
        config_path = args.config_dir / f"config_{dataset_name}.yaml"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with (manifest_path.parent / manifest["files"]["labels"]).open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            labels = sorted(csv.DictReader(handle), key=lambda row: int(row["start_index"]))
        with (manifest_path.parent / manifest["files"]["data"]).open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            columns = next(csv.reader(handle))
        feature_names = [name for name in columns if name != BENCHMARK_SPECS[dataset_name].time_column]
        target_names = list(BENCHMARK_SPECS[dataset_name].target_columns or feature_names)
        runs = {}
        for strategy in args.strategies:
            config = load_config(config_path, strategy_name=strategy, detector_name=args.detectors[0])
            config["data"]["path"] = str(manifest_path.parent / manifest["files"]["data"])
            config["online"]["device"] = args.device
            config["online"]["keep_predictions"] = True
            config["output"]["write_per_value_errors"] = False
            if config["data"]["horizon"] != 1 or config["data"].get("stride", 1) != 1:
                raise ValueError("these experiment scripts require horizon=1 and stride=1")
            run = run_forecast(config)
            runs[strategy] = run
            for event in run.events:
                forecast_rows.append({
                    "stream_id": stream_id, "dataset": dataset_name, "strategy": strategy,
                    "forecast_index": event.index,
                    "target_raw_index": event.index + config["data"]["context_length"],
                    "available_at": event.available_at,
                    "mae": event.mae, "mse": event.mse,
                })
            run_rows.append({
                "stream_id": stream_id, "dataset": dataset_name, "strategy": strategy,
                "forecast_count": run.metrics.forecasts_emitted,
                "mae": run.metrics.mae, "mse": run.metrics.mse,
                "adaptation_steps": run.metrics.adaptation_steps,
            })

        for detector_name in args.detectors:
            probe = load_config(config_path, strategy_name=args.strategies[0], detector_name=detector_name)
            source = probe["drift"]["source"]
            strategies = args.strategies if source == "residual" else [args.strategies[0]]
            for strategy in strategies:
                config = load_config(config_path, strategy_name=strategy, detector_name=detector_name)
                config["data"]["path"] = str(manifest_path.parent / manifest["files"]["data"])
                config["data"]["feature_names"] = feature_names
                config["data"]["target_names"] = target_names
                records = collect_drift_records(config, runs[strategy])
                if not records:
                    continue
                scope = strategy if source == "residual" else "shared"
                first = min(record.raw_signal_index for record in records) + args.warmup_rows - 1
                last = max(record.raw_signal_index for record in records)
                if first > last:
                    raise ValueError(f"warmup exceeds observed stream for {stream_id}/{detector_name}")
                raw_alarms = sorted(
                    record.raw_signal_index for record in records
                    if record.detected and first <= record.raw_signal_index <= last
                )
                collapsed: list[int] = []
                for raw_index in raw_alarms:
                    if not collapsed or raw_index - collapsed[-1] > args.cooldown_rows:
                        collapsed.append(raw_index)
                eligible_alarms = collapsed
                matched, false_alarms, excluded_alarms = match_events(
                    labels, eligible_alarms, first, last, args.max_delay,
                )
                for record in records:
                    if record.detected:
                        alarm_rows.append({
                            "stream_id": stream_id, "dataset": dataset_name, "strategy": scope,
                            "detector": detector_name, "source": source,
                            "forecast_index": record.index, "available_at": record.available_at,
                            "raw_signal_index": record.raw_signal_index,
                            "variable_name": record.variable_name,
                            "collapsed_alarm": record.raw_signal_index in eligible_alarms,
                        })
                for match in matched:
                    match_rows.append({
                        "stream_id": stream_id, "dataset": dataset_name, "strategy": scope,
                        "detector": detector_name, "source": source, **match,
                    })
                evaluated = [row for row in matched if row["status"] in {"hit", "miss"}]
                hits = [row for row in evaluated if row["status"] == "hit"]
                precision = len(hits) / (len(hits) + false_alarms) if hits or false_alarms else 0.0
                recall = len(hits) / len(evaluated) if evaluated else None
                if recall is None:
                    f1 = None
                elif precision + recall:
                    f1 = 2 * precision * recall / (precision + recall)
                else:
                    f1 = 0.0
                summary_rows.append({
                    "stream_id": stream_id, "dataset": dataset_name, "strategy": scope,
                    "detector": detector_name, "source": source,
                    "evaluated_events": len(evaluated), "hits": len(hits), "misses": len(evaluated) - len(hits),
                    "excluded_events": len(matched) - len(evaluated),
                    "alarms": len(eligible_alarms), "false_alarms": false_alarms,
                    "excluded_alarms": excluded_alarms,
                    "precision": precision, "recall": recall, "f1": f1,
                    "mean_delay": mean(row["delay_from_start"] for row in hits) if hits else None,
                    "false_alarms_per_1000": 1000 * false_alarms / (last - first + 1),
                    "first_evaluable_raw_index": first, "last_evaluable_raw_index": last,
                })
        print(f"completed {stream_id}", flush=True)

    write_rows(args.output / "forecast_steps.csv", [
        "stream_id", "dataset", "strategy", "forecast_index", "target_raw_index",
        "available_at", "mae", "mse",
    ], forecast_rows)
    write_rows(args.output / "alarm_events.csv", [
        "stream_id", "dataset", "strategy", "detector", "source", "forecast_index",
        "available_at", "raw_signal_index", "variable_name", "collapsed_alarm",
    ], alarm_rows)
    write_rows(args.output / "event_matches.csv", [
        "stream_id", "dataset", "strategy", "detector", "source", "event_id", "method",
        "transition", "start_index", "end_index_exclusive", "match_window_end_exclusive",
        "status", "alarm_raw_index", "delay_from_start", "affected_variables",
    ], match_rows)
    write_rows(args.output / "detection_summary.csv", [
        "stream_id", "dataset", "strategy", "detector", "source", "evaluated_events",
        "hits", "misses", "excluded_events", "alarms", "false_alarms", "excluded_alarms", "precision",
        "recall", "f1", "mean_delay", "false_alarms_per_1000",
        "first_evaluable_raw_index", "last_evaluable_raw_index",
    ], summary_rows)
    write_rows(args.output / "run_summary.csv", [
        "stream_id", "dataset", "strategy", "forecast_count", "mae", "mse", "adaptation_steps",
    ], run_rows)
    (args.output / "settings.json").write_text(json.dumps({
        "generated": str(args.generated.resolve()), "strategies": args.strategies,
        "detectors": args.detectors, "max_delay": args.max_delay,
        "warmup_rows": args.warmup_rows, "cooldown_rows": args.cooldown_rows,
        "device": args.device, "streams": len(manifest_paths),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()

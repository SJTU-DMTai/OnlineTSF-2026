# -*- coding: utf-8 -*-
"""Merge independently run experiment 1 outputs for experiment 2.

Example:
python scripts/merge_experiment1_results.py --inputs runs/experiment1_etth1 runs/experiment1_etth2 --output runs/experiment1_merged
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CSV_FILES = [
    "forecast_steps.csv",
    "alarm_events.csv",
    "event_matches.csv",
    "detection_summary.csv",
    "run_summary.csv",
]
SETTING_KEYS = ["strategies", "detectors", "max_delay", "warmup_rows", "cooldown_rows"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True, help="experiment 1 output directories")
    parser.add_argument("--output", type=Path, required=True, help="new merged output directory")
    args = parser.parse_args()

    input_dirs = [path.resolve() for path in args.inputs]
    if len(set(input_dirs)) != len(input_dirs):
        parser.error("--inputs contains the same directory more than once")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    headers: dict[str, list[str]] = {}
    reference_settings: dict | None = None
    seen_streams: set[str] = set()
    for input_dir in input_dirs:
        if not input_dir.is_dir():
            parser.error(f"input directory does not exist: {input_dir}")
        settings_path = input_dir / "settings.json"
        if not settings_path.is_file():
            parser.error(f"missing settings file: {settings_path}")
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        signature = {key: settings.get(key) for key in SETTING_KEYS}
        if reference_settings is None:
            reference_settings = signature
        elif signature != reference_settings:
            parser.error(f"experiment settings differ in: {input_dir}")

        for filename in CSV_FILES:
            csv_path = input_dir / filename
            if not csv_path.is_file():
                parser.error(f"missing result file: {csv_path}")
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    parser.error(f"missing CSV header: {csv_path}")
                if filename not in headers:
                    headers[filename] = reader.fieldnames
                elif reader.fieldnames != headers[filename]:
                    parser.error(f"CSV columns differ in: {csv_path}")
                if filename == "run_summary.csv":
                    input_streams = {row["stream_id"] for row in reader}
                    duplicates = seen_streams.intersection(input_streams)
                    if duplicates:
                        parser.error(f"duplicate streams across inputs: {sorted(duplicates)}")
                    seen_streams.update(input_streams)

    args.output.mkdir(parents=True)
    for filename in CSV_FILES:
        with (args.output / filename).open("w", encoding="utf-8", newline="") as output_handle:
            writer = csv.DictWriter(output_handle, fieldnames=headers[filename])
            writer.writeheader()
            for input_dir in input_dirs:
                with (input_dir / filename).open("r", encoding="utf-8", newline="") as input_handle:
                    writer.writerows(csv.DictReader(input_handle))

    settings = {
        "sources": [str(path) for path in input_dirs],
        "streams": len(seen_streams),
        **reference_settings,
    }
    (args.output / "settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(f"merged_streams={len(seen_streams)}")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()

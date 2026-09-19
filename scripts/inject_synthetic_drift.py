# -*- coding: utf-8 -*-
"""Step 2: inject labeled abrupt or gradual drift into stable source intervals."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


METHODS = ("shuffle", "mean", "scale", "permutation")
TRANSITIONS = ("abrupt", "gradual")


@dataclass(frozen=True)
class SourceInterval:
    """One stable source interval using half-open raw-row coordinates."""

    interval_id: int
    raw_start: int
    raw_end: int

    @property
    def length(self) -> int:
        return self.raw_end - self.raw_start


@dataclass(frozen=True)
class DriftLabel:
    """Ground-truth transition interval and its generation metadata."""

    event_id: int
    method: str
    transition: str
    start_index: int
    end_index_exclusive: int
    new_concept_full_from: int
    affected_variables: tuple[str, ...]
    source_before: int
    source_after: int
    parameters: dict[str, object]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inject labeled drift into intervals selected by select_stable_intervals.py"
    )
    parser.add_argument("--source", required=True, help="original benchmark CSV")
    parser.add_argument("--stable-intervals", required=True, help="Step 1 stable_intervals.csv")
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--transition", choices=TRANSITIONS, default="abrupt")
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="number of transition rows for gradual drift; abrupt drift always uses zero",
    )
    parser.add_argument("--value", type=float, help="additive value for mean or multiplier for scale")
    parser.add_argument("--variables", nargs="+", help="affected variables; defaults to all numeric columns")
    parser.add_argument("--interval-id", type=int, help="stable interval for non-shuffle methods; defaults to longest")
    parser.add_argument("--onset-fraction", type=float, default=0.5)
    parser.add_argument("--num-segments", type=int, default=3, help="stable intervals used by shuffle")
    parser.add_argument("--time-column", default="date")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True, help="new output directory")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.transition == "gradual" and args.width <= 0:
        raise ValueError("gradual drift requires --width greater than zero")
    if args.transition == "abrupt" and args.width != 0:
        raise ValueError("abrupt drift requires --width 0")
    if args.method in {"mean", "scale"} and args.value is None:
        raise ValueError(f"{args.method} drift requires --value")
    if args.value is not None and not math.isfinite(args.value):
        raise ValueError("--value must be finite")
    if args.method not in {"mean", "scale"} and args.value is not None:
        raise ValueError(f"{args.method} drift does not use --value")
    if args.method == "shuffle" and args.variables is not None:
        raise ValueError("shuffle drift always affects the complete multivariate regime")
    if args.method == "mean" and args.value == 0.0:
        raise ValueError("mean drift requires a non-zero additive value")
    if args.method == "scale" and args.value == 1.0:
        raise ValueError("scale drift requires a multiplier different from one")
    if not math.isfinite(args.onset_fraction) or not 0.0 < args.onset_fraction < 1.0:
        raise ValueError("--onset-fraction must be in (0, 1)")
    if args.method == "shuffle" and args.num_segments < 2:
        raise ValueError("--num-segments must be at least two")


def new_concept_probability(
    position: int,
    *,
    start: int,
    transition: str,
    width: int,
) -> float:
    """Return the shared old/new mixing progress at one output position."""

    if position < start:
        return 0.0
    if transition == "abrupt":
        return 1.0
    if transition != "gradual" or width <= 0:
        raise ValueError("gradual transition requires a positive width")
    if position >= start + width:
        return 1.0
    return (position - start + 1) / width


def read_source(path: str | Path) -> tuple[list[str], list[list[str]]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError(f"source CSV is empty: {path}") from error
        rows = [row for row in reader]
    if not header or any(len(row) != len(header) for row in rows):
        raise ValueError("source CSV has an invalid header or inconsistent row widths")
    return header, rows


def read_stable_intervals(path: str | Path, source_length: int) -> list[SourceInterval]:
    intervals: list[SourceInterval] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"interval_id", "raw_start", "raw_end_exclusive"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"stable interval CSV must contain {sorted(required)}")
        for row in reader:
            interval = SourceInterval(
                interval_id=int(row["interval_id"]),
                raw_start=int(row["raw_start"]),
                raw_end=int(row["raw_end_exclusive"]),
            )
            if interval.raw_start < 0 or interval.raw_end > source_length or interval.length <= 0:
                raise ValueError(f"invalid stable interval {interval.interval_id}")
            intervals.append(interval)
    if not intervals:
        raise ValueError("stable interval CSV contains no intervals")
    if len({interval.interval_id for interval in intervals}) != len(intervals):
        raise ValueError("stable interval IDs must be unique")
    return intervals


def numeric_columns(header: Sequence[str], rows: Sequence[Sequence[str]], time_column: str) -> list[str]:
    if time_column not in header:
        raise ValueError(f"time column {time_column!r} is missing from source CSV")
    if not rows:
        raise ValueError("source CSV contains no data rows")
    columns: list[str] = []
    for index, name in enumerate(header):
        if name == time_column:
            continue
        try:
            float(rows[0][index])
        except ValueError as error:
            raise ValueError(f"column {name!r} is not numeric") from error
        columns.append(name)
    if not columns:
        raise ValueError("source CSV has no numeric variables")
    return columns


def selected_variables(
    requested: Sequence[str] | None,
    available: Sequence[str],
    method: str,
) -> tuple[str, ...]:
    selected = tuple(requested or available)
    missing = set(selected).difference(available)
    if missing:
        raise ValueError(f"variables not found in source CSV: {sorted(missing)}")
    if len(set(selected)) != len(selected):
        raise ValueError("variables must not contain duplicates")
    if method == "permutation" and len(selected) < 2:
        raise ValueError("permutation drift requires at least two variables")
    return selected


def choose_interval(intervals: Sequence[SourceInterval], interval_id: int | None) -> SourceInterval:
    if interval_id is None:
        return max(intervals, key=lambda interval: interval.length)
    for interval in intervals:
        if interval.interval_id == interval_id:
            return interval
    raise ValueError(f"stable interval ID {interval_id} was not found")


def non_identity_permutation(variables: Sequence[str], rng: random.Random) -> tuple[str, ...]:
    permutation = list(variables)
    rng.shuffle(permutation)
    if permutation == list(variables):
        permutation = permutation[1:] + permutation[:1]
    return tuple(permutation)


def transformed_row(
    row: Sequence[str],
    *,
    header: Sequence[str],
    method: str,
    variables: Sequence[str],
    value: float | None,
    permutation: Sequence[str] | None,
) -> list[str]:
    result = list(row)
    positions = {name: header.index(name) for name in variables}
    if method == "mean":
        assert value is not None
        for name in variables:
            position = positions[name]
            result[position] = repr(float(row[position]) + value)
    elif method == "scale":
        assert value is not None
        for name in variables:
            position = positions[name]
            result[position] = repr(float(row[position]) * value)
    elif method == "permutation":
        assert permutation is not None
        original = {name: row[positions[name]] for name in variables}
        for destination, source in zip(variables, permutation, strict=True):
            result[positions[destination]] = original[source]
    else:
        raise ValueError(f"unsupported row transformation: {method}")
    return result


def inject_transformation(
    source_rows: Sequence[Sequence[str]],
    *,
    header: Sequence[str],
    interval: SourceInterval,
    method: str,
    transition: str,
    width: int,
    onset_fraction: float,
    variables: Sequence[str],
    value: float | None,
    rng: random.Random,
) -> tuple[list[list[str]], list[DriftLabel]]:
    rows = [list(row) for row in source_rows[interval.raw_start:interval.raw_end]]
    start = int(len(rows) * onset_fraction)
    if start <= 0 or start >= len(rows):
        raise ValueError("onset falls outside the selected stable interval")
    if transition == "gradual" and start + width >= len(rows):
        raise ValueError("gradual transition must leave at least one fully changed row")

    permutation = non_identity_permutation(variables, rng) if method == "permutation" else None
    changed_rows = 0
    transition_new_offsets: list[int] = []
    for position in range(start, len(rows)):
        probability = new_concept_probability(
            position, start=start, transition=transition, width=width
        )
        if rng.random() <= probability:
            rows[position] = transformed_row(
                rows[position],
                header=header,
                method=method,
                variables=variables,
                value=value,
                permutation=permutation,
            )
            changed_rows += 1
            if transition == "abrupt" and position == start:
                transition_new_offsets.append(0)
            elif transition == "gradual" and position < start + width:
                transition_new_offsets.append(position - start)

    transition_end = start + width if transition == "gradual" else start + 1
    full_from = start + width if transition == "gradual" else start
    parameters: dict[str, object] = {
        "value": value,
        "onset_fraction": onset_fraction,
        "realized_changed_rows": changed_rows,
        "transition_new_concept_offsets": transition_new_offsets,
    }
    if permutation is not None:
        parameters["destination_to_source"] = dict(zip(variables, permutation, strict=True))
    label = DriftLabel(
        event_id=0,
        method=method,
        transition=transition,
        start_index=start,
        end_index_exclusive=transition_end,
        new_concept_full_from=full_from,
        affected_variables=tuple(variables),
        source_before=interval.interval_id,
        source_after=interval.interval_id,
        parameters=parameters,
    )
    return rows, [label]


def shuffled_interval_order(
    intervals: Sequence[SourceInterval],
    count: int,
    rng: random.Random,
) -> list[SourceInterval]:
    if count > len(intervals):
        raise ValueError(f"requested {count} segments but only {len(intervals)} are available")
    selected = sorted(rng.sample(list(intervals), count), key=lambda interval: interval.raw_start)
    original_ids = [interval.interval_id for interval in selected]
    rng.shuffle(selected)
    if [interval.interval_id for interval in selected] == original_ids:
        selected = selected[1:] + selected[:1]
    return selected


def inject_shuffled_intervals(
    source_rows: Sequence[Sequence[str]],
    *,
    intervals: Sequence[SourceInterval],
    transition: str,
    width: int,
    time_index: int,
    variable_names: Sequence[str],
    rng: random.Random,
) -> tuple[list[list[str]], list[DriftLabel]]:
    output: list[list[str]] = []
    labels: list[DriftLabel] = []
    for segment_position, interval in enumerate(intervals):
        segment = [list(row) for row in source_rows[interval.raw_start:interval.raw_end]]
        if segment_position:
            previous = intervals[segment_position - 1]
            boundary = len(output)
            if transition == "gradual":
                if width >= len(segment) or width > previous.length:
                    raise ValueError("gradual shuffle width must fit both adjacent stable intervals")
                old_tail = output[-width:]
                realized_new_rows = 0
                transition_new_offsets: list[int] = []
                for offset in range(width):
                    probability = new_concept_probability(
                        boundary + offset,
                        start=boundary,
                        transition=transition,
                        width=width,
                    )
                    if rng.random() > probability:
                        timestamp = segment[offset][time_index]
                        segment[offset] = list(old_tail[offset])
                        segment[offset][time_index] = timestamp
                    else:
                        realized_new_rows += 1
                        transition_new_offsets.append(offset)
            else:
                realized_new_rows = 1
                transition_new_offsets = [0]

            transition_end = boundary + width if transition == "gradual" else boundary + 1
            full_from = boundary + width if transition == "gradual" else boundary
            labels.append(
                DriftLabel(
                    event_id=len(labels),
                    method="shuffle",
                    transition=transition,
                    start_index=boundary,
                    end_index_exclusive=transition_end,
                    new_concept_full_from=full_from,
                    affected_variables=tuple(variable_names),
                    source_before=previous.interval_id,
                    source_after=interval.interval_id,
                    parameters={
                        "transition_rows_from_new_concept": realized_new_rows,
                        "transition_rows": width if transition == "gradual" else 1,
                        "transition_new_concept_offsets": transition_new_offsets,
                    },
                )
            )
        output.extend(segment)

    if len(output) > len(source_rows):
        raise ValueError("shuffled output is longer than the available timestamp sequence")
    for index, row in enumerate(output):
        row[time_index] = source_rows[index][time_index]
    return output, labels


def write_dataset(path: Path, header: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_labels(path: Path, labels: Sequence[DriftLabel]) -> None:
    fields = (
        "event_id",
        "method",
        "transition",
        "start_index",
        "end_index_exclusive",
        "new_concept_full_from",
        "affected_variables",
        "source_before",
        "source_after",
        "parameters",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label in labels:
            row = asdict(label)
            row["affected_variables"] = json.dumps(label.affected_variables, ensure_ascii=False)
            row["parameters"] = json.dumps(label.parameters, ensure_ascii=False, sort_keys=True)
            writer.writerow(row)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    rng = random.Random(args.seed)
    header, source_rows = read_source(args.source)
    intervals = read_stable_intervals(args.stable_intervals, len(source_rows))
    available_variables = numeric_columns(header, source_rows, args.time_column)
    variables = selected_variables(args.variables, available_variables, args.method)
    time_index = header.index(args.time_column)

    source_order: list[int]
    if args.method == "shuffle":
        selected_intervals = shuffled_interval_order(intervals, args.num_segments, rng)
        rows, labels = inject_shuffled_intervals(
            source_rows,
            intervals=selected_intervals,
            transition=args.transition,
            width=args.width,
            time_index=time_index,
            variable_names=available_variables,
            rng=rng,
        )
        source_order = [interval.interval_id for interval in selected_intervals]
        used_intervals = selected_intervals
    else:
        interval = choose_interval(intervals, args.interval_id)
        rows, labels = inject_transformation(
            source_rows,
            header=header,
            interval=interval,
            method=args.method,
            transition=args.transition,
            width=args.width,
            onset_fraction=args.onset_fraction,
            variables=variables,
            value=args.value,
            rng=rng,
        )
        source_order = [interval.interval_id]
        used_intervals = [interval]

    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    write_dataset(destination / "data.csv", header, rows)
    write_labels(destination / "drift_labels.csv", labels)
    manifest = {
        "source": str(Path(args.source).resolve()),
        "stable_intervals": str(Path(args.stable_intervals).resolve()),
        "method": args.method,
        "transition": args.transition,
        "transition_width": args.width,
        "seed": args.seed,
        "variables": list(variables) if args.method != "shuffle" else available_variables,
        "source_interval_order": source_order,
        "source_intervals": [asdict(interval) for interval in used_intervals],
        "rows": len(rows),
        "drift_events": len(labels),
        "probability_schedule": "step" if args.transition == "abrupt" else "linear",
        "index_convention": "zero-based half-open intervals [start_index, end_index_exclusive)",
        "files": {"data": "data.csv", "labels": "drift_labels.csv"},
    }
    with (destination / "manifest.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"output={destination.resolve()}")


if __name__ == "__main__":
    main()

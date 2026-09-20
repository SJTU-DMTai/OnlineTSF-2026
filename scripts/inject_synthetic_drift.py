# -*- coding: utf-8 -*-
"""Build a long stable stream, then inject one labeled synthetic drift."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Sequence


DRIFT_REGION = (0.4, 0.8)
METHODS = ("mean", "scale", "permutation")
MEAN_STD_MULTIPLIER_RANGE = (0.5, 1.5)
SCALE_FACTOR_RANGES = ((0.5, 0.8), (1.25, 2.0))
GRADUAL_PROBABILITY = 0.5
SPLICE_GUARD_ROWS = 32


@dataclass(frozen=True)
class SourceInterval:
    interval_id: int
    raw_start: int
    raw_end: int

    @property
    def length(self) -> int:
        return self.raw_end - self.raw_start


@dataclass(frozen=True)
class DriftLabel:
    event_id: int
    method: str
    transition: str
    start_index: int
    end_index_exclusive: int
    new_concept_full_from: int
    affected_variables: tuple[str, ...]
    source_before: int | None
    source_after: int | None
    parameters: dict[str, object]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one labeled drift dataset from Step 1 stable intervals"
    )
    parser.add_argument("--source", required=True, help="original benchmark CSV")
    parser.add_argument("--stable-intervals", required=True, help="Step 1 stable_intervals.csv")
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--min-length", type=int, default=512, help="minimum generated stream rows")
    parser.add_argument("--time-column", default="date")
    parser.add_argument(
        "--splice-guard-rows",
        type=int,
        default=SPLICE_GUARD_ROWS,
        help="minimum generated rows between an injected transition and a splice",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True, help="new output directory")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.min_length < 2:
        raise ValueError("--min-length must be at least two")
    if args.splice_guard_rows < 0:
        raise ValueError("--splice-guard-rows must be non-negative")


def new_concept_probability(position: int, onset: int, width: int) -> float:
    """Shared progress: a step when width=0, a linear probability otherwise."""

    if width < 0:
        raise ValueError("width must be non-negative")
    if position < onset:
        return 0.0
    if width == 0 or position >= onset + width:
        return 1.0
    return (position - onset) / width


def read_source(path: str | Path) -> tuple[list[str], list[list[str]]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError(f"source CSV is empty: {path}") from error
        rows = list(reader)
    if not header or len(set(header)) != len(header):
        raise ValueError("source CSV needs unique column names")
    if any(len(row) != len(header) for row in rows):
        raise ValueError("source CSV has inconsistent row widths")
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
        raise ValueError(f"time column {time_column!r} is missing")
    if not rows:
        raise ValueError("source CSV contains no data rows")
    positions = [(index, name) for index, name in enumerate(header) if name != time_column]
    variables = [name for _, name in positions]
    if not variables:
        raise ValueError("source CSV has no variables")
    for row in rows:
        for index, name in positions:
            if not math.isfinite(float(row[index])):
                raise ValueError(f"non-finite value in variable {name!r}")
    return variables


def build_base_stream(
    source_rows: Sequence[Sequence[str]],
    intervals: Sequence[SourceInterval],
    *,
    min_length: int,
    variable_names: Sequence[str],
    rng: random.Random,
) -> tuple[list[list[str]], list[DriftLabel], list[dict[str, int]]]:
    """Draw shuffled interval rounds until length and two-segment requirements hold."""

    if min_length < 2 or not intervals:
        raise ValueError("at least one interval and a minimum length of two are required")
    rows: list[list[str]] = []
    labels: list[DriftLabel] = []
    provenance: list[dict[str, int]] = []
    previous: SourceInterval | None = None
    while len(rows) < min_length or len(provenance) < 2:
        order = list(intervals)
        rng.shuffle(order)
        if previous is not None and len(order) > 1 and order[0].interval_id == previous.interval_id:
            order[0], order[1] = order[1], order[0]
        for interval in order:
            if len(rows) >= min_length and len(provenance) >= 2:
                break
            boundary = len(rows)
            if previous is not None and previous.raw_end != interval.raw_start:
                labels.append(
                    DriftLabel(
                        event_id=len(labels),
                        method="splice",
                        transition="abrupt",
                        start_index=boundary,
                        end_index_exclusive=boundary + 1,
                        new_concept_full_from=boundary,
                        affected_variables=tuple(variable_names),
                        source_before=previous.interval_id,
                        source_after=interval.interval_id,
                        parameters={
                            "source_before_end_exclusive": previous.raw_end,
                            "source_after_start": interval.raw_start,
                        },
                    )
                )
            rows.extend(list(row) for row in source_rows[interval.raw_start:interval.raw_end])
            provenance.append(
                {
                    "interval_id": interval.interval_id,
                    "raw_start": interval.raw_start,
                    "raw_end_exclusive": interval.raw_end,
                    "output_start": boundary,
                    "output_end_exclusive": len(rows),
                }
            )
            previous = interval
    return rows, labels, provenance


def restore_timestamps(
    rows: list[list[str]], source_rows: Sequence[Sequence[str]], time_index: int
) -> None:
    """Give the shuffled values a monotone source-cadence timeline."""

    available = [row[time_index] for row in source_rows]
    if len(rows) > len(available):
        if len(available) < 2:
            raise ValueError("at least two source timestamps are needed to extend the timeline")
        try:
            last = datetime.fromisoformat(available[-1])
            step = last - datetime.fromisoformat(available[-2])
        except ValueError:
            last_number = Decimal(available[-1])
            step_number = last_number - Decimal(available[-2])
            if step_number <= 0:
                raise ValueError("source timestamps must increase")
            for _ in range(len(rows) - len(available)):
                last_number += step_number
                available.append(str(last_number))
        else:
            if step.total_seconds() <= 0:
                raise ValueError("source timestamps must increase")
            for _ in range(len(rows) - len(available)):
                last += step
                available.append(last.isoformat(sep=" "))
    for index, row in enumerate(rows):
        row[time_index] = available[index]


def choose_onset(
    length: int,
    width: int,
    rng: random.Random,
    splice_points: Sequence[int] = (),
    splice_guard_rows: int = SPLICE_GUARD_ROWS,
) -> int:
    first = math.ceil(DRIFT_REGION[0] * length)
    last = min(math.floor(DRIFT_REGION[1] * length), length - width - 1)
    if first > last:
        raise ValueError("drift width leaves no valid onset in the 40%-80% region")
    if splice_guard_rows < 0:
        raise ValueError("splice guard rows must be non-negative")
    candidates = [
        onset
        for onset in range(first, last + 1)
        if all(
            onset + max(width, 1) + splice_guard_rows <= splice
            or onset >= splice + splice_guard_rows
            for splice in splice_points
        )
    ]
    if not candidates:
        raise ValueError("no drift onset remains outside the splice guard regions")
    return rng.choice(candidates)


def choose_width(maximum: int, rng: random.Random) -> int:
    """Randomly choose abrupt drift or a short valid gradual transition."""

    if maximum < 0:
        raise ValueError("maximum gradual width must be non-negative")
    if maximum == 0 or rng.random() >= GRADUAL_PROBABILITY:
        return 0
    return rng.randint(1, maximum)


def choose_variables(variables: Sequence[str], method: str, rng: random.Random) -> tuple[str, ...]:
    minimum = 2 if method == "permutation" else 1
    if len(variables) < minimum:
        raise ValueError(f"{method} drift requires at least {minimum} numeric variables")
    count = rng.randint(minimum, len(variables))
    selected = set(rng.sample(list(variables), count))
    return tuple(name for name in variables if name in selected)


def non_identity_permutation(variables: Sequence[str], rng: random.Random) -> tuple[str, ...]:
    order = list(variables)
    rng.shuffle(order)
    mapping = dict(zip(order, order[1:] + order[:1], strict=True))
    return tuple(mapping[name] for name in variables)


def sample_drift_parameters(
    rows: Sequence[Sequence[str]],
    header: Sequence[str],
    method: str,
    variables: Sequence[str],
    rng: random.Random,
) -> dict[str, object]:
    """Sample drift magnitudes from the pre-drift stream's selected variables."""

    if method == "permutation":
        return {}

    centers: dict[str, float] = {}
    standard_deviations: dict[str, float] = {}
    for name in variables:
        position = header.index(name)
        values = [float(row[position]) for row in rows]
        center = math.fsum(values) / len(values)
        standard_deviation = math.sqrt(
            math.fsum((value - center) ** 2 for value in values) / len(values)
        )
        centers[name] = center
        standard_deviations[name] = standard_deviation

    if method == "mean":
        offsets = {}
        reference_scales = {}
        for name in variables:
            reference = standard_deviations[name]
            if reference == 0:
                reference = max(abs(centers[name]) * 0.1, 1.0)
            reference_scales[name] = reference
            multiplier = rng.uniform(*MEAN_STD_MULTIPLIER_RANGE)
            offsets[name] = reference * multiplier * rng.choice((-1.0, 1.0))
        return {
            "offsets": offsets,
            "reference_scales": reference_scales,
            "mean_std_multiplier_range": list(MEAN_STD_MULTIPLIER_RANGE),
        }

    factors = {}
    for name in variables:
        factor_range = rng.choice(SCALE_FACTOR_RANGES)
        factors[name] = rng.uniform(*factor_range)
    return {
        "centers": centers,
        "standard_deviations": standard_deviations,
        "factors": factors,
        "scale_factor_ranges": [list(value) for value in SCALE_FACTOR_RANGES],
    }


def inject_drift(
    rows: list[list[str]],
    *,
    header: Sequence[str],
    method: str,
    drift_parameters: dict[str, object],
    width: int,
    onset: int,
    variables: Sequence[str],
    rng: random.Random,
    event_id: int,
) -> DriftLabel:
    """Apply one persistent change, sampling old/new rows only during gradual drift."""

    positions = [header.index(name) for name in variables]
    permutation = non_identity_permutation(variables, rng) if method == "permutation" else None
    source_positions = [header.index(name) for name in permutation] if permutation else []
    transition_new_offsets: list[int] = []
    for index in range(onset, len(rows)):
        probability = new_concept_probability(index, onset, width)
        if probability == 0 or (probability < 1 and rng.random() >= probability):
            continue
        original = rows[index].copy()
        for destination_index, position in enumerate(positions):
            if method == "mean":
                offsets = drift_parameters["offsets"]
                rows[index][position] = repr(float(original[position]) + offsets[variables[destination_index]])
            elif method == "scale":
                centers = drift_parameters["centers"]
                factors = drift_parameters["factors"]
                name = variables[destination_index]
                rows[index][position] = repr(
                    centers[name] + (float(original[position]) - centers[name]) * factors[name]
                )
            else:
                rows[index][position] = original[source_positions[destination_index]]
        if index < onset + width:
            transition_new_offsets.append(index - onset)

    parameters: dict[str, object] = {
        **drift_parameters,
        "width": width,
        "transition_new_concept_offsets": transition_new_offsets,
    }
    if permutation is not None:
        parameters["destination_to_source"] = dict(zip(variables, permutation, strict=True))
    return DriftLabel(
        event_id=event_id,
        method=method,
        transition="abrupt" if width == 0 else "gradual",
        start_index=onset,
        end_index_exclusive=onset + max(width, 1),
        new_concept_full_from=onset + width,
        affected_variables=tuple(variables),
        source_before=None,
        source_after=None,
        parameters=parameters,
    )


def write_dataset(path: Path, header: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_labels(path: Path, labels: Sequence[DriftLabel]) -> None:
    fields = tuple(DriftLabel.__dataclass_fields__)
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
    variable_names = numeric_columns(header, source_rows, args.time_column)
    rows, labels, provenance = build_base_stream(
        source_rows,
        intervals,
        min_length=args.min_length,
        variable_names=variable_names,
        rng=rng,
    )
    maximum_gradual_width = min(
        min(segment["output_end_exclusive"] - segment["output_start"] for segment in provenance) // 4,
        len(rows) // 10,
    )
    restore_timestamps(rows, source_rows, header.index(args.time_column))
    width = choose_width(maximum_gradual_width, rng)
    onset = choose_onset(
        len(rows),
        width,
        rng,
        [label.start_index for label in labels if label.method == "splice"],
        args.splice_guard_rows,
    )
    variables = choose_variables(variable_names, args.method, rng)
    drift_parameters = sample_drift_parameters(rows, header, args.method, variables, rng)
    labels.append(
        inject_drift(
            rows,
            header=header,
            method=args.method,
            drift_parameters=drift_parameters,
            width=width,
            onset=onset,
            variables=variables,
            rng=rng,
            event_id=len(labels),
        )
    )
    labels.sort(key=lambda label: (label.start_index, label.event_id))
    labels = [replace(label, event_id=index) for index, label in enumerate(labels)]

    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    write_dataset(destination / "data.csv", header, rows)
    write_labels(destination / "drift_labels.csv", labels)
    manifest = {
        "source": str(Path(args.source).resolve()),
        "stable_intervals": str(Path(args.stable_intervals).resolve()),
        "method": args.method,
        "transition": "abrupt" if width == 0 else "gradual",
        "width": width,
        "maximum_gradual_width": maximum_gradual_width,
        "splice_guard_rows": args.splice_guard_rows,
        "drift_region": list(DRIFT_REGION),
        "drift_onset": onset,
        "seed": args.seed,
        "affected_variables": list(variables),
        "source_segments": provenance,
        "rows": len(rows),
        "drift_events": len(labels),
        "index_convention": "zero-based half-open [start_index, end_index_exclusive)",
        "files": {"data": "data.csv", "labels": "drift_labels.csv"},
    }
    with (destination / "manifest.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"output={destination.resolve()}")


if __name__ == "__main__":
    main()

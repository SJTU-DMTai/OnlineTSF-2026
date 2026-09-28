# -*- coding: utf-8 -*-
"""Convert labeled drift corpora to independent, benchmark-compatible series."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LABEL_FIELDS = (
    "event_id", "method", "transition", "start_index", "end_index_exclusive",
    "new_concept_full_from", "affected_variables", "source_before", "source_after",
    "parameters",
)


def event(method: str, transition: str, start: int, end: int, full_from: int | None, target: str,
          before: int | None = None, after: int | None = None) -> dict[str, object]:
    return {
        "method": method, "transition": transition, "start_index": start,
        "end_index_exclusive": end, "new_concept_full_from": full_from,
        "affected_variables": [target], "source_before": before,
        "source_after": after, "parameters": {},
    }


def write_case(destination: Path, rows: list[list[str]], columns: list[str],
               labels: list[dict[str, object]], manifest: dict[str, object],
               time_column: str, target_column: str, states: list[int] | None = None) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / "data.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
    with (destination / "drift_labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        for index, label in enumerate(labels):
            record = {**label, "event_id": index}
            record["affected_variables"] = json.dumps(record["affected_variables"])
            record["parameters"] = json.dumps(record["parameters"])
            writer.writerow(record)
    if states is not None:
        with (destination / "concept_state.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("index", "source_a_selected"))
            writer.writerows(enumerate(states))
    manifest = {
        **manifest, "rows": len(rows), "time_column": time_column,
        "feature_columns": columns[1:], "target_columns": [target_column],
        "index_convention": "zero-based half-open [start_index, end_index_exclusive)",
        "files": {"data": "data.csv", "labels": "drift_labels.csv"},
    }
    if states is not None:
        manifest["files"]["concept_state"] = "concept_state.csv"
    with (destination / "manifest.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def read_numeric(value: str, source: Path) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite value in {source}")
    return value


def prepare_mendeley(source: Path, output: Path) -> int:
    count = 0
    for name in ("synthetic_concept_drift", "synthetic_data_drift", "synthetic_gradual"):
        rows: list[list[str]] = []
        states: list[tuple[int, ...]] = []
        split_ends: dict[str, int] = {}
        paths: list[str] = []
        for split in ("train", "val", "test"):
            path = source / f"{name}_{split}.csv"
            paths.append(str(path.resolve()))
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                expected = {"timestamp", *(f"feature_{i}" for i in range(5)), "target"}
                label_columns = ("drift_start", "drift_end") if name == "synthetic_gradual" else ("drift_point",)
                if not reader.fieldnames or not expected.union(label_columns).issubset(reader.fieldnames):
                    raise ValueError(f"unexpected columns in {path}")
                for row in reader:
                    timestamp = int(row["timestamp"])
                    if timestamp != len(rows):
                        raise ValueError(f"non-contiguous timestamp in {path}: {timestamp}")
                    rows.append([str(timestamp), *(read_numeric(row[f"feature_{i}"], path) for i in range(5)),
                                 read_numeric(row["target"], path)])
                    state = tuple(int(row[column]) for column in label_columns)
                    if any(value not in (0, 1) for value in state):
                        raise ValueError(f"non-binary drift label in {path}")
                    states.append(state)
            split_ends[split] = len(rows)
        if name == "synthetic_gradual":
            starts = [i for i, state in enumerate(states) if state[0] and (i == 0 or not states[i - 1][0])]
            ends = [i for i, state in enumerate(states) if state[1] and (i == 0 or not states[i - 1][1])]
            if len(starts) != 1 or len(ends) != 1 or not starts[0] < ends[0]:
                raise ValueError(f"expected one gradual transition in {name}")
            labels = [event("gradual", "gradual", starts[0], ends[0], ends[0], "target")]
        else:
            starts = [i for i, state in enumerate(states) if state[0] and (i == 0 or not states[i - 1][0])]
            if len(starts) != 1:
                raise ValueError(f"expected one abrupt transition in {name}")
            kind = "concept" if name == "synthetic_concept_drift" else "data"
            labels = [event(kind, "abrupt", starts[0], starts[0] + 1, starts[0], "target")]
        write_case(output / "mendeley" / name, rows,
                   ["timestamp", *(f"feature_{i}" for i in range(5)), "target"], labels,
                   {"source": "Mendeley c9zkwnh4px/1", "source_paths": paths,
                    "split_end_exclusive": split_ends, "label_origin": "published binary columns"},
                   "timestamp", "target")
        count += 1
    return count


def prepare_liu(source: Path, output: Path, limit: int | None) -> int:
    count = 0
    for name, filename in (
        ("sudden", "sudden_concept_drift_df.csv"),
        ("incremental", "incremental_cd.csv"),
        ("gradual", "gradual_df.csv"),
    ):
        path = source / filename
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for series_id, row in enumerate(reader):
                if limit is not None and series_id >= limit:
                    break
                values = ast.literal_eval(row["series"])
                if len(values) != 2000:
                    raise ValueError(f"expected 2000 values in {path} row {series_id + 2}")
                rows = [[str(i), read_numeric(str(value), path)] for i, value in enumerate(values)]
                states = None
                if name == "sudden":
                    point = int(row["drift_point"])
                    if not 0 < point < len(rows):
                        raise ValueError(f"invalid drift point in {path} row {series_id + 2}")
                    labels = [event(name, "abrupt", point, point + 1, point, "series")]
                elif name == "incremental":
                    start, end = int(row["start_point"]), int(row["ending_point"])
                    if not 0 <= start < end <= len(rows):
                        raise ValueError(f"invalid drift interval in {path} row {series_id + 2}")
                    labels = [event(name, "incremental", start, end, end, "series")]
                else:
                    states = ast.literal_eval(row["alpha"])
                    if len(states) != len(rows) or any(state not in (0, 1) for state in states):
                        raise ValueError(f"invalid alpha vector in {path} row {series_id + 2}")
                    labels = [event(name, "gradual", 0, len(rows), len(rows), "series")]
                write_case(output / "liu2023" / name / f"{series_id:04d}", rows,
                           ["index", "series"], labels,
                           {"source": "Liu et al. 2023 AR(3) simulation", "source_path": str(path.resolve()),
                            "source_row": series_id + 2, "label_origin": "generator",
                            "source_a_selected": "1=A, 0=B" if states is not None else None},
                           "index", "series", states)
                count += 1
    return count


def prepare_sisc(source: Path, output: Path, limit: int | None) -> int:
    count = 0
    synthetic = source / "Series Sinteticas"
    for path in sorted(synthetic.rglob("*.csv")):
        if limit is not None and count >= limit:
            break
        relative = path.relative_to(synthetic)
        rows: list[list[str]] = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            for index, row in enumerate(reader):
                if len(row) != 1:
                    raise ValueError(f"expected one column in {path} row {index + 1}")
                rows.append([str(index), read_numeric(row[0], path)])
        if len(rows) != 20000:
            raise ValueError(f"expected 20000 values in {path}, got {len(rows)}")
        transition = "stepwise" if "graduais" in relative.parts else "abrupt"
        labels = [event("generator_regime_change", transition, point, point + 1, point, "value",
                        point // 2000 - 1, point // 2000) for point in range(2000, len(rows), 2000)]
        write_case(output / "sisc" / relative.with_suffix(""), rows,
                   ["index", "value"], labels,
                   {"source": "SISC synthetic series", "source_path": str(path.resolve()),
                    "concept_length": 2000, "label_origin": "generator segment boundaries",
                    "transition_note": "gradual categories change through successive regimes",
                    "source_value_column": None}, "index", "value")
        count += 1
    return count


def prepare_sisc_real(source: Path, output: Path) -> int:
    real = source / "Series Reais"
    for name, points, note in (
        ("Dow-drift", (124, 307, 510), "informações sobre Dow-drift.txt"),
        ("S&P500-drift", (448, 508, 1715, 2826, 4119), "informações sobre S&P500-drift.txt"),
    ):
        path = real / f"{name}.csv"
        rows: list[list[str]] = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for index, row in enumerate(csv.reader(handle)):
                if len(row) != 1:
                    raise ValueError(f"expected one column in {path} row {index + 1}")
                rows.append([str(index), read_numeric(row[0], path)])
        if any(point <= 0 or point >= len(rows) for point in points):
            raise ValueError(f"reference point outside series in {path}")
        labels = [
            event("reference_change_point", "unspecified", point, point + 1, None, "value")
            for point in points
        ]
        for label in labels:
            label["parameters"] = {"reference_only": True, "injected": False}
        write_case(
            output / "sisc" / "real" / name, rows, ["index", "value"], labels,
            {
                "source": "SISC real financial return series",
                "source_path": str(path.resolve()),
                "label_origin": "author-supplied reference points",
                "label_source_path": str((real / note).resolve()),
                "label_status": "reference_not_exact_ground_truth",
                "injected_drift": False,
                "source_value_column": None,
                "source_note": "Dow note says 735 points; CSV has 753 rows." if name == "Dow-drift" else None,
            },
            "index", "value",
        )
    return 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("mendeley", "liu2023", "sisc", "sisc_real", "all"), required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "labeled_drift")
    parser.add_argument("--limit", type=int, help="maximum series per Liu category or total SISC series")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    counts = {}
    if args.source in ("mendeley", "all"):
        counts["mendeley"] = prepare_mendeley(ROOT / "data" / "Mandeley", args.output)
    if args.source in ("liu2023", "all"):
        counts["liu2023"] = prepare_liu(ROOT / "data" / "Liu2023", args.output, args.limit)
    if args.source in ("sisc", "all"):
        counts["sisc"] = prepare_sisc(ROOT / "data" / "SISC", args.output, args.limit)
    if args.source in ("sisc", "sisc_real", "all"):
        counts["sisc_real"] = prepare_sisc_real(ROOT / "data" / "SISC", args.output)
    print(json.dumps({"output": str(args.output.resolve()), "series": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Read consensus annotations separately from forecasting inputs; sample sparse probes."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

from ..data import SlidingWindowDataset


@dataclass
class AnnotatedStream:
    dataset: SlidingWindowDataset
    phases: list[str]
    events: list[dict]
    intervals: list[dict]
    manifest: dict
    prefix_rows: int
    source: Path


def load_stream(directory: Path, context_length: int, horizon: int, prefix_rows: int | None = None) -> AnnotatedStream:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    source = directory / "labeled_data.csv"
    features = manifest["feature_columns"]
    if any(name.startswith("annotation_") or name in {"drift_event_id", "majority_vote_count"} for name in features):
        raise ValueError("manifest feature_columns must contain original measurements only")
    dataset = SlidingWindowDataset.from_csv(
        source, context_length, horizon, time_column=manifest["time_column"],
        feature_columns=features, target_columns=features, include_time_features=False, keep_decision_inputs=True,
    )
    phases = []
    with source.open(encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if int(row["annotation_index"]) != index:
                raise ValueError("annotation indices must be contiguous and zero-based")
            phases.append(row["annotation_phase"])
    if len(phases) != manifest["rows"]:
        raise ValueError("annotation length differs from manifest")
    if set(phases) - {"calibration", "stable", "transition", "drift"}:
        raise ValueError("unknown annotation phase")
    # Ceiling of 20% of raw rows; never shorten the annotation initialization prefix.
    minimum_prefix = max(manifest["calibration_rows"], (len(phases) + 4) // 5)
    prefix = minimum_prefix if prefix_rows is None else prefix_rows
    if prefix < minimum_prefix:
        raise ValueError(f"offline prefix must contain at least {minimum_prefix} rows (20% and calibration prefix)")
    if not context_length + horizon <= prefix < len(phases) - horizon:
        raise ValueError("offline prefix must contain a complete training window and leave online targets")
    train_windows = prefix - context_length - horizon + 1
    dataset.standardize(train_windows)
    if not dataset.values.isfinite().all():
        raise ValueError("forecast inputs must be finite")
    with (directory / "drift_points.csv").open(encoding="utf-8", newline="") as handle:
        events = list(csv.DictReader(handle))
    with (directory / "intervals.csv").open(encoding="utf-8", newline="") as handle:
        intervals = list(csv.DictReader(handle))
    # Hash the actual annotated input, independently of any original-source manifest hash.
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    manifest = dict(manifest, labeled_data_sha256=digest.hexdigest())
    return AnnotatedStream(dataset, phases, events, intervals, manifest, prefix, source)


def select_probes(stream: AnnotatedStream, start: int, stop: int, *, max_probes: int,
                  max_events: int, separation: int, branch_steps: int, offset: int, seed: int) -> list[dict]:
    """Cap total forks; labels affect diagnostic sampling only, never online controls."""
    rng = random.Random(seed)
    length = stream.dataset.context_length
    horizon = stream.dataset.horizon
    candidates = {name: [] for name in ("pre_event", "event", "post_event", "stable")}
    eligible = [event for event in stream.events
                if start <= int(event["start_index"]) - length + 1 < stop]
    # Spread event sampling across the stream rather than taking the first few alarms.
    count = min(max_events, len(eligible))
    selected_events = []
    for position in range(count):
        group = eligible[position * len(eligible) // count:(position + 1) * len(eligible) // count]
        selected_events.append(rng.choice(group))
    for event in selected_events:
        left, right = int(event["start_index"]), int(event["end_index_exclusive"])
        for kind, clock in (("pre_event", left - offset), ("event", left), ("post_event", right + offset)):
            index = clock - length + 1
            if start <= index and index + branch_steps <= stop:
                candidates[kind].append({"index": index, "clock": clock, "kind": kind,
                                         "event_id": int(event["event_id"]), "event_rows": right - left,
                                         "event_peak_votes": int(event["peak_votes"]) if "peak_votes" in event else None,
                                         "event_consensus_rows": int(event["consensus_rows"]) if "consensus_rows" in event else None})
    for interval in stream.intervals:
        if interval["phase"] != "stable":
            continue
        left, right = int(interval["start_index"]), int(interval["end_index_exclusive"])
        # Both context and scored future targets must lie in this stable candidate interval.
        low = max(start, left)
        high = min(stop - branch_steps, right - length - branch_steps - horizon + 1)
        if low <= high:
            index = rng.randint(low, high)
            candidates["stable"].append({"index": index, "clock": index + length - 1,
                                          "kind": "stable", "event_id": 0, "event_rows": 0})
    for group in candidates.values():
        rng.shuffle(group)
    selected = []
    while len(selected) < max_probes and any(candidates.values()):
        for group in candidates.values():
            if not group or len(selected) == max_probes:
                continue
            item = group.pop()
            if all(abs(item["index"] - other["index"]) >= separation for other in selected):
                selected.append(item)
    return sorted(selected, key=lambda item: item["index"])

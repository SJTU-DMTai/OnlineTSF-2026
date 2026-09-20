# -*- coding: utf-8 -*-
"""Focused tests for the labeled synthetic-drift generator."""

from __future__ import annotations

import csv
import json
import math
import random

import pytest

from onlinetsf.data import load_benchmark_dataset
from scripts.generate_drift_batch import main as batch_main
from scripts.inject_synthetic_drift import (
    DRIFT_REGION,
    SourceInterval,
    build_base_stream,
    choose_onset,
    choose_width,
    choose_variables,
    inject_drift,
    main,
    new_concept_probability,
    non_identity_permutation,
    restore_timestamps,
    sample_drift_parameters,
)


class FixedOrderRandom(random.Random):
    def shuffle(self, values):
        pass


def test_shared_progress_starts_at_zero_for_gradual_drift():
    assert new_concept_probability(9, 10, 0) == 0.0
    assert new_concept_probability(10, 10, 0) == 1.0
    assert new_concept_probability(10, 10, 4) == 0.0
    assert new_concept_probability(11, 10, 4) == 0.25
    assert new_concept_probability(13, 10, 4) == 0.75
    assert new_concept_probability(14, 10, 4) == 1.0


def test_adjacent_intervals_have_no_splice_label():
    source = [[str(index), str(index)] for index in range(10)]
    intervals = [SourceInterval(1, 0, 5), SourceInterval(2, 5, 10)]

    rows, labels, provenance = build_base_stream(
        source, intervals, min_length=10, variable_names=("x",), rng=FixedOrderRandom()
    )

    assert len(rows) == 10
    assert len(provenance) == 2
    assert labels == []


def test_nonadjacent_intervals_and_reused_interval_have_splice_labels():
    source = [[str(index), str(index)] for index in range(10)]
    intervals = [SourceInterval(1, 0, 4), SourceInterval(2, 6, 10)]

    rows, labels, provenance = build_base_stream(
        source, intervals, min_length=12, variable_names=("x",), rng=FixedOrderRandom()
    )

    assert len(rows) == 12
    assert len(provenance) == 3
    assert [label.start_index for label in labels] == [4, 8]
    assert all(label.method == "splice" for label in labels)
    assert labels[0].source_before == 1
    assert labels[0].source_after == 2
    assert labels[1].source_before == 2
    assert labels[1].source_after == 1


def test_one_interval_can_be_reused_to_reach_minimum_length():
    source = [[str(index), str(index)] for index in range(4)]

    rows, labels, provenance = build_base_stream(
        source,
        [SourceInterval(1, 0, 4)],
        min_length=12,
        variable_names=("x",),
        rng=random.Random(1),
    )

    assert len(rows) == 12
    assert len(provenance) == 3
    assert [label.start_index for label in labels] == [4, 8]


def test_onset_is_random_in_common_region_for_both_widths():
    for width in (0, 10):
        for seed in range(30):
            onset = choose_onset(100, width, random.Random(seed))
            assert math.ceil(DRIFT_REGION[0] * 100) <= onset <= math.floor(DRIFT_REGION[1] * 100)
            assert onset + width < 100


def test_onset_keeps_the_whole_injected_transition_away_from_splices():
    for seed in range(30):
        onset = choose_onset(512, 32, random.Random(seed), (256,), 32)
        assert onset + 32 + 32 <= 256 or onset >= 256 + 32


def test_width_randomly_selects_abrupt_or_a_short_gradual_transition():
    widths = {choose_width(8, random.Random(seed)) for seed in range(30)}

    assert 0 in widths
    assert any(1 <= width <= 8 for width in widths)


def test_timestamps_extend_when_intervals_are_reused():
    rows = [["3", "1"], ["0", "2"], ["1", "3"], ["2", "4"], ["0", "5"]]
    source = [[str(index), str(index)] for index in range(4)]

    restore_timestamps(rows, source, 0)

    assert [row[0] for row in rows] == ["0", "1", "2", "3", "4"]


def test_random_variable_selection_and_permutation_are_nontrivial():
    variables = ("a", "b", "c", "d")
    selected = choose_variables(variables, "mean", random.Random(2))
    assert 1 <= len(selected) <= 4
    assert set(selected).issubset(variables)
    permuted = non_identity_permutation(("a", "b"), random.Random(0))
    assert permuted == ("b", "a")
    permuted_four = non_identity_permutation(variables, random.Random(3))
    assert set(permuted_four) == set(variables)
    assert all(source != destination for source, destination in zip(permuted_four, variables))


def test_gradual_mean_keeps_onset_old_and_records_realized_rows():
    rows = [[str(index), str(index)] for index in range(20)]

    label = inject_drift(
        rows,
        header=("date", "x"),
        method="mean",
        drift_parameters={"offsets": {"x": 10.0}},
        width=4,
        onset=8,
        variables=("x",),
        rng=random.Random(2),
        event_id=0,
    )

    assert rows[8][1] == "8"
    assert float(rows[12][1]) == 22.0
    assert label.start_index == 8
    assert label.end_index_exclusive == 12
    assert label.new_concept_full_from == 12
    assert 0 not in label.parameters["transition_new_concept_offsets"]


def test_abrupt_scale_and_permutation_change_only_selected_variables():
    scale_rows = [[str(index), str(index + 1), str(index + 10)] for index in range(5)]
    scale_label = inject_drift(
        scale_rows,
        header=("date", "a", "b"),
        method="scale",
        drift_parameters={"centers": {"a": 0.0}, "factors": {"a": 2.0}},
        width=0,
        onset=2,
        variables=("a",),
        rng=random.Random(0),
        event_id=0,
    )
    assert scale_rows[1] == ["1", "2", "11"]
    assert scale_rows[2] == ["2", "6.0", "12"]
    assert scale_label.end_index_exclusive == 3
    assert scale_label.new_concept_full_from == 2

    permutation_rows = [[str(index), "10", "20", "30"] for index in range(5)]
    permutation_label = inject_drift(
        permutation_rows,
        header=("date", "a", "b", "c"),
        method="permutation",
        drift_parameters={},
        width=0,
        onset=2,
        variables=("a", "b"),
        rng=random.Random(0),
        event_id=0,
    )
    assert permutation_rows[1] == ["1", "10", "20", "30"]
    assert permutation_rows[2] == ["2", "20", "10", "30"]
    assert permutation_label.parameters["destination_to_source"] == {"a": "b", "b": "a"}


def test_integration_generates_schema_labels_and_batch(tmp_path):
    source = tmp_path / "source.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("date", "a", "b", "OT"))
        for index in range(60):
            writer.writerow((index, index, index + 100, index + 200))
    intervals = tmp_path / "stable_intervals.csv"
    with intervals.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("interval_id", "raw_start", "raw_end_exclusive"))
        writer.writerow((0, 2, 22))
        writer.writerow((1, 30, 50))

    output = tmp_path / "single"
    main(
        [
            "--source", str(source),
            "--stable-intervals", str(intervals),
            "--method", "scale",
            "--min-length", "40",
            "--splice-guard-rows", "0",
            "--seed", "1",
            "--output", str(output),
        ]
    )
    with (output / "data.csv").open("r", encoding="utf-8", newline="") as handle:
        generated = list(csv.reader(handle))
    with (output / "drift_labels.csv").open("r", encoding="utf-8", newline="") as handle:
        labels = list(csv.DictReader(handle))
    with (output / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert generated[0] == ["date", "a", "b", "OT"]
    assert len(generated) == 41
    assert len(labels) == 2
    assert {label["method"] for label in labels} == {"splice", "scale"}
    assert next(label for label in labels if label["method"] == "scale")["start_index"] == str(
        manifest["drift_onset"]
    )
    assert len(manifest["source_segments"]) == 2
    assert manifest["maximum_gradual_width"] == 4
    assert load_benchmark_dataset("etth1", output / "data.csv", 4, 1).num_features == 3
    splice = next(label for label in labels if label["method"] == "splice")
    assert int(splice["start_index"]) == manifest["source_segments"][1]["output_start"]
    full_from = manifest["drift_onset"] + manifest["width"]
    scale_label = next(label for label in labels if label["method"] == "scale")
    scale_parameters = json.loads(scale_label["parameters"])
    for segment in manifest["source_segments"]:
        for position in range(segment["output_start"], segment["output_end_exclusive"]):
            source_index = segment["raw_start"] + position - segment["output_start"]
            source_values = [float(source_index), float(source_index + 100), float(source_index + 200)]
            if position < manifest["drift_onset"]:
                assert [float(value) for value in generated[position + 1][1:]] == source_values
            elif position >= full_from:
                for variable in manifest["affected_variables"]:
                    variable_index = generated[0].index(variable)
                    factor = scale_parameters["factors"][variable]
                    center = scale_parameters["centers"][variable]
                    assert float(generated[position + 1][variable_index]) == (
                        center + (source_values[variable_index - 1] - center) * factor
                    )

    batch_output = tmp_path / "batch"
    batch_main(
        [
            "--source", str(source),
            "--stable-intervals", str(intervals),
            "--output", str(batch_output),
            "--repetitions", "1",
            "--min-length", "40",
            "--splice-guard-rows", "0",
        ]
    )
    assert sorted(path.name for path in batch_output.iterdir()) == [
        "mean-000", "permutation-000", "scale-000"
    ]
    assert all((path / "drift_labels.csv").is_file() for path in batch_output.iterdir())

    repeat = tmp_path / "repeat"
    main(
        [
            "--source", str(source),
            "--stable-intervals", str(intervals),
            "--method", "scale",
            "--min-length", "40",
            "--splice-guard-rows", "0",
            "--seed", "1",
            "--output", str(repeat),
        ]
    )
    assert (repeat / "data.csv").read_bytes() == (output / "data.csv").read_bytes()
    assert (repeat / "drift_labels.csv").read_bytes() == (output / "drift_labels.csv").read_bytes()

    sampled = sample_drift_parameters(
        [["0", "2", "10"], ["1", "4", "14"]],
        ("date", "a", "b"),
        "mean",
        ("a", "b"),
        random.Random(1),
    )
    assert set(sampled["offsets"]) == {"a", "b"}
    assert all(value != 0 for value in sampled["offsets"].values())

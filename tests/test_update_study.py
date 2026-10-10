"""Behavioral checks for update isolation, label timing, and paired experiments."""

import copy
import csv
import json
from itertools import combinations
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from onlinetsf.data import SlidingWindowDataset
from onlinetsf.update_study.analyze import analyze, read_csv
from onlinetsf.update_study.data import load_stream, select_probes
from onlinetsf.update_study.run import build_method, main, run_branch
from onlinetsf.update_study.reporting import Reporter, write_csv
from onlinetsf.update_study.runtime import ControlledMethod, Policy, Replay, restore_rng, rng_state


PROFILES = {
    "fsnet": {"forecasting": {"parameters": {"hidden_channels": 2, "depth": 1,
               "representation_channels": 4, "controller_hidden": 4, "memory_size": 2, "memory_top_k": 1}},
              "method": {"name": "fsnet", "learning_rate": 0.01, "n_inner": 1}},
    "onenet": {"forecasting": {"parameters": {"channels": [2], "dropout": 0.0}},
               "method": {"name": "onenet", "learning_rate": 0.01, "weight_learning_rate": 0.03,
                          "decision_learning_rate": 0.02, "decision_hidden": 4, "decision_dropout": 0.1}},
}


def tiny_replay(name, horizon=2, policy=Policy("native")):
    torch.manual_seed(13)
    dataset = SlidingWindowDataset(torch.randn(64, 2), context_length=4, horizon=horizon)
    method = build_method(name, dataset, PROFILES[name], "cpu")
    return dataset, Replay(ControlledMethod(method, policy), horizon)


@pytest.mark.parametrize("name", ["fsnet", "onenet"])
def test_no_grad_preserves_parameters_and_optimizer_state_but_advances_adaptation(name):
    dataset, replay = tiny_replay(name)
    for index in range(4):
        replay.step(dataset, index)
    method = replay.method.method
    parameters = {key: value.detach().clone() for key, value in method.model.named_parameters()}
    optimizers = {key: copy.deepcopy(value.state_dict()) for key, value in vars(method).items()
                  if isinstance(value, torch.optim.Optimizer)}
    buffers = {key: value.clone() for key, value in method.model.named_buffers()}
    replay.method.set_policy(Policy("no_grad", mode="no_grad"))
    if name == "onenet":
        # No hidden neural backpropagation is allowed in OneNet's formula/state updates.
        with patch.object(torch.Tensor, "backward", side_effect=AssertionError("unexpected backprop")):
            replay.step(dataset, 4)
    else:
        replay.step(dataset, 4)
    assert all(torch.equal(parameters[key], value) for key, value in method.model.named_parameters())
    for key, saved in optimizers.items():
        current = getattr(method, key).state_dict()
        for parameter, state in saved["state"].items():
            for field, value in state.items():
                assert torch.equal(value, current["state"][parameter][field])
    assert any(not torch.equal(buffers[key], value) for key, value in method.model.named_buffers())
    assert replay.method.gradient_steps == 0


def test_egd_is_the_exact_normalized_exponential_update_of_emitted_expert_losses():
    dataset, replay = tiny_replay("onenet", policy=Policy("no_grad", mode="no_grad"))
    replay.step(dataset, 0)
    method = replay.method.method
    time, variable = method._pending_predictions[0]
    target = dataset[0][1]
    losses = torch.stack(((time - target).square().mean(dim=0), (variable - target).square().mean(dim=0)))
    expected = torch.softmax(-method.weight_learning_rate * losses, dim=0)[0]
    # No feedback is available at index 1 for H=2.
    replay.step(dataset, 1)
    assert torch.equal(method.model.long_logits, torch.zeros(2))
    replay.step(dataset, 2)
    assert torch.allclose(method.model.weights(include_short_term=False), expected)


@pytest.mark.parametrize("name", ["fsnet", "onenet"])
def test_frozen_keeps_all_model_state_and_terminal_flush_never_adapts(name):
    dataset, replay = tiny_replay(name, policy=Policy("frozen", mode="frozen"))
    state = copy.deepcopy(replay.method.method.model.state_dict())
    for index in range(5):
        replay.step(dataset, index)
    assert all(torch.equal(value, replay.method.method.model.state_dict()[key]) for key, value in state.items())
    replay.finish(dataset)
    assert all(torch.equal(value, replay.method.method.model.state_dict()[key]) for key, value in state.items())
    assert replay.method.gradient_steps == replay.method.egd_steps == 0
    if name == "onenet":
        assert not replay.method.method._pending_predictions


@pytest.mark.parametrize("name", ["fsnet", "onenet"])
def test_finish_scores_remaining_predictions_without_training_even_for_native(name):
    dataset, replay = tiny_replay(name, horizon=3)
    for index in range(5):
        replay.step(dataset, index)
    state = copy.deepcopy(replay.method.method.model.state_dict())
    counts = (replay.method.gradient_steps, replay.method.egd_steps)
    tail = replay.finish(dataset)
    assert len(tail) == 3
    assert all(record["update"] is None for record in tail)
    assert all(torch.equal(value, replay.method.method.model.state_dict()[key]) for key, value in state.items())
    assert counts == (replay.method.gradient_steps, replay.method.egd_steps)


MODULE_CASES = [(name, frozenset(subset)) for name, groups in
                (("fsnet", ("base", "adapter", "head")), ("onenet", ("time", "variable", "decision")))
                for size in (1, 2) for subset in combinations(groups, size)]


@pytest.mark.parametrize("name,selected", MODULE_CASES)
def test_module_masks_preserve_excluded_parameters_even_with_existing_adam_momentum(name, selected):
    dataset, replay = tiny_replay(name)
    for index in range(4):
        replay.step(dataset, index)
    method = replay.method.method
    if name == "fsnet":
        groups = method.parameter_groups()
    else:
        groups = {"time": list(method.model.cross_time.parameters()),
                  "variable": list(method.model.cross_variable.parameters()),
                  "decision": list(method.model.decision.parameters())}
    before = {group: [value.detach().clone() for value in parameters] for group, parameters in groups.items()}
    replay.method.set_policy(Policy("mask", groups=selected))
    replay.step(dataset, 4)
    for group, parameters in groups.items():
        changes = [not torch.equal(value, old) for value, old in zip(parameters, before[group], strict=True)]
        assert any(changes) if group in selected else not any(changes)
    if name == "fsnet":
        parameters = [id(value) for group in groups.values() for value in group]
        assert len(parameters) == len(set(parameters)) == len(list(method.model.parameters()))


@pytest.mark.parametrize("name", ["fsnet", "onenet"])
def test_period_counts_raw_steps_and_retains_non_gradient_updates_between_training(name):
    dataset, replay = tiny_replay(name, horizon=2, policy=Policy("periodic", period=4))
    groups = []
    for index in range(12):
        for record in replay.step(dataset, index):
            groups.append(record["update"]["groups"])
    assert [position for position, group in enumerate(groups) if group] == [0, 4, 8]
    if name == "onenet":
        assert replay.method.egd_steps == 10


@pytest.mark.parametrize("name", ["fsnet", "onenet"])
def test_future_targets_cannot_affect_current_forecast_or_snapshot(name):
    dataset, replay = tiny_replay(name, horizon=3)
    for index in range(5):
        replay.step(dataset, index)
    changed = copy.deepcopy(dataset)
    clock = 5 + dataset.context_length - 1
    changed.values[clock + 1:] += 10000
    first, second = replay.fork(Policy("native")), replay.fork(Policy("native"))
    state = rng_state()
    first.step(dataset, 5)
    restore_rng(state)
    second.step(changed, 5)
    assert torch.equal(first.pending[-1][1], second.pending[-1][1])
    assert len(first.pending[0]) == 2  # Only origin and immutable emitted prediction; no future labels.


@pytest.mark.parametrize("name", ["fsnet", "onenet"])
def test_forks_copy_optimizer_pending_feedback_and_mutable_memory_without_affecting_parent(name):
    dataset, replay = tiny_replay(name)
    for index in range(5):
        replay.step(dataset, index)
    state = copy.deepcopy(replay.method.method.model.state_dict())
    saved_rng = rng_state()
    first = replay.fork(Policy("native"))
    second = replay.fork(Policy("native"))
    for index in range(5, 9):
        first.step(dataset, index)
    restore_rng(saved_rng)
    for index in range(5, 9):
        second.step(dataset, index)
    assert all(torch.equal(value, second.method.method.model.state_dict()[key])
               for key, value in first.method.method.model.state_dict().items())
    assert all(torch.equal(value, replay.method.method.model.state_dict()[key]) for key, value in state.items())
    assert torch.equal(first.pending[-1][1], second.pending[-1][1])


def write_annotations(root: Path) -> Path:
    directory = root / "Tiny"
    directory.mkdir(parents=True)
    phases = ["calibration"] * 32 + ["stable"] * 14 + ["transition"] * 2 + ["drift"] * 4 + ["transition"] * 2 + ["stable"] * 74
    with (directory / "labeled_data.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "a", "b", "annotation_index", "annotation_phase", "majority_vote_count"])
        for index, phase in enumerate(phases):
            writer.writerow([str(index), index / 10, (index % 7) / 3, index, phase, 123456])
    manifest = {"dataset": "Tiny", "time_column": "date", "feature_columns": ["a", "b"],
                "rows": len(phases), "calibration_rows": 32}
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (directory / "drift_points.csv").write_text("event_id,start_index,end_index_exclusive\n1,48,52\n", encoding="utf-8")
    (directory / "intervals.csv").write_text("phase,start_index,end_index_exclusive\ncalibration,0,32\nstable,32,46\ntransition,46,48\ndrift,48,52\ntransition,52,54\nstable,54,128\n", encoding="utf-8")
    return directory


def test_annotation_columns_are_excluded_and_normalization_uses_only_prefix(tmp_path):
    directory = write_annotations(tmp_path)
    stream = load_stream(directory, 4, 3)
    assert stream.dataset.feature_names == ("a", "b")
    assert stream.dataset.num_features == stream.dataset.num_targets == 2
    assert torch.allclose(stream.dataset.values[:32].mean(dim=0), torch.zeros(2), atol=1e-6)
    assert torch.allclose(stream.dataset.values[:32].std(dim=0, unbiased=False), torch.ones(2))
    assert stream.dataset.values[100, 0] > 5


@pytest.mark.parametrize("calibration,expected", [(16, 26), (32, 32)])
def test_offline_prefix_covers_twenty_percent_or_longer_calibration(tmp_path, calibration, expected):
    directory = write_annotations(tmp_path)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    manifest["calibration_rows"] = calibration
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rows = read_csv(directory / "labeled_data.csv")
    for row in rows:
        if int(row["annotation_index"]) < 32:
            row["annotation_phase"] = "calibration" if int(row["annotation_index"]) < calibration else "stable"
    write_csv(directory / "labeled_data.csv", rows)
    intervals = read_csv(directory / "intervals.csv")
    intervals[0]["end_index_exclusive"] = calibration
    intervals[1]["start_index"] = calibration
    write_csv(directory / "intervals.csv", intervals)
    stream = load_stream(directory, 4, 3)
    assert stream.prefix_rows == expected
    assert stream.prefix_rows / len(stream.phases) >= 0.2
    assert torch.allclose(stream.dataset.values[:expected].mean(dim=0), torch.zeros(2), atol=1e-6)
    with pytest.raises(ValueError, match="20%"):
        load_stream(directory, 4, 3, prefix_rows=expected - 1)


def test_probe_sampling_is_deterministic_capped_separated_and_leaves_full_scoring_windows(tmp_path):
    stream = load_stream(write_annotations(tmp_path), 4, 3)
    options = dict(max_probes=3, max_events=1, separation=8, branch_steps=8, offset=2, seed=2026)
    first = select_probes(stream, 28, len(stream.dataset), **options)
    assert first == select_probes(stream, 28, len(stream.dataset), **options)
    assert 0 < len(first) <= 3
    assert all(item["index"] + 8 <= len(stream.dataset) for item in first)
    assert all(right["index"] - left["index"] >= 8 for left, right in zip(first, first[1:]))
    for item in first:
        if item["kind"] == "stable":
            assert set(stream.phases[item["index"]:item["index"] + 4 + 8 + 3 - 1]) == {"stable"}


def test_branch_scoring_excludes_inherited_predictions_and_never_trains_on_tail(tmp_path):
    stream = load_stream(write_annotations(tmp_path), 4, 3)
    replay = Replay(ControlledMethod(build_method("onenet", stream.dataset, PROFILES["onenet"], "cpu"), Policy("native")), 3)
    for index in range(28, 33):
        replay.step(stream.dataset, index)
    origin, prediction = replay.pending[0]
    replay.pending[0] = (origin, prediction + 1000000)
    outcomes = run_branch(replay, stream, {"index": 33}, Policy("frozen", mode="frozen"), tmp_path / "branch", 8, [2, 8], True)
    forecasts = read_csv(tmp_path / "branch" / "forecasts.csv")
    assert len(forecasts) == 8
    assert min(int(row["forecast_index"]) for row in forecasts) == 33
    assert outcomes[1]["mae"] == pytest.approx(sum(float(row["mae"]) for row in forecasts) / 8)
    tensors = torch.load(tmp_path / "branch" / "predictions_00000.pt", weights_only=True)
    assert tensors["predictions"].shape == (8, 3, 2)
    assert not read_csv(tmp_path / "branch" / "updates.csv")


@pytest.mark.parametrize("name", ["fsnet", "onenet"])
def test_end_to_end_probes_do_not_change_native_replay_and_analysis_is_reproducible(tmp_path, name):
    annotation_root = tmp_path / "annotations"
    write_annotations(annotation_root)
    profile = tmp_path / "profiles.yaml"
    import yaml
    profile.write_text(yaml.safe_dump({"strategies": PROFILES}), encoding="utf-8")
    common = ["--annotations", str(annotation_root), "--datasets", "Tiny", "--profiles", str(profile),
              "--methods", name, "--horizons", "1", "3", "--context-length", "4", "--epochs", "1",
              "--branch-steps", "8", "--lookaheads", "2", "8", "--probe-separation", "8",
              "--probe-offset", "2", "--max-events", "1", "--max-steps", "64", "--lr-scales", "1",
              "--scope", "branches"]
    with_probes, no_probes = tmp_path / "with", tmp_path / "without"
    main(common + ["--max-probes", "3", "--output", str(with_probes)])
    main(common + ["--max-probes", "0", "--output", str(no_probes)])
    for horizon in (1, 3):
        job = Path("Tiny") / f"{name}_h{horizon}_seed0"
        assert (with_probes / job / "native" / "forecasts.csv").read_bytes() == (no_probes / job / "native" / "forecasts.csv").read_bytes()
        outcomes = read_csv(with_probes / job / "probe_outcomes.csv")
        assert {row["policy"] for row in outcomes} >= {"native", "frozen", "no_grad"}
    before = (with_probes / "paired_benefits.csv").read_bytes()
    analyze(with_probes)
    assert before == (with_probes / "paired_benefits.csv").read_bytes()
    assert json.loads((with_probes / "manifest.json").read_text(encoding="utf-8"))["status"] == "complete"


def test_learning_rate_controls_do_not_change_egd_or_offline_state():
    _, replay = tiny_replay("onenet")
    method = replay.method.method
    before = copy.deepcopy(method.model.state_dict())
    branch = replay.fork(Policy("smaller", lr_scale=0.3))
    assert branch.method.method.expert_optimizer.param_groups[0]["lr"] == pytest.approx(0.003)
    assert branch.method.method.decision_optimizer.param_groups[0]["lr"] == pytest.approx(0.006)
    assert branch.method.method.weight_learning_rate == method.weight_learning_rate
    assert all(torch.equal(value, branch.method.method.model.state_dict()[key]) for key, value in before.items())


def test_phase_metrics_follow_target_rows_while_updates_follow_feedback_clock(tmp_path):
    stream = load_stream(write_annotations(tmp_path), 4, 3)
    reporter = Reporter(tmp_path / "reported", stream, 41)
    error = torch.tensor([[1.0, 3.0], [2.0, 4.0], [3.0, 5.0]])
    reporter.add([{"index": 41, "available_at": 44, "mae": 3.0, "mse": 64 / 6,
                   "error": error, "prediction": error, "target": torch.zeros_like(error), "expert_errors": None,
                   "update": {"forecast_index": 41, "available_at": 44, "groups": "head", "gradient_steps": 1,
                              "egd_steps": 0, "adaptation_loss": 1.0, "seconds": 0.1}}])
    _, replay = tiny_replay("fsnet", horizon=3)
    summary = reporter.close(replay.method, 0.5)
    rows = read_csv(tmp_path / "reported" / "phase_metrics.csv")
    stable = next(row for row in rows if row["phase"] == "stable" and row["horizon_step"] == "0")
    transition = next(row for row in rows if row["phase"] == "transition" and row["horizon_step"] == "0")
    assert float(stable["mae"]) == 2.0 and int(stable["values"]) == 2
    assert float(transition["mae"]) == 3.5 and int(transition["values"]) == 4
    assert summary["mae"] == 3.0
    assert read_csv(tmp_path / "reported" / "updates.csv")[0]["feedback_phase"] == "transition"


def test_gpu_shard_merge_never_counts_intermediate_summaries_twice(tmp_path):
    for gpu, method in enumerate(("fsnet", "onenet")):
        shard = tmp_path / f"gpu{gpu}"
        job = shard / "Tiny" / f"{method}_h1_seed0"
        job.mkdir(parents=True)
        (job / "policies.json").write_text("[]", encoding="utf-8")
        identity = dict(dataset="Tiny", method=method, horizon=1, seed=0, probe_id="stable_40")
        write_csv(job / "summary.csv", [dict(dataset="Tiny", method=method, policy="native", mae=0.5)])
        write_csv(job / "probe_states.csv", [dict(identity, kind="stable", index=37, clock=40, event_id=0,
                                                   event_rows=0, input_mean_32=0.2)])
        write_csv(job / "probe_outcomes.csv", [dict(identity, kind="stable", lookahead=8, policy="native", mae=0.5),
                                               dict(identity, kind="stable", lookahead=8, policy="frozen", mae=0.6)])
        analyze(shard)
    analyze(tmp_path)
    assert len(read_csv(tmp_path / "summary.csv")) == 2
    assert len(read_csv(tmp_path / "paired_benefits.csv")) == 4
    analyze(tmp_path)
    assert len(read_csv(tmp_path / "summary.csv")) == 2

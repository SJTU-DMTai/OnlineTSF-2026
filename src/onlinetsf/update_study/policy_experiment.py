"""Train selectors on strictly earlier outcomes, then replay a long test interval.

Standalone experiment 2: no experiment-1 output is required. Plan mode fits
selectors from existing CSV results without importing torch. Actual causal replay
runs on the server, reusing initial.pt and preserving state throughout testing.
With --reuse-experiment, only Full is fitted and replayed; all other controllers
and their measured costs are retained from the completed source experiment.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .policy_support import (
    DEFAULT_ANNOTATIONS, DEFAULT_RUNS, file_sha256, find_job,
    load_existing_results, plot_forecasts, save_json, save_timings,
    oracle_decision, score_action, summarize_decisions,
)

from ..decision.config import DecisionConfig, FEATURE_SCHEMA_VERSION
from ..decision.training import fit_selector, historical_training_data


def block_metrics(forecasts: pd.DataFrame, decisions: list[dict], interval: int,
                  phases: np.ndarray) -> pd.DataFrame:
    """Keep complete local windows and the final partial block explicitly separate."""
    rows = []
    for block, offset in enumerate(range(0, len(forecasts), interval)):
        selected = forecasts.iloc[offset:offset + interval]
        clock = int(selected.origin_raw_index.iloc[0])
        for lookahead in sorted({min(8, len(selected)), min(32, len(selected)), len(selected)}):
            prefix = selected.iloc[:lookahead]
            rows.append({"block": block, "clock": clock, "block_steps": len(selected),
                         "phase_at_decision": phases[clock], "lookahead": lookahead,
                         "selected_action": decisions[block]["selected_action"],
                         "mae": prefix.mae.mean(), "mse": prefix.mse.mean()})
    return pd.DataFrame(rows)


def run_segment(replay, stream, start: int, steps: int, interval: int,
                controller: str, bundle: dict, actions: dict, directory: Path,
                *, oracle_metric: str | None = None) -> tuple[pd.DataFrame, list[dict], dict]:
    from onlinetsf.update_study.reporting import Reporter
    from onlinetsf.update_study.runtime import restore_rng, rng_state, state_features
    from onlinetsf.decision.runtime import DecisionEngine

    branch = replay.fork(actions["native"])
    engine = DecisionEngine(bundle, controller, branch.method.method)
    reporter = Reporter(directory, stream, start)
    decisions = []
    oracle_rows, outcomes = [], []
    update_clocks = set()
    stream_seconds = update_seconds = decision_seconds = search_seconds = 0.0
    search_gradient_steps = search_egd_steps = 0
    if branch.method.method.device.type == "cuda":
        import torch

        torch.cuda.synchronize(branch.method.method.device)
    began = perf_counter()
    for index in range(start, start + steps):
        if (index - start) % interval == 0:
            decision_began = perf_counter()
            clock = index + stream.dataset.context_length - 1
            features = state_features(stream, branch, index, names=engine.feature_names) if engine.feature_names else {}
            policy, scores = engine.decide(features)
            action = policy.name
            decision_seconds += perf_counter() - decision_began
            if oracle_metric is not None and start + steps - index >= interval:
                saved_rng = rng_state()
                search_began = perf_counter()
                local = []
                try:
                    for candidate in actions.values():
                        restore_rng(saved_rng)
                        local.append(score_action(branch, stream, index, interval, candidate))
                finally:
                    restore_rng(saved_rng)
                search_seconds += perf_counter() - search_began
                search_gradient_steps += sum(row["gradient_steps"] for row in local)
                search_egd_steps += sum(row["egd_steps"] for row in local)
                block = (index - start) // interval
                outcomes.extend(dict(block=block, clock=clock, block_steps=interval, **row) for row in local)
                oracle_rows.append(dict(block=block, clock=clock, **oracle_decision(local, action, oracle_metric)))
            decision_began = perf_counter()
            branch.method.switch_policy(policy)
            decision_seconds += perf_counter() - decision_began
            record = {"controller": controller, "decision_step": index - start + 1,
                      "steps_since_gradient": branch.method.gradient_age(index),
                      "clock": clock, "selected_action": action,
                      "phase_at_decision": stream.phases[clock]}
            for feature in bundle["selectors"]["policy_full"]["features"]:
                record[feature] = features.get(feature)
            if scores is not None:
                record["predicted_gains"] = json.dumps(scores, allow_nan=False)
            decisions.append(record)
        previous_steps = branch.method.gradient_steps
        stream_began = perf_counter()
        records = branch.step(stream.dataset, index)
        stream_seconds += perf_counter() - stream_began
        if branch.method.gradient_steps > previous_steps:
            update_clocks.add(index)
        update_seconds += sum(row["update"]["seconds"] for row in records if row["update"] is not None)
        reporter.add(records)
        if (index - start + 1) % 2048 == 0:
            print(f"{controller}: {index - start + 1}/{steps} test origins", flush=True)
    # Score all emitted forecasts. Terminal feedback never trains or emits forecasts.
    stream_began = perf_counter()
    reporter.add(branch.finish(stream.dataset))
    stream_seconds += perf_counter() - stream_began
    wall_seconds = perf_counter() - began
    summary = reporter.close(branch.method, wall_seconds - search_seconds)
    summary.update(gradient_update_clocks=len(update_clocks), stream_seconds=stream_seconds,
                   update_seconds=update_seconds, decision_seconds=decision_seconds,
                   candidate_search_seconds=search_seconds, candidate_gradient_steps=search_gradient_steps,
                   candidate_egd_steps=search_egd_steps, online_seconds=wall_seconds - search_seconds,
                   wall_seconds=wall_seconds, candidate_search_role="diagnostic" if oracle_metric is not None else "none")
    forecasts = pd.read_csv(directory / "forecasts.csv")
    if len(forecasts) != steps or not np.array_equal(forecasts.forecast_index, np.arange(start, start + steps)):
        raise RuntimeError("segment did not score each newly emitted forecast exactly once")
    if oracle_metric is not None:
        local_frame = pd.DataFrame(oracle_rows)
        local_frame.to_csv(directory / "oracle_decisions.csv", index=False)
        pd.DataFrame(outcomes, columns=["block", "clock", "block_steps", "action", "mae", "mse", "forecasts",
                                      "gradient_steps", "egd_steps", "seconds"]).to_csv(directory / "action_outcomes.csv", index=False)
        for row in local_frame.itertuples():
            actual = forecasts.iloc[row.block * interval:(row.block + 1) * interval][oracle_metric].mean()
            if not np.isclose(actual, row.selected_loss, rtol=1e-7, atol=1e-9):
                raise RuntimeError("selected Oracle branch differs from the automatic trajectory")
        local_summary = (summarize_decisions(local_frame, oracle_metric) if not local_frame.empty else
                         {"metric": oracle_metric, "decisions": 0, "acc": None, "mean_relative_regret": None})
        save_json(directory / "oracle_summary.json", local_summary)
    return forecasts, decisions, summary


def main(argv: list[str] | None = None) -> None:
    began = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--training-data", type=Path,
                        help="completed collect_policy_training.py output for one training seed")
    parser.add_argument("--reuse-experiment", type=Path,
                        help="completed stage-3 directory; rerun only Full and reuse the other trajectories")
    parser.add_argument("--controllers", nargs="+",
                        choices=["native", "best_fixed", "policy_full", "policy_data", "frozen"],
                        default=["native", "best_fixed", "policy_full", "policy_data"],
                        help="test only these controllers; Native-relative metrics are omitted without native")
    parser.add_argument("--dataset", default="ETTh2")
    parser.add_argument("--method", choices=["onenet", "fsnet", "dsof"], default="onenet")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--test-seed", type=int, default=2)
    boundary = parser.add_mutually_exclusive_group()
    boundary.add_argument("--test-start", type=int, help="first test forecast origin, raw zero-based row")
    boundary.add_argument("--test-start-fraction", type=float, default=.6, help="default: floor(0.6 * raw row count)")
    parser.add_argument("--steps", type=int, help="default: every remaining prediction origin")
    parser.add_argument("--decision-interval", type=int, help="positive interval matching collected branch length")
    for policy in ("data", "full"):
        parser.add_argument(f"--{policy}-max-depth", type=int)
        parser.add_argument(f"--{policy}-min-samples-leaf", type=int,
                            help="minimum training states per leaf; a split requires at least twice this number")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--oracle-diagnostics", choices=["none", "policy_full", "policy_data", "both"], default="none",
                        help="optional ACC diagnostic branches (default: none); does not change policy decisions or stage-4 Oracle")
    parser.add_argument("--oracle-metric", choices=["mae", "mse"], default="mae")
    parser.add_argument("--plan-only", action="store_true", help="fit and audit the historical selector without importing torch")
    parser.add_argument("--output", type=Path, default=Path("runs/policy-exp2-temporal-onenet-etth2-h24"))
    args = parser.parse_args(argv)
    if len(args.controllers) != len(set(args.controllers)):
        parser.error("controllers must be unique")
    if args.reuse_experiment is not None and args.controllers != ["native", "best_fixed", "policy_full", "policy_data"]:
        parser.error("--reuse-experiment requires the default controllers")
    collected = None
    if args.training_data is not None:
        collected = json.loads((args.training_data / "manifest.json").read_text(encoding="utf-8"))
        decision = DecisionConfig.from_dict(collected["decision_config"])
    else:
        source_manifest = json.loads((args.runs / "manifest.json").read_text(encoding="utf-8"))
        decision = DecisionConfig.from_dict(source_manifest["decision_config"])
    args.decision_interval = decision.interval if args.decision_interval is None else args.decision_interval
    tree_overrides = {}
    for policy in ("data", "full"):
        for parameter in ("max_depth", "min_samples_leaf"):
            key = f"{policy}_{parameter}"
            value = getattr(args, key)
            value = getattr(decision, key) if value is None else value
            if value <= 0:
                parser.error(f"--{key.replace('_', '-')} must be positive")
            setattr(args, key, value)
            tree_overrides[key] = value
    from dataclasses import replace

    decision = replace(decision, interval=args.decision_interval, **tree_overrides)
    if (min(args.horizon, args.threads, args.decision_interval) <= 0
            or args.steps is not None and args.steps <= 0):
        parser.error("sizes must be positive")
    if not args.train_seeds or len(args.train_seeds) != len(set(args.train_seeds)):
        parser.error("training seeds must be nonempty and unique")
    if not 0 < args.test_start_fraction < 1:
        parser.error("test-start-fraction must be between zero and one")
    interval = args.decision_interval
    dataset, method, horizon, seed = args.dataset, args.method, args.horizon, args.test_seed
    jobs = {item: find_job(args.runs, dataset, method, horizon, item) for item in sorted(set([*args.train_seeds, seed]))}
    job = jobs[seed]
    checkpoint_path = job / "initial.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    annotation_dir = args.annotations / dataset
    annotation_manifest = json.loads((annotation_dir / "manifest.json").read_text(encoding="utf-8"))
    source_hash = file_sha256(annotation_dir / "labeled_data.csv")
    if collected is not None:
        if collected.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
            raise ValueError("recollect training states with the current feature schema")
        if collected.get("data_feature_precision") != "float64":
            raise ValueError("recollect training states using float64 decision inputs")
        if (collected["status"] != "complete" or args.train_seeds != [collected["seed"]]
                or any(collected[key] != value for key, value in
                       [("dataset", dataset), ("method", method), ("horizon", horizon),
                        ("decision_interval", interval), ("input_sha256", source_hash)])):
            raise ValueError("collected training data identity, interval or completion status differs")
        if collected["checkpoint_sha256"] != file_sha256(jobs[collected["seed"]] / "initial.pt"):
            raise ValueError("collected training data used a different checkpoint")
    catalogs, context_lengths, verified_prefixes = [], [], []
    state_frames, outcome_frames = [], []
    source_audit = []
    for item, source_job in jobs.items():
        catalogs.append(collected["actions"] if collected is not None else
                        json.loads((source_job / "policies.json").read_text(encoding="utf-8")))
        source_manifest = source_job.parent.parent / "manifest.json"
        source_record = json.loads(source_manifest.read_text(encoding="utf-8"))
        if source_record.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
            raise ValueError("regenerate the source checkpoint with the current feature schema")
        if collected is None and source_record.get("data_feature_precision") != "float64":
            raise ValueError("regenerate historical training states using float64 decision inputs")
        record = next(record for record in source_record["jobs"] if record["dataset"] == dataset and
                      record["method"] == method and record["horizon"] == horizon and record["seed"] == item)
        if record["input_sha256"] != source_hash:
            raise ValueError("annotations differ from source run")
        verified_prefixes.append(record["prefix_rows"])
        audit = {"seed": item, "job": str(source_job), "manifest": str(source_manifest)}
        if item in args.train_seeds:
            training_dir = args.training_data if collected is not None else source_job
            states_hash = file_sha256(training_dir / "probe_states.csv")
            outcomes_hash = file_sha256(training_dir / "probe_outcomes.csv")
            if collected is not None and (states_hash != collected["states_sha256"]
                                          or outcomes_hash != collected["outcomes_sha256"]):
                raise ValueError("collected training CSV hashes differ from collection manifest")
            states = pd.read_csv(training_dir / "probe_states.csv")
            outcomes = pd.read_csv(training_dir / "probe_outcomes.csv")
            for frame in [states, outcomes]:
                if not (frame.seed.eq(item) & frame.dataset.eq(dataset) & frame.method.eq(method) & frame.horizon.eq(horizon)).all():
                    raise ValueError("source row identity differs from requested job")
            context_lengths.extend((states.clock - states["index"] + 1).unique().tolist())
            state_frames.append(states)
            outcome_frames.append(outcomes)
            audit.update(training_directory=str(training_dir), states_sha256=states_hash,
                         outcomes_sha256=outcomes_hash)
        source_audit.append(audit)
    if any(catalog != catalogs[0] for catalog in catalogs[1:]):
        raise ValueError("source action definitions differ")
    if len(set(context_lengths)) != 1 or len(set(verified_prefixes)) != 1:
        raise ValueError("source jobs used inconsistent context lengths or offline prefixes")
    prefix = verified_prefixes[0]
    context_length = int(context_lengths[0])
    phases = pd.read_csv(annotation_dir / "labeled_data.csv", usecols=["annotation_phase"]).annotation_phase.to_numpy()
    clock = args.test_start if args.test_start is not None else int(len(phases) * args.test_start_fraction)
    if collected is not None and (collected["test_start"] != clock or collected["prefix_rows"] != prefix
                                  or collected["context_length"] != context_length):
        raise ValueError("collection and test replay must use the same time boundary, prefix and context length")
    available_steps = len(phases) - horizon - clock
    steps = available_steps if args.steps is None else args.steps
    if prefix < context_length + horizon or not prefix <= clock or not 0 < steps <= available_steps:
        parser.error("test-start must follow the offline prefix and leave complete forecast targets")
    catalog = catalogs[0]
    names = [item["name"] for item in catalog]
    if len(names) != len(set(names)) or "native" not in names:
        raise ValueError("source catalog must have unique actions and native")
    names = ["native"] + [name for name in names if name != "native"]
    catalog = sorted(catalog, key=lambda item: names.index(item["name"]))
    train, losses, split_audit = historical_training_data(
        pd.concat(state_frames, ignore_index=True), pd.concat(outcome_frames, ignore_index=True),
        args.train_seeds, clock, interval, horizon, names, "mae")
    if (train.clock < prefix - 1).any():
        raise ValueError("historical online training states precede the offline prefix")
    capabilities = collected if collected is not None else record
    feature_sets = decision.feature_sets(capabilities["method_features"])
    reused_manifest = reused_bundle = None
    if args.reuse_experiment is not None:
        reused_manifest = json.loads((args.reuse_experiment / "manifest.json").read_text(encoding="utf-8"))
        reused_path = args.reuse_experiment / "selectors.json"
        reused_bundle = json.loads(reused_path.read_text(encoding="utf-8"))
        if (reused_manifest["status"] != "complete" or reused_bundle["schema_version"] != FEATURE_SCHEMA_VERSION
                or file_sha256(reused_path) != reused_manifest["selector_sha256"]):
            raise ValueError("reused experiment must be complete with its original saved selectors")
        expected = {"dataset": dataset, "method": method, "horizon": horizon, "train_seeds": args.train_seeds,
                    "test_seed": seed, "test_start": clock, "steps": steps, "prefix_rows": int(prefix),
                    "decision_interval": interval, "input_sha256": source_hash,
                    "checkpoint_sha256": file_sha256(checkpoint_path)}
        for key, value in expected.items():
            if reused_manifest[key] != value:
                raise ValueError(f"reused experiment differs on {key}")
        for key, value in {"actions": catalog, "context_length": context_length,
                           "best_fixed": losses.mean().idxmin()}.items():
            if reused_bundle[key] != value:
                raise ValueError(f"reused experiment differs on {key}")
        old_config = DecisionConfig.from_dict(reused_bundle["decision_config"]).to_dict()
        for key, value in decision.to_dict().items():
            if key not in {"full_max_depth", "full_min_samples_leaf"} and old_config[key] != value:
                raise ValueError(f"Full-only reuse cannot change decision.{key}")
        old_sources = {item["seed"]: (item["states_sha256"], item["outcomes_sha256"])
                       for item in reused_manifest["source_jobs"] if "states_sha256" in item}
        new_sources = {item["seed"]: (item["states_sha256"], item["outcomes_sha256"])
                       for item in source_audit if "states_sha256" in item}
        if old_sources != new_sources:
            raise ValueError("reused experiment used different historical training data")
        for name, features in feature_sets.items():
            if reused_bundle["selectors"][name]["features"] != features:
                raise ValueError(f"reused experiment differs on {name} features")
        if (args.oracle_diagnostics in {"policy_data", "both"}
                and not (args.reuse_experiment / "policy_data/action_outcomes.csv").exists()):
            raise ValueError("Full-only reuse cannot add Data diagnostics; use --oracle-diagnostics policy_full or none")
    for name in feature_sets:
        if reused_bundle is not None and name == "policy_data":
            continue
        leaf_size = decision.tree_parameters(name)["min_samples_leaf"]
        if len(train) < 2 * leaf_size:
            option = f"--{name.removeprefix('policy_')}-min-samples-leaf"
            parser.error(f"{name}: only {len(train)} historical training states remain after time isolation; "
                         f"{option} {leaf_size} requires at least {2 * leaf_size} to split. "
                         f"Reduce {option} or collect more historical probes before running the forecasting replay.")
    missing = set(feature_sets["policy_full"]) - set(train.columns)
    if missing:
        raise ValueError(f"training states lack configured features: {sorted(missing)}; collect them first")
    gains = (1 - losses.div(losses.native, axis=0)) * 100
    # Load the library before measuring fits, so the first selector does not
    # absorb sklearn's one-time import cost. This belongs to preparation time.
    import sklearn.tree

    timings = {"preparation_seconds": perf_counter() - began}
    if collected is not None:
        timings.update(collected.get("timings", {}))
    selectors, rules, diagnostics = {}, [], []
    for name, features in feature_sets.items():
        parameters = decision.tree_parameters(name)
        if reused_bundle is not None and name == "policy_data":
            selectors[name] = reused_bundle["selectors"][name]
            text = f"Reused unchanged from {args.reuse_experiment}; see its tree_rules.txt."
            timings[f"selector_{name}_seconds"] = reused_manifest["timings"][f"selector_{name}_seconds"]
        else:
            fit_began = perf_counter()
            selectors[name], text = fit_selector(train, gains, names, features, **parameters)
            timings[f"selector_{name}_seconds"] = perf_counter() - fit_began
        rules.append(name + "\n" + text)
        for action, tree in selectors[name]["trees"].items():
            diagnostics.append({"controller": name, "action": action, "training_states": len(train),
                                **parameters,
                                "nodes": len(tree["value"]), "leaves": tree["left"].count(-1)})
        split_count = sum(len(tree["value"]) > 1 for tree in selectors[name]["trees"].values())
        print(f"{name}: max_depth={parameters['max_depth']}, min_samples_leaf={parameters['min_samples_leaf']}; "
              f"{len(train)} historical states; "
              f"{split_count}/{len(selectors[name]['trees'])} action trees split on features", flush=True)
    bundle = {"schema_version": FEATURE_SCHEMA_VERSION, "data_feature_precision": "float64", "decision_config": decision.to_dict(),
              "update_groups": capabilities["update_groups"], "dataset": dataset, "method": method, "horizon": horizon,
              "context_length": context_length, "prefix_rows": int(prefix), "input_sha256": source_hash,
              "train_seeds": args.train_seeds, "test_seed": seed, "test_start": clock,
              "decision_interval": interval, "metric": "mae", "actions": catalog,
              "best_fixed": losses.mean().idxmin(), "selectors": selectors,
              "training_positions": train.probe_id.nunique(), "training_states": len(train),
              "train_target_end_exclusive": int((train.clock + interval + horizon).max()),
              "selection_rule": "training target_end_exclusive <= test_start"}
    plan = {"status": "planned", "dataset": dataset, "method": method, "horizon": horizon,
            "train_seeds": args.train_seeds, "test_seed": seed, "split": "strict_time",
            "test_start": clock, "test_origin_end_exclusive": clock + steps,
            "test_target_end_exclusive": clock + steps + horizon, "prefix_rows": int(prefix),
            "steps": steps, "decision_interval": interval, "action_count": len(catalog),
            "training_positions": bundle["training_positions"], "training_states": len(train),
            "train_target_end_exclusive": bundle["train_target_end_exclusive"],
            "controllers": args.controllers,
            "input_sha256": source_hash, "source_jobs": source_audit,
            "checkpoint": str(checkpoint_path), "checkpoint_sha256": file_sha256(checkpoint_path),
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "timings": timings}
    plan["executed_controllers"] = ["policy_full"] if reused_bundle is not None else plan["controllers"]
    plan["reused_controllers"] = ["native", "best_fixed", "policy_data"] if reused_bundle is not None else []
    if reused_manifest is not None:
        plan["reused_experiment"] = str(args.reuse_experiment.resolve())
        plan["reused_selector_sha256"] = reused_manifest["selector_sha256"]
        plan["timing_definition"] = ("reused controller and Data-fit costs retain original measurements; "
                                     "experiment_wall_seconds measures only this invocation")
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "selectors.json", bundle)
    plan["selector_sha256"] = file_sha256(args.output / "selectors.json")
    split_audit.to_csv(args.output / "training_split.csv", index=False)
    train.to_csv(args.output / "training_states.csv", index=False)
    losses.assign(seed=train.seed, probe_id=train.probe_id).to_csv(args.output / "training_losses.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(args.output / "selector_diagnostics.csv", index=False)
    (args.output / "tree_rules.txt").write_text("\n".join(rules), encoding="utf-8")
    save_json(args.output / "manifest.json", plan)
    print(json.dumps(plan, indent=2), flush=True)
    timings["experiment_wall_seconds"] = perf_counter() - began
    save_timings(args.output, timings)
    save_json(args.output / "manifest.json", plan)
    if args.plan_only:
        return

    # The forecasting runtime is only loaded on this execution path, on the server.
    setup_began = perf_counter()
    import torch
    from onlinetsf.update_study.data import load_stream
    from onlinetsf.update_study.runtime import ControlledMethod, Policy, Replay, load_replay, restore_rng, rng_state

    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    stream = load_stream(args.annotations / dataset, bundle["context_length"], horizon, bundle["prefix_rows"])
    if stream.manifest["feature_columns"] != annotation_manifest["feature_columns"]:
        raise ValueError("forecast feature columns changed")
    actions = {item["name"]: Policy.from_dict(item) for item in bundle["actions"]}
    start = clock - stream.dataset.context_length + 1
    reuse_history = reused_manifest is not None or collected is not None and collected["seed"] == seed
    if reuse_history:
        history_source = reused_manifest if reused_manifest is not None else collected
        history_path = (args.reuse_experiment if reused_manifest is not None else args.training_data) / "evaluation_start.pt"
        if file_sha256(history_path) != history_source["evaluation_start_sha256"]:
            raise ValueError("saved test-start state hash differs")
        replay, saved_rng = load_replay(history_path, args.device)
        if replay.last_index != start - 1 or replay.horizon != horizon or replay.method.policy != actions["native"]:
            raise ValueError("collected state is not the native state immediately before testing")
        if replay.config.feature_window != decision.feature_window or replay.config.error_window != decision.error_window:
            raise ValueError("saved feature history configuration differs")
        replay.config = decision
        forecasting_method = replay.method.method
        restore_rng(saved_rng)
    else:
        # initial.pt is this project's trusted, locally produced full-method checkpoint.
        checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
        if checkpoint["prefix_rows"] != bundle["prefix_rows"]:
            raise ValueError("checkpoint prefix differs from selector")
        forecasting_method = checkpoint["method"]
        forecasting_method.device = torch.device(args.device)
        forecasting_method.model.to(forecasting_method.device)
        initial_rng = (checkpoint["rng"][0].cpu(),
                       [item.cpu() for item in checkpoint["rng"][1]] if checkpoint["rng"][1] is not None else None)
        if initial_rng[1] is not None and torch.cuda.is_available() and len(initial_rng[1]) != torch.cuda.device_count():
            raise ValueError("match source visible GPU count (normally set CUDA_VISIBLE_DEVICES to one GPU)")
        restore_rng(initial_rng)
        replay = Replay(ControlledMethod(forecasting_method, actions["native"]), horizon, decision)
    outcomes, decisions, trajectory_summaries = [], [], []
    plan["status"] = "running"
    save_json(args.output / "manifest.json", plan)
    try:
        if forecasting_method.device.type == "cuda":
            torch.cuda.synchronize(forecasting_method.device)
        timings["runtime_setup_seconds"] = perf_counter() - setup_began
        if reuse_history:
            timings["warmup_seconds"] = 0.0
            plan["evaluation_start_source"] = ("experiment_saved_state" if reused_manifest is not None else
                                                "collection_saved_state")
            shutil.copyfile(history_path, args.output / "evaluation_start.pt")
        else:
            warmup_began = perf_counter()
            first = stream.prefix_rows - stream.dataset.context_length
            for index in range(first, start):
                replay.step(stream.dataset, index)
                if (index - first) % 2048 == 0:
                    print(f"native warmup: raw clock {index + stream.dataset.context_length - 1}", flush=True)
            if forecasting_method.device.type == "cuda":
                torch.cuda.synchronize(forecasting_method.device)
            timings["warmup_seconds"] = perf_counter() - warmup_began
            plan["evaluation_start_source"] = "native_history_for_test_seed"
            saved_rng = rng_state()
            torch.save({"replay": replay, "rng": saved_rng}, args.output / "evaluation_start.pt")
        plan["evaluation_start_sha256"] = file_sha256(args.output / "evaluation_start.pt")
        frames = {}
        if reused_manifest is not None:
            reused_decisions = pd.read_csv(args.reuse_experiment / "decisions.csv")
            reused_summaries = pd.read_csv(args.reuse_experiment / "trajectory_summary.csv")
        for controller in plan["controllers"]:
            restore_rng(saved_rng)
            directory = args.output / controller
            reused = controller in plan["reused_controllers"]
            if reused:
                frame, _ = load_existing_results(args.reuse_experiment, controller, start, steps)
                selected = reused_decisions[reused_decisions.controller == controller]
                if not np.array_equal(selected.clock, np.arange(clock, clock + steps, interval)):
                    raise ValueError(f"{controller}: reused decisions do not cover the test interval")
                trajectory_decisions = selected.to_dict("records")
                summary = reused_summaries[reused_summaries.controller == controller].iloc[0].to_dict()
                shutil.copytree(args.reuse_experiment / controller, directory)
            else:
                frame, trajectory_decisions, summary = run_segment(
                    replay, stream, start, steps, interval, controller, bundle, actions, directory,
                    oracle_metric=args.oracle_metric if controller.startswith("policy_") and
                    args.oracle_diagnostics in {controller, "both"} else None)
            summary.update(result_source="experiment_saved" if reused else "experiment_measured",
                           executed_in_experiment=not reused)
            frames[controller] = frame
            decisions.extend(dict(item, seed=seed) for item in trajectory_decisions)
            trajectory_summaries.append(dict(summary, controller=controller, seed=seed))
            timings[f"test_{controller}_seconds"] = summary["seconds"]
            timings[f"test_{controller}_candidate_search_seconds"] = summary["candidate_search_seconds"]
            outcomes.append(block_metrics(frame, trajectory_decisions, interval, phases).assign(controller=controller, seed=seed))
            pd.DataFrame(decisions).to_csv(args.output / "decisions.csv", index=False)
            pd.DataFrame(trajectory_summaries).to_csv(args.output / "trajectory_summary.csv", index=False)
            save_timings(args.output, timings)
            print(f"{controller}: {steps} test origins {'reused' if reused else 'complete'}", flush=True)
        postprocessing_began = perf_counter()
        plot_forecasts(frames, args.output / "test_errors", f"{dataset} {method} H={horizon} | start {clock}")
        result = pd.concat(outcomes, ignore_index=True)
        local_summary = result.groupby(["controller", "lookahead"]).agg(
            mae=("mae", "mean"), mse=("mse", "mean"), blocks=("block", "count")).reset_index()
        summary = pd.DataFrame(trajectory_summaries)
        phase_metrics = ["mae", "mse"]
        if "native" in args.controllers:
            keys = ["seed", "block", "lookahead"]
            native = result[result.controller == "native"][keys + ["mae", "mse"]].rename(columns={"mae": "native_mae", "mse": "native_mse"})
            result = result.merge(native, on=keys, validate="many_to_one")
            native_summary = summary[summary.controller == "native"].iloc[0]
            for metric in ["mae", "mse"]:
                result[f"{metric}_gain_pct"] = 100 * (1 - result[metric] / result[f"native_{metric}"])
                summary[f"{metric}_gain_pct"] = 100 * (1 - summary[metric] / native_summary[metric])
            local_summary = result.groupby(["controller", "lookahead"]).agg(
                mae=("mae", "mean"), mse=("mse", "mean"), native_mae=("native_mae", "mean"),
                native_mse=("native_mse", "mean"), blocks=("block", "count"),
                mae_win_fraction=("mae_gain_pct", lambda x: (x > 0).mean()),
                mse_win_fraction=("mse_gain_pct", lambda x: (x > 0).mean())).reset_index()
            for metric in ["mae", "mse"]:
                local_summary[f"{metric}_gain_pct"] = 100 * (1 - local_summary[metric] / local_summary[f"native_{metric}"])
            phase_metrics.extend(["mae_gain_pct", "mse_gain_pct"])
        result.to_csv(args.output / "block_outcomes.csv", index=False)
        summary.to_csv(args.output / "summary.csv", index=False)
        local_summary.to_csv(args.output / "local_summary.csv", index=False)
        result.groupby(["controller", "lookahead", "phase_at_decision"])[phase_metrics].mean().to_csv(args.output / "phase_summary.csv")
        plan["status"] = "complete"
        timings["postprocessing_seconds"] = perf_counter() - postprocessing_began
        print(summary.to_string(index=False), flush=True)
    except Exception as error:
        plan.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        timings["experiment_wall_seconds"] = perf_counter() - began
        save_timings(args.output, timings)
        save_json(args.output / "manifest.json", plan)
        print("Experiment timings (seconds): " + json.dumps(timings), flush=True)


if __name__ == "__main__":
    main()

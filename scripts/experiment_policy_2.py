"""Train selectors on strictly earlier outcomes, then replay a long test interval.

Standalone experiment 2: no experiment-1 output is required. Plan mode fits
selectors from existing CSV results without importing torch. Actual causal replay
runs on the server, reusing initial.pt and preserving state throughout testing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from update_policy_common import (
    DATA_FEATURES, DEFAULT_ANNOTATIONS, DEFAULT_RUNS, EXPERT_FEATURES, FEEDBACK_FEATURES,
    decision_features, file_sha256, find_job, fit_selector, historical_training_data,
    numeric_inputs, plot_forecasts, save_json, save_timings,
    oracle_decision, score_action, summarize_decisions,
)


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


def switch_policy(controlled, policy) -> None:
    """Keep cumulative accounting; reset periodic cadence only when action changes."""
    if controlled.policy == policy:
        return
    gradient_steps, egd_steps = controlled.gradient_steps, controlled.egd_steps
    controlled.set_policy(policy)
    controlled.gradient_steps, controlled.egd_steps = gradient_steps, egd_steps


def run_segment(replay, stream, values: np.ndarray, start: int, steps: int, interval: int,
                controller: str, bundle: dict, actions: dict, directory: Path,
                *, oracle_metric: str | None = None) -> tuple[pd.DataFrame, list[dict], dict]:
    from onlinetsf.update_study.reporting import Reporter
    from onlinetsf.update_study.runtime import restore_rng, rng_state, state_features
    from onlinetsf.selection import select_action

    branch = replay.fork(actions["native"])
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
            features = decision_features(state_features(stream, branch, index), values, clock)
            scores = None
            if controller.startswith("policy_"):
                action, scores = select_action(bundle["selectors"][controller], features)
            else:
                action = "native" if controller == "native" else bundle["best_fixed"]
            decision_seconds += perf_counter() - decision_began
            if oracle_metric is not None and start + steps - index >= interval:
                saved_rng = rng_state()
                search_began = perf_counter()
                local = []
                try:
                    for policy in actions.values():
                        restore_rng(saved_rng)
                        local.append(score_action(branch, stream, index, interval, policy))
                finally:
                    restore_rng(saved_rng)
                search_seconds += perf_counter() - search_began
                search_gradient_steps += sum(row["gradient_steps"] for row in local)
                search_egd_steps += sum(row["egd_steps"] for row in local)
                block = (index - start) // interval
                outcomes.extend(dict(block=block, clock=clock, block_steps=interval, **row) for row in local)
                oracle_rows.append(dict(block=block, clock=clock, **oracle_decision(local, action, oracle_metric)))
            decision_began = perf_counter()
            switch_policy(branch.method, actions[action])
            decision_seconds += perf_counter() - decision_began
            record = {"controller": controller, "decision_step": index - start + 1,
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
    parser.add_argument("--dataset", default="ETTh2")
    parser.add_argument("--method", choices=["onenet", "fsnet", "dsof"], default="onenet")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--test-seed", type=int, default=2)
    boundary = parser.add_mutually_exclusive_group()
    boundary.add_argument("--test-start", type=int, help="first test forecast origin, raw zero-based row")
    boundary.add_argument("--test-start-fraction", type=float, default=.6, help="default: floor(0.6 * raw row count)")
    parser.add_argument("--steps", type=int, help="default: every remaining prediction origin")
    parser.add_argument("--decision-interval", type=int, default=128, help="positive interval matching collected branch length")
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--min-samples-leaf", type=int, default=4,
                        help="minimum training states per leaf; a split requires at least twice this number")
    parser.add_argument("--prefix-rows", type=int, help="override only when original run manifest is absent")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--oracle-diagnostics", choices=["none", "policy_full", "policy_data", "both"], default="none",
                        help="evaluate local Oracle branches during the first automatic run; no later policy replay")
    parser.add_argument("--oracle-metric", choices=["mae", "mse"], default="mae")
    parser.add_argument("--plan-only", action="store_true", help="fit and audit the historical selector without importing torch")
    parser.add_argument("--output", type=Path, default=Path("runs/policy-exp2-temporal-onenet-etth2-h24"))
    args = parser.parse_args(argv)
    if (min(args.horizon, args.max_depth, args.min_samples_leaf, args.threads, args.decision_interval) <= 0
            or args.steps is not None and args.steps <= 0
            or args.prefix_rows is not None and args.prefix_rows <= 0):
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
    collected = None
    if args.training_data is not None:
        collected = json.loads((args.training_data / "manifest.json").read_text(encoding="utf-8"))
        if (collected["status"] != "complete" or args.train_seeds != [collected["seed"]]
                or any(collected[key] != value for key, value in
                       [("dataset", dataset), ("method", method), ("horizon", horizon),
                        ("decision_interval", interval), ("input_sha256", source_hash)])):
            raise ValueError("collected training data identity, interval or completion status differs")
        if collected["checkpoint_sha256"] != file_sha256(jobs[collected["seed"]] / "initial.pt"):
            raise ValueError("collected training data used a different checkpoint")
    prefix = args.prefix_rows or max(annotation_manifest["calibration_rows"], (annotation_manifest["rows"] + 4) // 5)
    catalogs, context_lengths, verified_prefixes = [], [], []
    state_frames, outcome_frames = [], []
    source_audit = []
    for item, source_job in jobs.items():
        catalogs.append(json.loads((source_job / "policies.json").read_text(encoding="utf-8")))
        source_manifest = source_job.parent.parent / "manifest.json"
        if source_manifest.exists():
            source_record = json.loads(source_manifest.read_text(encoding="utf-8"))
            record = next(record for record in source_record["jobs"] if record["dataset"] == dataset and
                          record["method"] == method and record["horizon"] == horizon and record["seed"] == item)
            if record["input_sha256"] != source_hash:
                raise ValueError("annotations differ from source run")
            verified_prefixes.append(record["prefix_rows"])
        audit = {"seed": item, "job": str(source_job), "manifest_verified": source_manifest.exists()}
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
    if collected is not None and collected["actions"] != catalogs[0]:
        raise ValueError("collected action definitions differ from source run")
    if len(set(context_lengths)) != 1 or (verified_prefixes and len(set(verified_prefixes)) != 1):
        raise ValueError("source jobs used inconsistent context lengths or offline prefixes")
    if verified_prefixes:
        if args.prefix_rows is not None and prefix != verified_prefixes[0]:
            raise ValueError("prefix override differs from source run")
        prefix = verified_prefixes[0]
    context_length = int(context_lengths[0])
    values, phases, _ = numeric_inputs(annotation_dir, prefix)
    clock = args.test_start if args.test_start is not None else int(len(values) * args.test_start_fraction)
    if collected is not None and (collected["test_start"] != clock or collected["prefix_rows"] != prefix
                                  or collected["context_length"] != context_length):
        raise ValueError("collection and test replay must use the same time boundary, prefix and context length")
    available_steps = len(values) - horizon - clock
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
    if len(train) < 2 * args.min_samples_leaf:
        parser.error(f"only {len(train)} historical training states remain after time isolation; "
                     f"--min-samples-leaf {args.min_samples_leaf} requires at least "
                     f"{2 * args.min_samples_leaf} to split. Reduce --min-samples-leaf "
                     "or collect more historical probes before running the forecasting replay.")
    train = pd.DataFrame([decision_features(row, values, int(row["clock"])) for row in train.to_dict("records")])
    gains = (1 - losses.div(losses.native, axis=0)) * 100
    # Load the library before measuring fits, so the first selector does not
    # absorb sklearn's one-time import cost. This belongs to preparation time.
    import sklearn.tree

    timings = {"preparation_seconds": perf_counter() - began}
    if collected is not None:
        timings.update(collected.get("timings", {}))
    selectors, rules, diagnostics = {}, [], []
    for name, features in [("policy_full", DATA_FEATURES + FEEDBACK_FEATURES + EXPERT_FEATURES[method]),
                           ("policy_data", DATA_FEATURES)]:
        fit_began = perf_counter()
        selectors[name], text = fit_selector(train, gains, names, features, args.max_depth, args.min_samples_leaf)
        timings[f"selector_{name}_seconds"] = perf_counter() - fit_began
        rules.append(name + "\n" + text)
        for action, tree in selectors[name]["trees"].items():
            diagnostics.append({"controller": name, "action": action, "training_states": len(train),
                                "nodes": len(tree["value"]), "leaves": tree["left"].count(-1)})
        split_count = sum(len(tree["value"]) > 1 for tree in selectors[name]["trees"].values())
        print(f"{name}: {len(train)} historical states; "
              f"{split_count}/{len(selectors[name]['trees'])} action trees split on features", flush=True)
    bundle = {"schema": 2, "dataset": dataset, "method": method, "horizon": horizon,
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
            "controllers": ["native", "best_fixed", "policy_full", "policy_data"],
            "input_sha256": source_hash, "source_jobs": source_audit,
            "checkpoint": str(checkpoint_path), "checkpoint_sha256": file_sha256(checkpoint_path),
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "timings": timings}
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
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import torch
    from onlinetsf.update_study.data import load_stream
    from onlinetsf.update_study.runtime import ControlledMethod, Policy, Replay, restore_rng, rng_state

    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    stream = load_stream(args.annotations / dataset, bundle["context_length"], horizon, bundle["prefix_rows"])
    if stream.manifest["feature_columns"] != annotation_manifest["feature_columns"]:
        raise ValueError("forecast feature columns changed")
    actions = {item["name"]: Policy(item["name"], item["mode"], item["period"], item["lr_scale"],
                                   None if item["groups"] == "all" else frozenset(item["groups"]))
               for item in bundle["actions"]}
    source_catalog = json.loads((job / "policies.json").read_text(encoding="utf-8"))
    if any(item not in source_catalog for item in bundle["actions"]):
        raise ValueError("checkpoint action definitions differ from training jobs")
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
    replay = Replay(ControlledMethod(forecasting_method, actions["native"]), horizon)
    first = stream.prefix_rows - stream.dataset.context_length
    start = clock - stream.dataset.context_length + 1
    outcomes, decisions, trajectory_summaries = [], [], []
    plan["status"] = "running"
    save_json(args.output / "manifest.json", plan)
    try:
        if forecasting_method.device.type == "cuda":
            torch.cuda.synchronize(forecasting_method.device)
        timings["runtime_setup_seconds"] = perf_counter() - setup_began
        warmup_began = perf_counter()
        # Native warmup stops immediately before the first test decision.
        for index in range(first, start):
            replay.step(stream.dataset, index)
            if (index - first) % 2048 == 0:
                print(f"native warmup: raw clock {index + stream.dataset.context_length - 1}", flush=True)
        if forecasting_method.device.type == "cuda":
            torch.cuda.synchronize(forecasting_method.device)
        timings["warmup_seconds"] = perf_counter() - warmup_began
        saved_rng = rng_state()
        torch.save({"replay": replay, "rng": saved_rng}, args.output / "evaluation_start.pt")
        plan["evaluation_start_sha256"] = file_sha256(args.output / "evaluation_start.pt")
        frames = {}
        for controller in plan["controllers"]:
            restore_rng(saved_rng)
            directory = args.output / controller
            frame, trajectory_decisions, summary = run_segment(
                replay, stream, values, start, steps, interval, controller, bundle, actions, directory,
                oracle_metric=args.oracle_metric if controller.startswith("policy_") and
                args.oracle_diagnostics in {controller, "both"} else None)
            frames[controller] = frame
            decisions.extend(dict(item, seed=seed) for item in trajectory_decisions)
            trajectory_summaries.append(dict(summary, controller=controller, seed=seed))
            timings[f"test_{controller}_seconds"] = summary["seconds"]
            timings[f"test_{controller}_candidate_search_seconds"] = summary["candidate_search_seconds"]
            outcomes.append(block_metrics(frame, trajectory_decisions, interval, phases).assign(controller=controller, seed=seed))
            pd.DataFrame(decisions).to_csv(args.output / "decisions.csv", index=False)
            pd.DataFrame(trajectory_summaries).to_csv(args.output / "trajectory_summary.csv", index=False)
            save_timings(args.output, timings)
            print(f"{controller}: {steps} continuous test origins complete", flush=True)
        postprocessing_began = perf_counter()
        plot_forecasts(frames, args.output / "test_errors", f"{dataset} {method} H={horizon} | start {clock}")
        result = pd.concat(outcomes, ignore_index=True)
        keys = ["seed", "block", "lookahead"]
        native = result[result.controller == "native"][keys + ["mae", "mse"]].rename(columns={"mae": "native_mae", "mse": "native_mse"})
        result = result.merge(native, on=keys, validate="many_to_one")
        for metric in ["mae", "mse"]:
            result[f"{metric}_gain_pct"] = 100 * (1 - result[metric] / result[f"native_{metric}"])
        result.to_csv(args.output / "block_outcomes.csv", index=False)
        local_summary = result.groupby(["controller", "lookahead"]).agg(
            mae=("mae", "mean"), mse=("mse", "mean"), native_mae=("native_mae", "mean"),
            native_mse=("native_mse", "mean"), blocks=("block", "count"),
            mae_win_fraction=("mae_gain_pct", lambda x: (x > 0).mean()),
            mse_win_fraction=("mse_gain_pct", lambda x: (x > 0).mean())).reset_index()
        summary = pd.DataFrame(trajectory_summaries)
        native = summary[summary.controller == "native"].iloc[0]
        for metric in ["mae", "mse"]:
            summary[f"{metric}_gain_pct"] = 100 * (1 - summary[metric] / native[metric])
            local_summary[f"{metric}_gain_pct"] = 100 * (1 - local_summary[metric] / local_summary[f"native_{metric}"])
        summary.to_csv(args.output / "summary.csv", index=False)
        local_summary.to_csv(args.output / "local_summary.csv", index=False)
        result.groupby(["controller", "lookahead", "phase_at_decision"])[[
            "mae", "mse", "mae_gain_pct", "mse_gain_pct"
        ]].mean().to_csv(args.output / "phase_summary.csv")
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

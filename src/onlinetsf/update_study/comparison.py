"""Reuse existing method results and evaluate only new frozen/Oracle baselines.

Frozen never adapts after offline training. Oracle uses future labels to choose
one action per block, so it is a hindsight reference, not a deployable policy or
an exact global optimum. Search costs and executed-update costs are separate.
With --reuse-comparison, frozen and greedy Oracle are also read from a completed
comparison with matching inputs, actions, metric and test-start state.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from ..decision.config import DecisionConfig, FEATURE_SCHEMA_VERSION
from .policy_support import (
    DEFAULT_ANNOTATIONS, DEFAULT_RUNS, file_sha256, find_job,
    load_existing_results, plot_forecasts, save_json, score_action, oracle_decision, summarize_decisions,
)


def run_comparison(replay, stream, start: int, steps: int, interval: int,
                   controller: str, actions: dict, metric: str, directory: Path) -> tuple[pd.DataFrame, dict]:
    """Execute only new frozen/Oracle baselines; existing trajectories are read from disk."""
    if controller not in {"frozen", "oracle_greedy"}:
        raise ValueError("existing trajectories must be loaded from experiment 2")
    from onlinetsf.update_study.reporting import Reporter
    from onlinetsf.update_study.runtime import restore_rng, rng_state

    trajectory = replay.fork(actions["frozen"] if controller == "frozen" else actions["native"])
    reporter = Reporter(directory, stream, start)
    decisions, outcomes = [], []
    search_seconds = decision_seconds = stream_seconds = update_seconds = 0.0
    search_gradient_steps = search_egd_steps = 0
    update_clocks = set()
    began = perf_counter()
    for block, offset in enumerate(range(0, steps, interval)):
        index = start + offset
        count = min(interval, steps - offset)
        clock = index + stream.dataset.context_length - 1
        selected = "frozen"
        if controller == "oracle_greedy":
            saved_rng = rng_state()
            search_began = perf_counter()
            local = []
            try:
                for policy in actions.values():
                    restore_rng(saved_rng)
                    local.append(score_action(trajectory, stream, index, count, policy))
            finally:
                restore_rng(saved_rng)
            search_seconds += perf_counter() - search_began
            search_gradient_steps += sum(row["gradient_steps"] for row in local)
            search_egd_steps += sum(row["egd_steps"] for row in local)
            decision_began = perf_counter()
            selected = min(local, key=lambda row: row[metric])["action"]
            decision_seconds += perf_counter() - decision_began
            outcomes.extend(dict(block=block, clock=clock, block_steps=count,
                                 full_block=count == interval, **row) for row in local)
        decision_began = perf_counter()
        trajectory.method.switch_policy(actions[selected])
        decision_seconds += perf_counter() - decision_began
        decisions.append({"block": block, "clock": clock, "block_steps": count, "selected_action": selected,
                          "steps_since_gradient": trajectory.method.gradient_age(index)})
        for current in range(index, index + count):
            previous_steps = trajectory.method.gradient_steps
            stream_began = perf_counter()
            records = trajectory.step(stream.dataset, current)
            stream_seconds += perf_counter() - stream_began
            if trajectory.method.gradient_steps > previous_steps:
                update_clocks.add(current)
            update_seconds += sum(row["update"]["seconds"] for row in records if row["update"] is not None)
            reporter.add(records)
    stream_began = perf_counter()
    reporter.add(trajectory.finish(stream.dataset))
    stream_seconds += perf_counter() - stream_began
    wall_seconds = perf_counter() - began
    summary = reporter.close(trajectory.method, wall_seconds)
    summary.pop("seconds")
    summary.update(controller=controller, gradient_update_clocks=len(update_clocks),
                   stream_seconds=stream_seconds, update_seconds=update_seconds,
                   decision_seconds=decision_seconds, candidate_search_seconds=search_seconds,
                   candidate_gradient_steps=search_gradient_steps, candidate_egd_steps=search_egd_steps,
                   online_seconds=wall_seconds, wall_seconds=wall_seconds,
                   candidate_search_role="decision" if controller == "oracle_greedy" else "none",
                   result_source="comparison_measured", executed_in_comparison=True)
    pd.DataFrame(decisions).to_csv(directory / "decisions.csv", index=False)
    if outcomes:
        pd.DataFrame(outcomes).to_csv(directory / "action_outcomes.csv", index=False)
    forecasts = pd.read_csv(directory / "forecasts.csv")
    if len(forecasts) != steps or not np.array_equal(forecasts.forecast_index, np.arange(start, start + steps)):
        raise RuntimeError("trajectory did not score every forecast exactly once")
    for row in pd.DataFrame(outcomes).itertuples():
        if row.action != decisions[row.block]["selected_action"]:
            continue
        actual = forecasts.iloc[row.block * interval:row.block * interval + row.block_steps][metric].mean()
        if not np.isclose(actual, getattr(row, metric), rtol=1e-7, atol=1e-9):
            raise RuntimeError(f"selected Oracle branch differs at block {row.block}")
    save_json(directory / "summary.json", summary)
    return forecasts, summary


def main(argv: list[str] | None = None) -> None:
    began = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True, help="completed experiment-2 directory")
    parser.add_argument("--reuse-comparison", type=Path,
                        help="completed stage-4 directory; reuse frozen and greedy Oracle without model execution")
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--controller", choices=["policy_full", "policy_data"], default="policy_full",
                        help="read saved local Oracle diagnostics for this automatic controller; never replay it")
    parser.add_argument("--metric", choices=["mae", "mse"], default="mae")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--plan-only", action="store_true", help="validate inputs without importing PyTorch")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.threads <= 0:
        parser.error("threads must be positive")
    manifest = json.loads((args.experiment / "manifest.json").read_text(encoding="utf-8"))
    bundle_path = args.experiment / "selectors.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if bundle.get("schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("unsupported selector artifact; refit with the current feature schema")
    if manifest["status"] != "complete" or file_sha256(bundle_path) != manifest["selector_sha256"]:
        raise ValueError("source experiment must be complete with its original saved selectors")
    for key in ["dataset", "method", "horizon", "test_seed", "test_start", "prefix_rows", "decision_interval", "input_sha256"]:
        if bundle[key] != manifest[key]:
            raise ValueError(f"selector and source experiment disagree on {key}")
    dataset, method, horizon, seed = bundle["dataset"], bundle["method"], bundle["horizon"], bundle["test_seed"]
    interval, steps = bundle["decision_interval"], manifest["steps"]
    if interval <= 0 or steps <= 0 or bundle["train_target_end_exclusive"] > bundle["test_start"]:
        raise ValueError("source must have a positive interval and strictly historical selector training")
    selector = bundle["selectors"][args.controller]
    names = [item["name"] for item in bundle["actions"]]
    if selector["actions"] != names or len(names) != len(set(names)) or not {"native", "frozen"}.issubset(names):
        raise ValueError("selector must use the complete, unique catalog including native and frozen")
    job = find_job(args.runs, dataset, method, horizon, seed)
    checkpoint_path = job / "initial.pt"
    if file_sha256(checkpoint_path) != manifest["checkpoint_sha256"]:
        raise ValueError("initial checkpoint differs from source experiment")
    annotation_dir = args.annotations / dataset
    if file_sha256(annotation_dir / "labeled_data.csv") != bundle["input_sha256"]:
        raise ValueError("annotated input differs from source experiment")
    annotation = json.loads((annotation_dir / "manifest.json").read_text(encoding="utf-8"))
    clock = bundle["test_start"]
    if not bundle["prefix_rows"] <= clock or clock + steps + horizon > annotation["rows"]:
        raise ValueError("source test interval lies outside available prediction targets")
    reference_decisions = pd.read_csv(args.experiment / "decisions.csv")
    reference_decisions = reference_decisions[reference_decisions.controller == args.controller].sort_values("clock")
    expected_clocks = np.arange(clock, clock + steps, interval)
    if not np.array_equal(reference_decisions.clock, expected_clocks):
        raise ValueError("source decision records do not cover the test trajectory")
    reference_forecasts = pd.read_csv(args.experiment / args.controller / "forecasts.csv")
    start = clock - bundle["context_length"] + 1
    if len(reference_forecasts) != steps or not np.array_equal(reference_forecasts.forecast_index, np.arange(start, start + steps)):
        raise ValueError("source forecasts do not cover the test trajectory")
    plan = {"status": "planned", "experiment": str(args.experiment), "controller": args.controller,
            "dataset": dataset, "method": method, "horizon": horizon, "seed": seed,
            "metric": args.metric, "test_start": clock, "test_steps": steps,
            "decision_interval": interval, "local_oracle_decisions": steps // interval,
            "local_oracle_excluded_tail_steps": steps % interval, "actions": names,
            "local_oracle_action_comparisons": 0,
            "greedy_oracle_action_comparisons": ((steps + interval - 1) // interval) * len(names),
            "branch_prediction_origins": steps * len(names),
            "selector_sha256": file_sha256(bundle_path), "checkpoint_sha256": file_sha256(checkpoint_path),
            "input_sha256": bundle["input_sha256"], "device": args.device, "threads": args.threads,
            "acc_tie_rtol": 1e-7, "acc_tie_atol": 1e-10,
            "relative_regret_definition": "mean((selected_loss - optimal_loss) / optimal_loss) over full blocks",
            "oracle_definition": "greedy hindsight action selection on its own trajectory; not an exact global optimum",
            "local_oracle_definition": "statistics from saved experiment-2 candidate outcomes only",
            "frozen_definition": "no adaptation after offline training, including the pre-test warmup",
            "controllers": ["frozen", "native", "best_fixed", "policy_full", "policy_data", "oracle_greedy"],
            "timing_definition": "existing trajectory costs read from experiment 2; only frozen and Oracle are executed",
            "search_definition": "oracle_greedy search is included in its online_seconds; candidate costs separate"}
    if args.reuse_comparison is not None:
        previous = json.loads((args.reuse_comparison / "manifest.json").read_text(encoding="utf-8"))
        if previous["status"] != "complete":
            raise ValueError("reused comparison must be complete")
        for key in ("dataset", "method", "horizon", "seed", "metric", "test_start", "test_steps",
                    "decision_interval", "checkpoint_sha256", "input_sha256", "actions"):
            if previous[key] != plan[key]:
                raise ValueError(f"reused comparison differs on {key}")
        previous_experiment = Path(manifest["reused_experiment"] if "reused_experiment" in manifest else previous["experiment"])
        previous_bundle_path = previous_experiment / "selectors.json"
        previous_bundle = json.loads(previous_bundle_path.read_text(encoding="utf-8"))
        previous_manifest = json.loads((previous_experiment / "manifest.json").read_text(encoding="utf-8"))
        if (previous_manifest["status"] != "complete"
                or file_sha256(previous_bundle_path) != previous["selector_sha256"]
                or previous_manifest["evaluation_start_sha256"] != manifest["evaluation_start_sha256"]
                or file_sha256(args.experiment / "evaluation_start.pt") != manifest["evaluation_start_sha256"]):
            raise ValueError("reused comparison has a different selector source or test-start state")
        for key in ("actions", "context_length", "prefix_rows", "update_groups"):
            if previous_bundle[key] != bundle[key]:
                raise ValueError(f"reused comparison differs on {key}")
        previous_config = DecisionConfig.from_dict(previous_bundle["decision_config"]).to_dict()
        current_config = DecisionConfig.from_dict(bundle["decision_config"]).to_dict()
        for key, value in current_config.items():
            if key not in {"data_max_depth", "data_min_samples_leaf", "full_max_depth", "full_min_samples_leaf"}:
                if previous_config[key] != value:
                    raise ValueError(f"reused comparison differs on decision.{key}")
        plan.update(reused_comparison=str(args.reuse_comparison.resolve()),
                    reused_controllers=["frozen", "oracle_greedy"],
                    timing_definition="all trajectory costs retain their original measurements; no models are executed",
                    oracle_initialization_source="comparison_saved", oracle_initialization_seconds=0.0)
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "manifest.json", plan)
    print(json.dumps(plan, indent=2), flush=True)
    if args.plan_only:
        return

    plan["status"] = "running"
    save_json(args.output / "manifest.json", plan)
    try:
        frames, summaries = {}, []
        statistics_began = perf_counter()
        for controller in ["native", "best_fixed", "policy_full", "policy_data"]:
            forecasts, summary = load_existing_results(args.experiment, controller, start, steps)
            frames[controller] = forecasts
            summaries.append(summary)
            print(f"{controller}: reused saved forecasts, updates and costs; no model replay", flush=True)
        plan["reuse_statistics_seconds"] = perf_counter() - statistics_began
        outcomes_path = args.experiment / args.controller / "action_outcomes.csv"
        plan["local_oracle_available"] = outcomes_path.exists()
        if outcomes_path.exists():
            saved_outcomes = pd.read_csv(outcomes_path)
            local_rows = []
            for block, group in saved_outcomes.groupby("block", sort=True):
                selected = reference_decisions.iloc[int(block)].selected_action
                local_rows.append(dict(block=int(block), clock=int(group.clock.iloc[0]),
                                       **oracle_decision(group.to_dict("records"), selected, args.metric)))
            local = pd.DataFrame(local_rows)
            local_summary = (summarize_decisions(local, args.metric) if not local.empty else
                             {"metric": args.metric, "decisions": 0, "acc": None, "mean_relative_regret": None})
            local.to_csv(args.output / "oracle_decisions.csv", index=False)
            save_json(args.output / "oracle_summary.json", local_summary)
        if args.reuse_comparison is not None:
            for controller in ("frozen", "oracle_greedy"):
                directory = args.reuse_comparison / controller
                forecasts = pd.read_csv(directory / "forecasts.csv")
                summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
                if (len(forecasts) != steps
                        or not np.array_equal(forecasts.forecast_index, np.arange(start, start + steps))):
                    raise ValueError(f"{controller}: reused forecasts do not cover the test interval")
                if summary["controller"] != controller or any(
                    not np.isclose(summary[loss], forecasts[loss].mean(), rtol=1e-7, atol=1e-9)
                    for loss in ("mae", "mse")
                ):
                    raise ValueError(f"{controller}: reused summary and forecasts disagree")
                if controller == "frozen" and (summary["gradient_steps"] or summary["egd_steps"]):
                    raise ValueError("reused frozen baseline performed an update")
                summary.update(result_source="comparison_saved", executed_in_comparison=False)
                shutil.copytree(directory, args.output / controller)
                save_json(args.output / controller / "summary.json", summary)
                frames[controller] = forecasts
                summaries.append(summary)
                print(f"{controller}: reused saved baseline; no model execution", flush=True)
        else:
            # Forecasting imports are deferred; reuse needs no forecasting runtime.
            import torch
            from onlinetsf.update_study.data import load_stream
            from onlinetsf.update_study.runtime import ControlledMethod, Policy, Replay, load_replay, move_method, restore_rng

            torch.set_num_threads(args.threads)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            stream = load_stream(annotation_dir, bundle["context_length"], horizon, bundle["prefix_rows"])
            actions = {item["name"]: Policy.from_dict(item) for item in bundle["actions"]}
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if checkpoint["prefix_rows"] != bundle["prefix_rows"]:
                raise ValueError("checkpoint prefix differs from saved selector")
            forecasting_method = checkpoint["method"]
            initial_rng = (checkpoint["rng"][0].cpu(),
                           [item.cpu() for item in checkpoint["rng"][1]] if checkpoint["rng"][1] is not None else None)
            if initial_rng[1] is not None and torch.cuda.is_available() and len(initial_rng[1]) != torch.cuda.device_count():
                raise ValueError("match the source visible GPU count, normally CUDA_VISIBLE_DEVICES=1")
            move_method(forecasting_method, args.device)
            initial = Replay(ControlledMethod(forecasting_method, actions["native"]), horizon,
                             DecisionConfig.from_dict(bundle["decision_config"]))
            restore_rng(initial_rng)
            forecasts, summary = run_comparison(initial, stream, start, steps, interval, "frozen", actions,
                                                 args.metric, args.output / "frozen")
            if summary["gradient_steps"] or summary["egd_steps"]:
                raise RuntimeError("frozen baseline performed an update")
            frames["frozen"] = forecasts
            summaries.append(summary)
            initialization_began = perf_counter()
            cache_path = args.experiment / "evaluation_start.pt"
            if file_sha256(cache_path) != manifest["evaluation_start_sha256"]:
                raise ValueError("saved test-start state differs from experiment 2")
            oracle_start, saved_rng = load_replay(cache_path, args.device)
            if oracle_start.last_index != start - 1 or oracle_start.horizon != horizon:
                raise ValueError("saved state does not immediately precede testing")
            restore_rng(saved_rng)
            plan["oracle_initialization_source"] = "experiment2_saved_state"
            plan["oracle_initialization_seconds"] = perf_counter() - initialization_began
            forecasts, summary = run_comparison(oracle_start, stream, start, steps, interval, "oracle_greedy", actions,
                                                 args.metric, args.output / "oracle_greedy")
            frames["oracle_greedy"] = forecasts
            summaries.append(summary)
        comparison = pd.DataFrame(summaries)
        native = comparison[comparison.controller == "native"].iloc[0]
        frozen = comparison[comparison.controller == "frozen"].iloc[0]
        for loss in ["mae", "mse"]:
            comparison[f"{loss}_gain_vs_native_pct"] = 100 * (1 - comparison[loss] / native[loss]) if native[loss] else np.nan
            comparison[f"{loss}_gain_vs_frozen_pct"] = 100 * (1 - comparison[loss] / frozen[loss]) if frozen[loss] else np.nan
        comparison["gradient_steps_saved_vs_native_pct"] = (100 * (1 - comparison.gradient_steps / native.gradient_steps)
                                                               if native.gradient_steps else np.nan)
        comparison["online_time_saved_vs_native_pct"] = 100 * (1 - comparison.online_seconds / native.online_seconds)
        comparison.to_csv(args.output / "comparison.csv", index=False)
        plot_forecasts(frames, args.output / "comparison_errors", f"{dataset} {method} H={horizon}")
        save_json(args.output / "preparation_costs.json", {"source_experiment_timings": manifest["timings"],
                  "oracle_initialization_seconds": plan["oracle_initialization_seconds"],
                  "oracle_initialization_source": plan["oracle_initialization_source"]})
        plan.update(status="complete", existing_trajectories_reused=True, selected_oracle_branches_match_trajectory=True)
        print(comparison.to_string(index=False), flush=True)
    except Exception as error:
        plan.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        plan["wall_seconds"] = perf_counter() - began
        save_json(args.output / "manifest.json", plan)


if __name__ == "__main__":
    main()

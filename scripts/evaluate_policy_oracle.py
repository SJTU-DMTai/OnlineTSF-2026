"""Evaluate local action optimality along an existing experiment-2 policy trajectory.

At each complete decision block, all actions start from the same pre-feedback
state and RNG. Future outcomes score the actions but never choose the action on
the main trajectory. Unchanged actions retain their periodic-update cadence.
The final partial block is replayed, but excluded from oracle metrics.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from experiment_policy_2 import switch_policy
from update_policy_common import (
    DEFAULT_ANNOTATIONS, DEFAULT_RUNS, decision_features, file_sha256,
    find_job, numeric_inputs, save_json, select_actions,
)


def oracle_decision(outcomes: list[dict], selected: str, metric: str) -> dict:
    """Score a choice against the best fixed action over this local window.

    ACC accepts all numerically tied optima (rtol=1e-7, atol=1e-10).
    Relative regret is (selected - optimum) / optimum, not a percentage.
    """
    losses = np.array([row[metric] for row in outcomes], dtype=float)
    if not np.isfinite(losses).all() or (losses < 0).any():
        raise ValueError("oracle losses must be finite and nonnegative")
    best = float(losses.min())
    if best == 0:
        raise ValueError("relative regret is undefined for a zero optimal loss")
    names = [row["action"] for row in outcomes]
    chosen = float(losses[names.index(selected)])
    optimal = np.isclose(losses, best, rtol=1e-7, atol=1e-10)
    relative = (chosen - best) / best
    return {"metric": metric, "selected_action": selected,
            "oracle_action": names[int(losses.argmin())],
            "optimal_actions": json.dumps([name for name, match in zip(names, optimal) if match]),
            "selected_loss": chosen, "oracle_loss": best,
            "native_loss_same_state": float(losses[names.index("native")]),
            "correct": bool(optimal[names.index(selected)]),
            "relative_regret": relative, "relative_regret_pct": relative * 100,
            "within_1pct": bool(chosen <= best * 1.01 + 1e-12)}


def summarize_decisions(decisions: pd.DataFrame, metric: str) -> dict:
    """Each full decision interval has equal weight in these metrics."""
    return {"metric": metric, "decisions": len(decisions),
            "correct_decisions": int(decisions.correct.sum()), "acc": float(decisions.correct.mean()),
            "acc_pct": float(decisions.correct.mean() * 100),
            "mean_relative_regret": float(decisions.relative_regret.mean()),
            "mean_relative_regret_pct": float(decisions.relative_regret.mean() * 100),
            "median_relative_regret_pct": float(decisions.relative_regret.median() * 100),
            "within_1pct_decisions": int(decisions.within_1pct.sum()),
            "within_1pct_fraction": float(decisions.within_1pct.mean()),
            "within_1pct_pct": float(decisions.within_1pct.mean() * 100),
            "mean_selected_loss": float(decisions.selected_loss.mean()),
            "mean_oracle_loss_same_states": float(decisions.oracle_loss.mean()),
            "mean_native_loss_same_states": float(decisions.native_loss_same_state.mean())}


def score_action(replay, stream, index: int, steps: int, policy) -> dict:
    """Preserve unchanged-policy cadence and score only newly emitted forecasts."""
    began = perf_counter()
    branch = copy.deepcopy(replay)
    before_gradient, before_egd = branch.method.gradient_steps, branch.method.egd_steps
    switch_policy(branch.method, policy)
    count, absolute, squared = 0, 0., 0.
    for current in range(index, index + steps):
        for record in branch.step(stream.dataset, current):
            if record["index"] >= index:
                count += 1
                absolute += record["mae"]
                squared += record["mse"]
    # Mature terminal predictions without applying future gradient updates.
    for record in branch.finish(stream.dataset):
        if record["index"] >= index:
            count += 1
            absolute += record["mae"]
            squared += record["mse"]
    if count != steps:
        raise RuntimeError(f"{policy.name}: scored {count} forecasts, expected {steps}")
    return {"action": policy.name, "mae": absolute / steps, "mse": squared / steps,
            "forecasts": count, "gradient_steps": branch.method.gradient_steps - before_gradient,
            "egd_steps": branch.method.egd_steps - before_egd, "seconds": perf_counter() - began}


def run_oracle_replay(replay, stream, values: np.ndarray, start: int, steps: int, interval: int,
                      controller: str, bundle: dict, actions: dict, metric: str,
                      reference_actions: list[str], directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    from onlinetsf.update_study.reporting import Reporter
    from onlinetsf.update_study.runtime import restore_rng, rng_state, state_features

    trajectory = replay.fork(actions["native"])
    reporter = Reporter(directory / "trajectory", stream, start)
    selector = bundle["selectors"][controller]
    decisions, outcomes, oracle_rows = [], [], []
    began = perf_counter()
    for block, offset in enumerate(range(0, steps, interval)):
        index = start + offset
        clock = index + stream.dataset.context_length - 1
        features = decision_features(state_features(stream, trajectory, index), values, clock)
        chosen, scores = select_actions(selector, pd.DataFrame([features]))
        selected = chosen[0]
        if selected != reference_actions[block]:
            raise RuntimeError(f"decision at raw row {clock} differs from the original experiment: "
                               f"{selected} != {reference_actions[block]}")
        record = {"controller": controller, "seed": bundle["test_seed"], "block": block,
                  "decision_step": offset + 1, "clock": clock,
                  "block_steps": min(interval, steps - offset), "selected_action": selected,
                  "phase_at_decision": stream.phases[clock],
                  "predicted_gains": json.dumps(dict(zip(selector["actions"], scores[0].tolist())))}
        record.update({name: features.get(name) for name in selector["features"]})
        decisions.append(record)
        if steps - offset >= interval:
            saved_rng = rng_state()
            local = []
            try:
                for policy in actions.values():
                    restore_rng(saved_rng)
                    local.append(score_action(trajectory, stream, index, interval, policy))
            finally:
                # Oracle branches must not consume the main trajectory's RNG.
                restore_rng(saved_rng)
            outcomes.extend(dict(controller=controller, seed=bundle["test_seed"], block=block,
                                 decision_step=offset + 1, clock=clock,
                                 target_end_exclusive=clock + interval + bundle["horizon"], **row) for row in local)
            scored = oracle_decision(local, selected, metric)
            oracle_rows.append(dict(controller=controller, seed=bundle["test_seed"], block=block,
                                    decision_step=offset + 1, clock=clock,
                                    phase_at_decision=stream.phases[clock], **scored))
            pd.DataFrame(outcomes).to_csv(directory / "action_outcomes.csv", index=False)
            frame = pd.DataFrame(oracle_rows)
            frame.to_csv(directory / "oracle_decisions.csv", index=False)
            pd.DataFrame([summarize_decisions(frame, metric)]).to_csv(directory / "summary.csv", index=False)
            print(f"oracle {block + 1}/{steps // interval}: raw row {clock}, selected={selected}, "
                  f"best={scored['oracle_action']}, regret={scored['relative_regret_pct']:.4f}%", flush=True)
        # Apply only the learned selector's action. Never use oracle scores here.
        switch_policy(trajectory.method, actions[selected])
        for current in range(index, index + record["block_steps"]):
            reporter.add(trajectory.step(stream.dataset, current))
        pd.DataFrame(decisions).to_csv(directory / "decisions.csv", index=False)
    reporter.add(trajectory.finish(stream.dataset))
    trajectory_summary = reporter.close(trajectory.method, perf_counter() - began)
    trajectory_summary["seconds_include_oracle_branches"] = True
    save_json(directory / "trajectory_summary.json", trajectory_summary)
    forecasts = pd.read_csv(directory / "trajectory/forecasts.csv")
    if len(forecasts) != steps or not np.array_equal(forecasts.forecast_index, np.arange(start, start + steps)):
        raise RuntimeError("main trajectory did not score all newly emitted forecasts exactly once")
    frame = pd.DataFrame(oracle_rows)
    # A selected action's fork must reproduce its actual trajectory block.
    for row in frame.itertuples():
        actual = forecasts.iloc[row.block * interval:(row.block + 1) * interval][metric].mean()
        if not np.isclose(actual, row.selected_loss, rtol=1e-7, atol=1e-9):
            raise RuntimeError(f"selected-action branch differs from main trajectory at block {row.block}")
    return forecasts, frame


def main(argv: list[str] | None = None) -> None:
    began = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True, help="completed experiment-2 directory")
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--controller", choices=["policy_full", "policy_data"], default="policy_full")
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
    if manifest["status"] != "complete" or file_sha256(bundle_path) != manifest["selector_sha256"]:
        raise ValueError("source experiment must be complete with its original saved selectors")
    for key in ["dataset", "method", "horizon", "test_seed", "test_start", "prefix_rows", "decision_interval", "input_sha256"]:
        if bundle[key] != manifest[key]:
            raise ValueError(f"selector and source experiment disagree on {key}")
    dataset, method, horizon, seed = bundle["dataset"], bundle["method"], bundle["horizon"], bundle["test_seed"]
    interval, steps = bundle["decision_interval"], manifest["steps"]
    if interval <= 0 or steps < interval or bundle["train_target_end_exclusive"] > bundle["test_start"]:
        raise ValueError("source must have full test intervals and strictly historical selector training")
    selector = bundle["selectors"][args.controller]
    names = [item["name"] for item in bundle["actions"]]
    if selector["actions"] != names or len(names) != len(set(names)) or "native" not in names:
        raise ValueError("selector must use the complete, unique saved action catalog")
    job = find_job(args.runs, dataset, method, horizon, seed)
    checkpoint_path = job / "initial.pt"
    if file_sha256(checkpoint_path) != manifest["checkpoint_sha256"]:
        raise ValueError("initial checkpoint differs from source experiment")
    source_actions = json.loads((job / "policies.json").read_text(encoding="utf-8"))
    if {item["name"]: item for item in source_actions} != {item["name"]: item for item in bundle["actions"]}:
        raise ValueError("source action definitions differ from saved selector")
    annotation_dir = args.annotations / dataset
    if file_sha256(annotation_dir / "labeled_data.csv") != bundle["input_sha256"]:
        raise ValueError("annotated input differs from source experiment")
    values, _, _ = numeric_inputs(annotation_dir, bundle["prefix_rows"])
    clock = bundle["test_start"]
    if not bundle["prefix_rows"] <= clock or clock + steps + horizon > len(values):
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
            "decision_interval": interval, "evaluated_decisions": steps // interval,
            "excluded_partial_steps": steps % interval, "actions": names,
            "action_comparisons": (steps // interval) * len(names),
            "branch_prediction_origins": (steps // interval) * len(names) * interval,
            "selector_sha256": file_sha256(bundle_path), "checkpoint_sha256": file_sha256(checkpoint_path),
            "input_sha256": bundle["input_sha256"], "device": args.device, "threads": args.threads,
            "acc_tie_rtol": 1e-7, "acc_tie_atol": 1e-10,
            "relative_regret_definition": "mean((selected_loss - optimal_loss) / optimal_loss) over full blocks",
            "oracle_definition": "best fixed action for one block, conditional on the learned policy's own state"}
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "manifest.json", plan)
    print(json.dumps(plan, indent=2), flush=True)
    if args.plan_only:
        return

    plan["status"] = "running"
    save_json(args.output / "manifest.json", plan)
    try:
        # Forecasting imports are deliberately deferred: local plan/tests need no torch.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        import torch
        from onlinetsf.update_study.data import load_stream
        from onlinetsf.update_study.runtime import ControlledMethod, Policy, Replay, restore_rng

        torch.set_num_threads(args.threads)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        stream = load_stream(annotation_dir, bundle["context_length"], horizon, bundle["prefix_rows"])
        actions = {item["name"]: Policy(item["name"], item["mode"], item["period"], item["lr_scale"],
                                       None if item["groups"] == "all" else frozenset(item["groups"]))
                   for item in bundle["actions"]}
        checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
        if checkpoint["prefix_rows"] != bundle["prefix_rows"]:
            raise ValueError("checkpoint prefix differs from saved selector")
        forecasting_method = checkpoint["method"]
        forecasting_method.device = torch.device(args.device)
        forecasting_method.model.to(forecasting_method.device)
        initial_rng = (checkpoint["rng"][0].cpu(),
                       [item.cpu() for item in checkpoint["rng"][1]] if checkpoint["rng"][1] is not None else None)
        if initial_rng[1] is not None and torch.cuda.is_available() and len(initial_rng[1]) != torch.cuda.device_count():
            raise ValueError("match the source visible GPU count, normally CUDA_VISIBLE_DEVICES=1")
        restore_rng(initial_rng)
        replay = Replay(ControlledMethod(forecasting_method, actions["native"]), horizon)
        warmup_began = perf_counter()
        first = stream.prefix_rows - stream.dataset.context_length
        for index in range(first, start):
            replay.step(stream.dataset, index)
            if (index - first) % 2048 == 0:
                print(f"native warmup: raw clock {index + stream.dataset.context_length - 1}", flush=True)
        plan["warmup_seconds"] = perf_counter() - warmup_began
        forecasts, decisions = run_oracle_replay(
            replay, stream, values, start, steps, interval, args.controller, bundle, actions,
            args.metric, reference_decisions.selected_action.tolist(), args.output)
        if len(decisions) != steps // interval:
            raise RuntimeError("oracle did not evaluate every full decision block")
        # Confirm that adding hindsight branches preserved the original trajectory.
        for column in ["mae", "mse", "time_expert_mae", "variable_expert_mae"]:
            if not np.allclose(forecasts[column], reference_forecasts[column], rtol=1e-6, atol=1e-8, equal_nan=True):
                raise RuntimeError(f"replayed {column} differs from the original experiment")
        summary = summarize_decisions(decisions, args.metric)
        summary.update(controller=args.controller, seed=seed)
        pd.DataFrame([summary]).to_csv(args.output / "summary.csv", index=False)
        save_json(args.output / "summary.json", summary)
        plan.update(status="complete", trajectory_matches_source=True, selected_branches_match_trajectory=True)
        print(pd.DataFrame([summary]).to_string(index=False), flush=True)
    except Exception as error:
        plan.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        plan["wall_seconds"] = perf_counter() - began
        save_json(args.output / "manifest.json", plan)


if __name__ == "__main__":
    main()

"""Experiment 2: run saved selectors continuously on a few short replay segments.

Run this script on the training server. It reuses initial.pt, performs native
warmup once, and forks short trajectories; it does not retrain the offline model.
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
    DEFAULT_ANNOTATIONS, DEFAULT_RUNS, decision_features, file_sha256, find_job, numeric_inputs,
    plot_forecasts, save_json, select_actions,
)


def choose_segments(clocks: list[int], steps: int, horizon: int, last_clock: int, count: int) -> list[int]:
    """Spread sparse, non-overlapping segments across eligible test positions."""
    candidates = [clock for clock in sorted(set(clocks)) if clock + steps <= last_clock + 1]
    selected = []
    # Greedy spacing makes target intervals, as well as origin intervals, disjoint.
    for clock in candidates:
        if not selected or clock >= selected[-1] + steps + horizon - 1:
            selected.append(clock)
    if not selected:
        raise ValueError("no complete segment fits; use fewer steps or explicit earlier start-clocks")
    if len(selected) > count:
        selected = [selected[index] for index in np.linspace(0, len(selected) - 1, count, dtype=int)]
    return selected


def switch_policy(controlled, policy) -> None:
    """Keep cumulative accounting; reset periodic cadence only when action changes."""
    if controlled.policy == policy:
        return
    gradient_steps, egd_steps = controlled.gradient_steps, controlled.egd_steps
    controlled.set_policy(policy)
    controlled.gradient_steps, controlled.egd_steps = gradient_steps, egd_steps


def run_segment(replay, stream, values: np.ndarray, start: int, steps: int, interval: int,
                controller: str, bundle: dict, actions: dict, directory: Path) -> tuple[pd.DataFrame, list[dict], dict]:
    from onlinetsf.update_study.reporting import Reporter
    from onlinetsf.update_study.runtime import state_features

    branch = replay.fork(actions["native"])
    reporter = Reporter(directory, stream, start)
    decisions = []
    if branch.method.method.device.type == "cuda":
        import torch

        torch.cuda.synchronize(branch.method.method.device)
    began = perf_counter()
    for index in range(start, start + steps):
        if (index - start) % interval == 0:
            clock = index + stream.dataset.context_length - 1
            features = decision_features(state_features(stream, branch, index), values, clock)
            scores = None
            if controller.startswith("policy_"):
                selected, predicted = select_actions(bundle["selectors"][controller], pd.DataFrame([features]))
                action = selected[0]
                scores = dict(zip(bundle["selectors"][controller]["actions"], predicted[0].tolist()))
            else:
                action = "native" if controller == "native" else bundle["best_fixed"]
            switch_policy(branch.method, actions[action])
            record = {"controller": controller, "decision_step": index - start + 1,
                      "clock": clock, "selected_action": action,
                      "phase_at_decision": stream.phases[clock]}
            for feature in bundle["selectors"]["policy_full"]["features"]:
                record[feature] = features.get(feature)
            if scores is not None:
                record["predicted_gains"] = json.dumps(scores, allow_nan=False)
            decisions.append(record)
        reporter.add(branch.step(stream.dataset, index))
    # Score all emitted forecasts. Terminal feedback never trains or emits forecasts.
    reporter.add(branch.finish(stream.dataset))
    summary = reporter.close(branch.method, perf_counter() - began)
    forecasts = pd.read_csv(directory / "forecasts.csv")
    if len(forecasts) != steps or not np.array_equal(forecasts.forecast_index, np.arange(start, start + steps)):
        raise RuntimeError("segment did not score each newly emitted forecast exactly once")
    return forecasts, decisions, summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment1", type=Path, default=Path("runs/policy-exp1-onenet-etth2-h24"))
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--segments", type=int, default=3)
    parser.add_argument("--start-clocks", type=int, nargs="+", help="raw, zero-based decision clocks")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--plan-only", action="store_true", help="validate segments without importing torch")
    parser.add_argument("--output", type=Path, default=Path("runs/policy-exp2-onenet-etth2-h24"))
    args = parser.parse_args(argv)
    bundle = json.loads((args.experiment1 / "selectors.json").read_text(encoding="utf-8"))
    if bundle["schema"] != 1 or min(args.steps, args.segments, args.threads) <= 0:
        parser.error("invalid selector schema or non-positive size")
    interval = bundle["lookahead"]
    if args.steps % interval:
        parser.error("steps must be a multiple of the selector training lookahead")
    dataset, method, horizon, seed = bundle["dataset"], bundle["method"], bundle["horizon"], bundle["test_seed"]
    job = find_job(args.runs, dataset, method, horizon, seed)
    checkpoint_path = job / "initial.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    values, _, annotation_manifest = numeric_inputs(args.annotations / dataset, bundle["prefix_rows"])
    if file_sha256(args.annotations / dataset / "labeled_data.csv") != bundle["input_sha256"]:
        raise ValueError("annotation inputs differ from experiment 1")
    last_clock = len(values) - horizon - 1
    if args.start_clocks is None:
        clocks = choose_segments(bundle["test_clocks"], args.steps, horizon, last_clock, args.segments)
    else:
        clocks = sorted(args.start_clocks)
        if len(clocks) != len(set(clocks)) or any(
            right < left + args.steps + horizon - 1 for left, right in zip(clocks, clocks[1:])
        ):
            parser.error("explicit segments must be unique and have disjoint target ranges")
    for clock in clocks:
        if clock < bundle["prefix_rows"] - 1 or clock + args.steps > last_clock + 1:
            parser.error("segment is outside the available online prediction origins")
        if bundle["split"] == "seed-time" and clock < bundle["train_target_end_exclusive"]:
            parser.error("segment overlaps training targets")
    plan = {"status": "planned", "dataset": dataset, "method": method, "horizon": horizon,
            "test_seed": seed, "split": bundle["split"], "start_clocks": clocks,
            "steps": args.steps, "decision_interval": interval, "action_count": len(bundle["actions"]),
            "controllers": ["native", "best_fixed", "policy_full", "policy_data"],
            "input_sha256": bundle["input_sha256"], "selector_sha256": file_sha256(args.experiment1 / "selectors.json"),
            "checkpoint": str(checkpoint_path), "checkpoint_sha256": file_sha256(checkpoint_path),
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "manifest.json", plan)
    print(json.dumps(plan, indent=2), flush=True)
    if args.plan_only:
        return

    # The forecasting runtime is only loaded on this execution path, on the server.
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
    starts = {clock - stream.dataset.context_length + 1: clock for clock in clocks}
    outcomes, decisions, trajectory_summaries = [], [], []
    plan["status"] = "running"
    save_json(args.output / "manifest.json", plan)
    try:
        for index in range(first, max(starts) + 1):
            if index in starts:
                clock = starts[index]
                saved_rng = rng_state()
                frames = {}
                for controller in plan["controllers"]:
                    restore_rng(saved_rng)
                    directory = args.output / f"start_{clock}" / controller
                    frame, segment_decisions, summary = run_segment(
                        replay, stream, values, index, args.steps, interval, controller, bundle, actions, directory)
                    frames[controller] = frame
                    decisions.extend(dict(item, segment_clock=clock, seed=seed) for item in segment_decisions)
                    trajectory_summaries.append(dict(summary, controller=controller, segment_clock=clock, seed=seed))
                    for block in range(args.steps // interval):
                        selected = frame.iloc[block * interval:(block + 1) * interval]
                        for lookahead in sorted(set([min(8, interval), min(32, interval), interval])):
                            prefix = selected.iloc[:lookahead]
                            outcomes.append({"controller": controller, "seed": seed, "segment_clock": clock,
                                             "block": block, "clock": clock + block * interval,
                                             "phase_at_decision": stream.phases[clock + block * interval],
                                             "lookahead": lookahead, "selected_action": segment_decisions[block]["selected_action"],
                                             "mae": prefix.mae.mean(), "mse": prefix.mse.mean()})
                restore_rng(saved_rng)
                plot_forecasts(frames, args.output / f"start_{clock}_errors", f"{dataset} {method} H={horizon} | start {clock}")
                pd.DataFrame(decisions).to_csv(args.output / "decisions.csv", index=False)
                pd.DataFrame(trajectory_summaries).to_csv(args.output / "trajectory_summary.csv", index=False)
                print(f"segment {clock}: {len(plan['controllers'])} trajectories complete", flush=True)
            if index < max(starts):
                replay.step(stream.dataset, index)
            if (index - first) % 2048 == 0:
                print(f"native warmup: raw clock {index + stream.dataset.context_length - 1}", flush=True)
        result = pd.DataFrame(outcomes)
        keys = ["seed", "segment_clock", "block", "lookahead"]
        native = result[result.controller == "native"][keys + ["mae", "mse"]].rename(columns={"mae": "native_mae", "mse": "native_mse"})
        result = result.merge(native, on=keys, validate="many_to_one")
        for metric in ["mae", "mse"]:
            result[f"{metric}_gain_pct"] = 100 * (1 - result[metric] / result[f"native_{metric}"])
        result.to_csv(args.output / "block_outcomes.csv", index=False)
        summary = result.groupby(["controller", "lookahead"])[["mae", "mse", "native_mae", "native_mse"]].mean().reset_index()
        for metric in ["mae", "mse"]:
            summary[f"{metric}_gain_pct"] = 100 * (1 - summary[metric] / summary[f"native_{metric}"])
        summary.to_csv(args.output / "summary.csv", index=False)
        result.groupby(["controller", "lookahead", "phase_at_decision"])[[
            "mae", "mse", "mae_gain_pct", "mse_gain_pct"
        ]].mean().to_csv(args.output / "phase_summary.csv")
        plan["status"] = "complete"
        print(summary[summary.lookahead == interval].to_string(index=False), flush=True)
    except Exception as error:
        plan.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save_json(args.output / "manifest.json", plan)


if __name__ == "__main__":
    main()

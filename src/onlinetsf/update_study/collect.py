"""Collect uniformly spaced, strictly historical action comparisons in parallel.

One producer replays native history once and supplies complete snapshots to
branch workers. The final pre-test snapshot is reused by experiment 2.
No offline fitting is repeated. --plan-only never imports PyTorch.
By default only consolidated results are written; --save-branch-logs also saves
per-action forecasts, updates, phase metrics and summaries.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd

from ..decision.actions import policies
from ..decision.config import DecisionConfig, FEATURE_SCHEMA_VERSION
from .policy_support import DEFAULT_ANNOTATIONS, DEFAULT_RUNS, file_sha256, find_job, save_json, save_timings


def initialize_worker(plan: dict) -> None:
    """Load immutable inputs once per worker, rather than once per snapshot."""
    global _worker_stream
    import torch
    from onlinetsf.update_study.data import load_stream

    torch.set_num_threads(plan["threads"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    _worker_stream = load_stream(Path(plan["annotations"]) / plan["dataset"], plan["context_length"],
                                 plan["horizon"], plan["prefix_rows"])


def collect_worker(plan: dict, clock: int, snapshot: Path) -> tuple[list[dict], list[dict]]:
    """Evaluate actions from a producer snapshot; never replay historical updates."""
    from onlinetsf.update_study.run import run_branch
    from onlinetsf.update_study.runtime import Policy, load_replay, restore_rng, state_features

    replay, saved_rng = load_replay(snapshot, plan["device"])
    actions = [Policy.from_dict(item) for item in plan["actions"]]
    metadata = {key: plan[key] for key in ["dataset", "method", "horizon", "seed"]}
    index = clock - plan["context_length"] + 1
    if replay.last_index != index - 1:
        raise ValueError("snapshot is not immediately before the requested decision")
    probe_id = f"uniform_{clock}"
    state = dict(metadata, probe_id=probe_id, index=index, clock=clock, kind="uniform",
                 **state_features(_worker_stream, replay, index))
    state["steps_since_gradient"] = replay.method.gradient_age(index)
    outcomes = []
    for action in actions:
        restore_rng(saved_rng)
        result = run_branch(replay, _worker_stream, {"index": index}, action,
                            Path(plan["output"]) / "probes" / probe_id / action.name,
                            plan["decision_interval"], [plan["decision_interval"]],
                            save_logs=plan["save_branch_logs"])
        outcomes.extend(dict(metadata, probe_id=probe_id, **row) for row in result)
    print(f"{probe_id}: {len(actions)} actions complete", flush=True)
    return [state], outcomes


def collect_history(plan: dict) -> tuple[list[dict], list[dict]]:
    """Advance one native trajectory, with at most workers temporary snapshots."""
    import torch
    from onlinetsf.update_study.data import load_stream
    from onlinetsf.update_study.runtime import ControlledMethod, Policy, Replay, restore_rng, rng_state

    torch.set_num_threads(plan["threads"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    stream = load_stream(Path(plan["annotations"]) / plan["dataset"], plan["context_length"],
                         plan["horizon"], plan["prefix_rows"])
    checkpoint = torch.load(plan["checkpoint"], map_location=plan["device"], weights_only=False)
    if checkpoint["prefix_rows"] != plan["prefix_rows"]:
        raise ValueError("checkpoint prefix differs from collection plan")
    method = checkpoint["method"]
    method.device = torch.device(plan["device"])
    initial_rng = (checkpoint["rng"][0].cpu(),
                   [item.cpu() for item in checkpoint["rng"][1]] if checkpoint["rng"][1] is not None else None)
    if initial_rng[1] is not None and torch.cuda.is_available() and len(initial_rng[1]) != torch.cuda.device_count():
        raise ValueError("use the same visible GPU count as the source checkpoint, normally one")
    restore_rng(initial_rng)
    replay = Replay(ControlledMethod(method, Policy("native")), plan["horizon"],
                    DecisionConfig.from_dict(plan["decision_config"]))
    length = plan["context_length"]
    selected = set(plan["clocks"])
    states, outcomes = [], []
    native_seconds = 0.0
    with TemporaryDirectory(prefix="snapshots-", dir=plan["output"]) as temporary:
        with ProcessPoolExecutor(max_workers=plan["workers"], mp_context=multiprocessing.get_context("spawn"),
                                 initializer=initialize_worker, initargs=(plan,)) as pool:
            pending = {}
            # Stop before receiving feedback or predicting at the first test clock.
            for index in range(plan["prefix_rows"] - length, plan["test_start"] - length + 1):
                clock = index + length - 1
                if clock in selected:
                    snapshot = Path(temporary) / f"{clock}.pt"
                    torch.save({"replay": replay, "rng": rng_state()}, snapshot)
                    pending[pool.submit(collect_worker, plan, clock, snapshot)] = snapshot
                    if len(pending) == plan["workers"]:
                        done, _ = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            state_rows, outcome_rows = future.result()
                            states.extend(state_rows)
                            outcomes.extend(outcome_rows)
                            pending.pop(future).unlink()
                step_began = perf_counter()
                replay.step(stream.dataset, index)
                if method.device.type == "cuda":
                    torch.cuda.synchronize(method.device)
                native_seconds += perf_counter() - step_began
            torch.save({"replay": replay, "rng": rng_state()}, Path(plan["output"]) / "evaluation_start.pt")
            for future, snapshot in pending.items():
                state_rows, outcome_rows = future.result()
                states.extend(state_rows)
                outcomes.extend(outcome_rows)
                snapshot.unlink()
    plan.setdefault("timings", {})["collection_native_seconds"] = native_seconds
    return states, outcomes


def main(argv: list[str] | None = None) -> None:
    began = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--dataset", default="ETTh2")
    parser.add_argument("--method", choices=["onenet", "fsnet", "dsof"], default="onenet")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1)
    boundary = parser.add_mutually_exclusive_group()
    boundary.add_argument("--test-start", type=int)
    boundary.add_argument("--test-start-fraction", type=float, default=.6)
    parser.add_argument("--positions", type=int, default=16)
    parser.add_argument("--decision-config", type=Path, help="override features/actions without retraining the offline model")
    parser.add_argument("--decision-interval", type=int, help="positive branch length and test decision interval")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--save-branch-logs", action="store_true",
                        help="also write detailed per-action logs (default: consolidated results only)")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.horizon, args.positions, args.workers, args.threads) <= 0 or args.workers > args.positions:
        parser.error("sizes must be positive and workers must not exceed positions")
    if not 0 < args.test_start_fraction < 1:
        parser.error("test-start-fraction must be between zero and one")
    job = find_job(args.runs, args.dataset, args.method, args.horizon, args.seed)
    source_manifest = json.loads((job.parent.parent / "manifest.json").read_text(encoding="utf-8"))
    if source_manifest.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("regenerate the source checkpoint with the current feature schema")
    decision = (DecisionConfig.load(args.decision_config) if args.decision_config else
                DecisionConfig.from_dict(source_manifest["decision_config"]))
    if args.decision_interval is not None:
        from dataclasses import replace

        decision = replace(decision, interval=args.decision_interval)
    args.decision_interval = decision.interval
    record = next(row for row in source_manifest["jobs"] if row["dataset"] == args.dataset
                  and row["method"] == args.method and row["horizon"] == args.horizon and row["seed"] == args.seed)
    annotation_dir = args.annotations / args.dataset
    annotation = json.loads((annotation_dir / "manifest.json").read_text(encoding="utf-8"))
    source_hash = file_sha256(annotation_dir / "labeled_data.csv")
    if source_hash != record["input_sha256"]:
        raise ValueError("annotations differ from source run")
    prefix = int(record["prefix_rows"])
    test_start = args.test_start if args.test_start is not None else int(annotation["rows"] * args.test_start_fraction)
    # Leave enough native history for 128 matured forecast errors at the first decision.
    first = prefix + args.horizon + decision.error_window
    last = test_start - args.decision_interval - args.horizon
    if last - first + 1 < args.positions or test_start > annotation["rows"] - args.horizon - 1:
        parser.error("training interval is too short for the requested positions or test boundary")
    clocks = np.linspace(first, last, args.positions, dtype=int).tolist()
    actions = [item.to_dict() for item in policies(tuple(record["update_groups"]), decision)]
    names = [item["name"] for item in actions]
    if "native" not in names or len(names) != len(set(names)):
        raise ValueError("source actions must include native and have unique names")
    plan = {"status": "planned", "data_feature_precision": "float64", "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "dataset": args.dataset, "method": args.method, "horizon": args.horizon,
            "seed": args.seed, "prefix_rows": prefix, "context_length": int(record["context_length"]),
            "test_start": test_start, "decision_interval": args.decision_interval,
            "positions": args.positions, "clocks": clocks, "workers": args.workers,
            "device": args.device, "threads": args.threads, "annotations": str(args.annotations),
            "save_branch_logs": args.save_branch_logs,
            "output": str(args.output), "source_job": str(job), "input_sha256": source_hash,
            "checkpoint": str(job / "initial.pt"), "checkpoint_sha256": file_sha256(job / "initial.pt"),
            "decision_config": decision.to_dict(), "update_groups": record["update_groups"],
            "method_features": record["method_features"],
            "actions": actions, "train_target_end_exclusive": last + args.decision_interval + args.horizon}
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "manifest.json", plan)
    pd.DataFrame({"clock": clocks, "target_end_exclusive": np.array(clocks) + args.decision_interval + args.horizon}
                 ).to_csv(args.output / "positions.csv", index=False)
    print(json.dumps(plan, indent=2), flush=True)
    if args.plan_only:
        return
    plan["status"] = "running"
    save_json(args.output / "manifest.json", plan)
    try:
        states, outcomes = collect_history(plan)
        state_frame = pd.DataFrame(states).sort_values("clock").reset_index(drop=True)
        outcome_frame = pd.DataFrame(outcomes).sort_values(["probe_id", "policy"]).reset_index(drop=True)
        if len(state_frame) != args.positions or len(outcome_frame) != args.positions * len(actions):
            raise RuntimeError("collection did not return all state/action comparisons")
        if set(state_frame.clock) != set(clocks) or state_frame.probe_id.duplicated().any():
            raise RuntimeError("collection returned incorrect or duplicate decision positions")
        if not np.isfinite(outcome_frame[["mae", "mse"]].to_numpy()).all():
            raise FloatingPointError("collection returned nonfinite losses")
        state_frame.to_csv(args.output / "probe_states.csv", index=False)
        outcome_frame.to_csv(args.output / "probe_outcomes.csv", index=False)
        plan.update(status="complete", states_sha256=file_sha256(args.output / "probe_states.csv"),
                    outcomes_sha256=file_sha256(args.output / "probe_outcomes.csv"),
                    evaluation_start_sha256=file_sha256(args.output / "evaluation_start.pt"))
        plan.setdefault("timings", {})["collection_branch_seconds_sum"] = float(outcome_frame.branch_seconds.sum())
    except Exception as error:
        plan.update(status="failed", error=str(error))
        raise
    finally:
        timings = plan.setdefault("timings", {})
        timings["collection_wall_seconds"] = perf_counter() - began
        save_timings(args.output, timings)
        save_json(args.output / "manifest.json", plan)
        print("Collection timings (seconds): " + json.dumps(timings), flush=True)


if __name__ == "__main__":
    main()

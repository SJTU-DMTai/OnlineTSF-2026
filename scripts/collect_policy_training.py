"""Collect uniformly spaced, strictly historical action comparisons in parallel.

Workers share the visible GPU but have independent native replays. Each action
fork restores the same RNG state, and the native history is restored afterwards.
No offline fitting is repeated. --plan-only never imports PyTorch.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from update_policy_common import DEFAULT_ANNOTATIONS, DEFAULT_RUNS, file_sha256, find_job, save_json


def collect_worker(plan: dict, clocks: list[int], worker: int) -> tuple[list[dict], list[dict]]:
    """Replay one shard; action forks never modify the native state or RNG."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import torch
    from onlinetsf.update_study.data import load_stream
    from onlinetsf.update_study.run import run_branch
    from onlinetsf.update_study.runtime import ControlledMethod, Policy, Replay, restore_rng, rng_state, state_features

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
    method.model.to(method.device)
    initial_rng = (checkpoint["rng"][0].cpu(),
                   [item.cpu() for item in checkpoint["rng"][1]] if checkpoint["rng"][1] is not None else None)
    if initial_rng[1] is not None and torch.cuda.is_available() and len(initial_rng[1]) != torch.cuda.device_count():
        raise ValueError("use the same visible GPU count as the source checkpoint, normally one")
    restore_rng(initial_rng)
    replay = Replay(ControlledMethod(method, Policy("native")), plan["horizon"])
    actions = [Policy(item["name"], item["mode"], item["period"], item["lr_scale"],
                      None if item["groups"] == "all" else frozenset(item["groups"])) for item in plan["actions"]]
    metadata = {key: plan[key] for key in ["dataset", "method", "horizon", "seed"]}
    states, outcomes = [], []
    length = stream.dataset.context_length
    selected = set(clocks)
    for index in range(stream.prefix_rows - length, max(clocks) - length + 2):
        clock = index + length - 1
        if clock in selected:
            probe_id = f"uniform_{clock}"
            state = dict(metadata, probe_id=probe_id, index=index, clock=clock, kind="uniform",
                         **state_features(stream, replay, index))
            states.append(state)
            saved_rng = rng_state()
            for action in actions:
                restore_rng(saved_rng)
                result = run_branch(replay, stream, {"index": index}, action,
                                    Path(plan["output"]) / "probes" / probe_id / action.name,
                                    plan["decision_interval"], [plan["decision_interval"]])
                outcomes.extend(dict(metadata, probe_id=probe_id, **row) for row in result)
            restore_rng(saved_rng)
            print(f"worker {worker}: {probe_id}, {len(actions)} actions complete", flush=True)
        if clock < max(clocks):
            replay.step(stream.dataset, index)
    return states, outcomes


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--dataset", default="ETTh2")
    parser.add_argument("--method", choices=["onenet", "fsnet"], default="onenet")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1)
    boundary = parser.add_mutually_exclusive_group()
    boundary.add_argument("--test-start", type=int)
    boundary.add_argument("--test-start-fraction", type=float, default=.6)
    parser.add_argument("--positions", type=int, default=16)
    parser.add_argument("--decision-interval", type=int, choices=[8, 32, 128], default=128)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.horizon, args.positions, args.workers, args.threads) <= 0 or args.workers > args.positions:
        parser.error("sizes must be positive and workers must not exceed positions")
    if not 0 < args.test_start_fraction < 1:
        parser.error("test-start-fraction must be between zero and one")
    job = find_job(args.runs, args.dataset, args.method, args.horizon, args.seed)
    source_manifest = json.loads((job.parent.parent / "manifest.json").read_text(encoding="utf-8"))
    record = next(row for row in source_manifest["jobs"] if row["dataset"] == args.dataset
                  and row["method"] == args.method and row["horizon"] == args.horizon and row["seed"] == args.seed)
    annotation_dir = args.annotations / args.dataset
    annotation = json.loads((annotation_dir / "manifest.json").read_text(encoding="utf-8"))
    source_hash = file_sha256(annotation_dir / "labeled_data.csv")
    if source_hash != record["input_sha256"]:
        raise ValueError("annotations differ from source run")
    source_states = pd.read_csv(job / "probe_states.csv")
    lengths = (source_states.clock - source_states["index"] + 1).unique()
    if len(lengths) != 1:
        raise ValueError("source context lengths differ")
    prefix = int(record["prefix_rows"])
    test_start = args.test_start if args.test_start is not None else int(annotation["rows"] * args.test_start_fraction)
    # Leave enough native history for 128 matured forecast errors at the first decision.
    first = prefix + args.horizon + 128
    last = test_start - args.decision_interval - args.horizon
    if last - first + 1 < args.positions or test_start > annotation["rows"] - args.horizon - 1:
        parser.error("training interval is too short for the requested positions or test boundary")
    clocks = np.linspace(first, last, args.positions, dtype=int).tolist()
    actions = json.loads((job / "policies.json").read_text(encoding="utf-8"))
    names = [item["name"] for item in actions]
    if "native" not in names or len(names) != len(set(names)):
        raise ValueError("source actions must include native and have unique names")
    plan = {"status": "planned", "dataset": args.dataset, "method": args.method, "horizon": args.horizon,
            "seed": args.seed, "prefix_rows": prefix, "context_length": int(lengths[0]),
            "test_start": test_start, "decision_interval": args.decision_interval,
            "positions": args.positions, "clocks": clocks, "workers": args.workers,
            "device": args.device, "threads": args.threads, "annotations": str(args.annotations),
            "output": str(args.output), "source_job": str(job), "input_sha256": source_hash,
            "checkpoint": str(job / "initial.pt"), "checkpoint_sha256": file_sha256(job / "initial.pt"),
            "actions": actions, "train_target_end_exclusive": last + args.decision_interval + args.horizon}
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "manifest.json", plan)
    pd.DataFrame({"clock": clocks, "target_end_exclusive": np.array(clocks) + args.decision_interval + args.horizon,
                  "worker": np.arange(args.positions) % args.workers}).to_csv(args.output / "positions.csv", index=False)
    print(json.dumps(plan, indent=2), flush=True)
    if args.plan_only:
        return
    plan["status"] = "running"
    save_json(args.output / "manifest.json", plan)
    try:
        states, outcomes = [], []
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            pending = [pool.submit(collect_worker, plan, clocks[worker::args.workers], worker)
                       for worker in range(args.workers)]
            for future in as_completed(pending):
                state_rows, outcome_rows = future.result()
                states.extend(state_rows)
                outcomes.extend(outcome_rows)
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
                    outcomes_sha256=file_sha256(args.output / "probe_outcomes.csv"))
    except Exception as error:
        plan.update(status="failed", error=str(error))
        raise
    finally:
        save_json(args.output / "manifest.json", plan)


if __name__ == "__main__":
    main()

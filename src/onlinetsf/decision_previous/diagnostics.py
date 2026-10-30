"""Add local Oracle diagnostics without repeating fitting or other trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .config import FEATURE_SCHEMA_VERSION
from ..update_study.policy_support import file_sha256, oracle_decision, save_json, summarize_decisions


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True, help="completed experiment/ directory")
    parser.add_argument("--annotations", type=Path, required=True, help="parent of the labeled dataset directory")
    parser.add_argument("--controller", choices=["policy_full", "policy_data", "both"], default="policy_full")
    parser.add_argument("--metric", choices=["mae", "mse"], default="mae")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True, help="new diagnostics directory; original results stay intact")
    args = parser.parse_args(argv)
    if args.threads <= 0:
        parser.error("threads must be positive")
    source = json.loads((args.experiment / "manifest.json").read_text(encoding="utf-8"))
    selector_path = args.experiment / "selectors.json"
    bundle = json.loads(selector_path.read_text(encoding="utf-8"))
    if source["status"] != "complete" or file_sha256(selector_path) != source["selector_sha256"]:
        raise ValueError("use a complete experiment with its original selector artifact")
    if bundle.get("schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("unsupported selector artifact; refit with the current feature schema")
    annotation = args.annotations / bundle["dataset"]
    if file_sha256(annotation / "labeled_data.csv") != bundle["input_sha256"]:
        raise ValueError("annotated data differ from the original experiment")
    start = bundle["test_start"] - bundle["context_length"] + 1
    steps, interval = source["steps"], bundle["decision_interval"]
    reference_decisions = pd.read_csv(args.experiment / "decisions.csv")
    controllers = ["policy_full", "policy_data"] if args.controller == "both" else [args.controller]
    args.output.mkdir(parents=True, exist_ok=False)
    plan = {"status": "running", "experiment": str(args.experiment), "controllers": controllers,
            "metric": args.metric, "selector_sha256": source["selector_sha256"],
            "input_sha256": bundle["input_sha256"], "steps": steps, "decision_interval": interval,
            "offline_retrained": False, "selectors_refitted": False, "pretest_history_replayed": False,
            "results": []}
    began = perf_counter()
    replay = stream = saved_rng = None
    save_json(args.output / "manifest.json", plan)
    try:
        for controller in controllers:
            original = pd.read_csv(args.experiment / controller / "forecasts.csv")
            decisions = reference_decisions[reference_decisions.controller == controller].sort_values("clock")
            if (not np.array_equal(original.forecast_index, np.arange(start, start + steps))
                    or not np.array_equal(decisions.clock, np.arange(bundle["test_start"], bundle["test_start"] + steps, interval))):
                raise ValueError(f"{controller}: original results do not cover the test interval")
            directory = args.output / controller
            outcomes_path = args.experiment / controller / "action_outcomes.csv"
            if outcomes_path.exists():
                directory.mkdir()
                outcomes = pd.read_csv(outcomes_path)
                result = {"controller": controller, "source": "saved_action_outcomes", "trajectory_replayed": False}
            else:
                # Load the pre-test state only if candidate outcomes are absent.
                # No initial.pt, source model fitting, or native warmup is needed.
                if replay is None:
                    import torch

                    from ..update_study.data import load_stream
                    from ..update_study.runtime import load_replay

                    torch.set_num_threads(args.threads)
                    torch.backends.cudnn.benchmark = False
                    torch.backends.cudnn.deterministic = True
                    snapshot = args.experiment / "evaluation_start.pt"
                    if file_sha256(snapshot) != source["evaluation_start_sha256"]:
                        raise ValueError("pre-test snapshot differs from the original experiment")
                    replay, saved_rng = load_replay(snapshot, args.device)
                    if replay.last_index != start - 1 or replay.horizon != bundle["horizon"]:
                        raise ValueError("snapshot does not immediately precede testing")
                    stream = load_stream(annotation, bundle["context_length"], bundle["horizon"], bundle["prefix_rows"])
                from ..update_study.policy_experiment import run_segment
                from ..update_study.runtime import restore_rng
                from .actions import Policy

                restore_rng(saved_rng)
                actions = {item["name"]: Policy.from_dict(item) for item in bundle["actions"]}
                forecasts, replayed_decisions, costs = run_segment(
                    replay, stream, start, steps, interval, controller, bundle, actions, directory,
                    oracle_metric=args.metric,
                )
                if ([row["selected_action"] for row in replayed_decisions] != decisions.selected_action.tolist()
                        or not np.allclose(forecasts[["mae", "mse"]], original[["mae", "mse"]], rtol=1e-7, atol=1e-9)):
                    raise RuntimeError(f"{controller}: replay differs from saved actions or forecast losses; diagnostics rejected")
                outcomes = pd.read_csv(directory / "action_outcomes.csv")
                result = {"controller": controller, "source": "verified_controller_replay",
                          "trajectory_replayed": True, "matches_original": True, "costs": costs}
            rows = []
            names = [item["name"] for item in bundle["actions"]]
            for block in range(steps // interval):
                group = outcomes[outcomes.block == block]
                if (len(group) != len(names) or set(group.action) != set(names)
                        or not group.block_steps.eq(interval).all()
                        or not group.clock.eq(bundle["test_start"] + block * interval).all()):
                    raise ValueError(f"{controller}: incomplete candidate outcomes at block {block}")
                selected = decisions.iloc[block].selected_action
                row = oracle_decision(group.to_dict("records"), selected, args.metric)
                actual = original.iloc[block * interval:(block + 1) * interval][args.metric].mean()
                if not np.isclose(actual, row["selected_loss"], rtol=1e-7, atol=1e-9):
                    raise RuntimeError(f"{controller}: selected candidate loss differs from original forecasts")
                rows.append(dict(block=block, clock=int(group.clock.iloc[0]), **row))
            frame = pd.DataFrame(rows)
            summary = (summarize_decisions(frame, args.metric) if rows else
                       {"metric": args.metric, "decisions": 0, "acc": None, "mean_relative_regret": None})
            outcomes.to_csv(directory / "action_outcomes.csv", index=False)
            frame.to_csv(directory / "oracle_decisions.csv", index=False)
            save_json(directory / "oracle_summary.json", summary)
            result["summary"] = summary
            plan["results"].append(result)
            save_json(args.output / "manifest.json", plan)
            print(f"{controller}: {len(rows)} full blocks; {result['source']}; output={directory}", flush=True)
        plan["status"] = "complete"
    except (Exception, SystemExit, KeyboardInterrupt) as error:
        plan.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        plan["wall_seconds"] = perf_counter() - began
        save_json(args.output / "manifest.json", plan)

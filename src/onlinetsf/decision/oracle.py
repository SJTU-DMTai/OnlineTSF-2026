"""Greedy hindsight Oracle over the current finite update-action catalog."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import yaml

from ..online import OnlineExecutor
from .actions import positive_integer
from .catalog import ActionCatalog
from .experiment import evaluate_policy, restore_rng, save_json, score_action


class OracleExecutor:
    """Select before each data clock, then let the ordinary executor train causally.

    Every candidate starts from the same pre-observation state and RNG. Search
    rolls back its branches; the winner is executed again on the real trajectory.
    The policy therefore retains normal period phases, pending labels and batches.
    """

    def __init__(self, executor: OnlineExecutor, dataset, catalog: ActionCatalog, stop: int) -> None:
        self.executor, self.dataset, self.catalog, self.stop = executor, dataset, catalog, stop
        self.method = executor.method
        self.method.expires_at = None
        self._events = executor._events
        self.candidate_search_seconds = 0.
        self.candidate_evaluations = 0

    def step(self, index, context, target, *, decision_context=None, final=False):
        selection = None
        if self.method.expires_at is None or index >= self.method.expires_at:
            began = perf_counter()
            outcomes = []
            for name, action in zip(self.catalog.ids, self.catalog.actions):
                features, score = score_action(self.executor, self.dataset, index, action, stop=self.stop)
                outcomes.append({"action_id": name, "duration": action.duration,
                                 "scored_steps": min(action.duration, self.stop - index), **score})
            losses = np.asarray([[row["mae"] for row in outcomes]])
            chosen = int(self.catalog.best(-losses)[0])
            seconds = perf_counter() - began
            self.candidate_search_seconds += seconds
            self.candidate_evaluations += len(outcomes)
            action = self.catalog.actions[chosen]
            self.method.policy, self.method.expires_at = action, None
            selection = {"action_id": self.catalog.ids[chosen], "score_kind": "future_mae", "features": features,
                         "selected_future_mae": outcomes[chosen]["mae"],
                         "selected_future_mse": outcomes[chosen]["mse"],
                         "executed_steps": outcomes[chosen]["scored_steps"],
                         "candidate_search_seconds": seconds, "candidate_outcomes": outcomes}
            print(f"oracle origin={index} action={self.catalog.ids[chosen]} H={action.duration} "
                  f"steps={outcomes[chosen]['scored_steps']} future_mae={outcomes[chosen]['mae']:.6f}", flush=True)
        result = self.executor.step(index, context, target, decision_context=decision_context, final=final)
        if selection is not None:
            self.method.last_decision.update(selection)
        return result

    def flush(self):
        return self.executor.flush()


def evaluate_oracle(data_directory: Path, output: Path, *, actions: Path | None = None,
                    steps: int | None = None) -> dict:
    """Evaluate just Oracle from an existing collection, without fitting selectors."""
    manifest_path = data_directory / "manifest.json"
    collection = json.loads(manifest_path.read_text(encoding="utf-8"))
    if collection["status"] != "complete":
        raise ValueError("collection must complete before Oracle evaluation")
    torch.set_num_threads(collection["config"].get("threads", 1))
    saved = torch.load(data_directory / "evaluation_start.pt", weights_only=False)
    executor, dataset = saved["executor"], saved["dataset"]
    capabilities = executor.method.capabilities
    method = collection["catalog"]["method"]
    catalog = (ActionCatalog.from_dict(collection["catalog"], capabilities) if actions is None
               else ActionCatalog.load(actions, method, capabilities))
    start, stop = collection["test_start"], collection["test_stop"]
    if steps is not None:
        positive_integer(steps, "steps")
        stop = min(stop, start + steps)
    oracle = OracleExecutor(executor, dataset, catalog, stop)
    output.mkdir(parents=True, exist_ok=False)
    (output / "actions.yaml").write_text(yaml.safe_dump(catalog.to_dict(), sort_keys=False), encoding="utf-8")
    plan = {"status": "evaluating", "data": str(data_directory.resolve()),
            "collection_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "controller": "oracle_greedy", "uses_future_labels_for_selection": True,
            "objective": "minimum standardized MAE over each candidate's next min(H, remaining) origins",
            "test_start": start, "test_stop": stop, "candidate_count": len(catalog.actions)}
    save_json(output / "manifest.json", plan)
    try:
        restore_rng(saved["rng"])
        directory = output / "oracle_greedy"
        summary = evaluate_policy(oracle, dataset, start, stop, directory, name="oracle_greedy",
                                  target_mean=np.asarray(collection["target_mean"]),
                                  target_std=np.asarray(collection["target_std"]))
        summary.update(candidate_search_seconds=oracle.candidate_search_seconds,
                       candidate_evaluations=oracle.candidate_evaluations,
                       candidate_search_role="decision", uses_future_labels_for_selection=True)
        # Check that searched losses really describe the executed winning branches.
        with (directory / "forecasts.csv").open(encoding="utf-8", newline="") as handle:
            forecasts = list(csv.DictReader(handle))
        with (directory / "decisions.jsonl").open(encoding="utf-8") as decision_file, \
                (directory / "action_outcomes.csv").open("w", encoding="utf-8", newline="") as outcome_file:
            writer = csv.DictWriter(outcome_file, fieldnames=["decision_index", "origin", "action_id", "duration",
                                                             "scored_steps", "mae", "mse", "selected"])
            writer.writeheader()
            for line in decision_file:
                decision = json.loads(line)
                offset, count = decision["index"] - start, decision["executed_steps"]
                selected_forecasts = forecasts[offset:offset + count]
                for metric in ("mae", "mse"):
                    actual = np.mean([float(row[metric]) for row in selected_forecasts])
                    if not np.isclose(actual, decision[f"selected_future_{metric}"], rtol=1e-7, atol=1e-10):
                        raise RuntimeError(f"Oracle selected branch differs from executed {metric} at {decision['index']}")
                for candidate in decision["candidate_outcomes"]:
                    writer.writerow({"decision_index": decision["decision_index"], "origin": decision["index"],
                                     **candidate, "selected": candidate["action_id"] == decision["action_id"]})
        save_json(directory / "summary.json", summary)
        with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
            fields = [key for key, value in summary.items() if not isinstance(value, list)]
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerow(summary)
        plan.update(status="complete", results=[summary], selected_branches_match_trajectory=True)
    except BaseException as error:
        plan.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save_json(output / "manifest.json", plan)
    return summary

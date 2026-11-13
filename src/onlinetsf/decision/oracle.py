"""Per-origin hindsight Oracles over the current update-action catalog."""

from __future__ import annotations

import csv
import copy
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
from .selector import resolve_ema_beta


class OracleExecutor:
    """Select before each data clock, then let the ordinary executor train causally.

    Every candidate starts from the same pre-observation state and RNG. Search
    rolls back its branches; the winner is executed again on the real trajectory.
    The policy retains pending labels and optimizer state.
    """

    def __init__(self, executor: OnlineExecutor, dataset, catalog: ActionCatalog, stop: int,
                 *, ema_beta: float | None = None) -> None:
        self.executor, self.dataset, self.catalog, self.stop = executor, dataset, catalog, stop
        self.method = executor.method
        self._events = executor._events
        self.beta = resolve_ema_beta(ema_beta) if ema_beta is not None else None
        self.smoothed_maes: np.ndarray | None = None
        self.candidate_search_seconds = 0.
        self.candidate_evaluations = 0

    def step(self, index, context, target, *, decision_context=None, final=False):
        began = perf_counter()
        outcomes = []
        for name, action in zip(self.catalog.ids, self.catalog.actions):
            features, score = score_action(self.executor, self.dataset, index, action, steps=1, stop=self.stop)
            outcomes.append({"action_id": name, "scored_steps": 1, **score})
        losses = np.asarray([row["mae"] for row in outcomes])
        if self.beta is not None:
            self.smoothed_maes = (losses.copy() if self.smoothed_maes is None else
                                  self.beta * self.smoothed_maes + (1 - self.beta) * losses)
        ranked = losses if self.smoothed_maes is None else self.smoothed_maes
        for row, value in zip(outcomes, ranked):
            row["ema_mae"] = float(value)
        chosen = int(self.catalog.best(-ranked[None, :])[0])
        seconds = perf_counter() - began
        self.candidate_search_seconds += seconds
        self.candidate_evaluations += len(outcomes)
        action = self.catalog.actions[chosen]
        self.method.policy = action
        selection = {"action_id": self.catalog.ids[chosen],
                     "score_kind": "current_mae" if self.beta is None else "ema_current_mae",
                     "features": features, "ema_beta": self.beta,
                     "selected_current_mae": outcomes[chosen]["mae"],
                     "selected_ema_mae": outcomes[chosen]["ema_mae"],
                     "selected_current_mse": outcomes[chosen]["mse"],
                     "executed_steps": outcomes[chosen]["scored_steps"],
                     "candidate_search_seconds": seconds, "candidate_outcomes": outcomes}
        result = self.executor.step(index, context, target, decision_context=decision_context, final=final)
        self.method.last_decision.update(selection)
        return result

    def flush(self):
        return self.executor.flush()


def evaluate_oracle(data_directory: Path, output: Path, *, actions: Path | None = None,
                    steps: int | None = None, ema_beta: float | None = None) -> dict:
    """Evaluate just Oracle from an existing collection, without fitting selectors."""
    manifest_path = data_directory / "manifest.json"
    collection = json.loads(manifest_path.read_text(encoding="utf-8"))
    if collection["status"] != "complete":
        raise ValueError("collection must complete before Oracle evaluation")
    beta = resolve_ema_beta(ema_beta if ema_beta is not None else
                            collection["config"].get("evaluation", {}).get("ema_beta"), collection["tau"])
    torch.set_num_threads(collection["config"].get("threads", 1))
    saved = torch.load(data_directory / "evaluation_start.pt", weights_only=False)
    dataset = saved["dataset"]
    capabilities = saved["executor"].method.capabilities
    method = collection["catalog"]["method"]
    catalog = (ActionCatalog.from_dict(collection["catalog"], capabilities) if actions is None
               else ActionCatalog.load(actions, method, capabilities))
    start, stop = collection["test_start"], collection["test_stop"]
    if steps is not None:
        positive_integer(steps, "steps")
        stop = min(stop, start + steps)
    output.mkdir(parents=True, exist_ok=False)
    (output / "actions.yaml").write_text(yaml.safe_dump(catalog.to_dict(), sort_keys=False), encoding="utf-8")
    plan = {"status": "evaluating", "data": str(data_directory.resolve()),
            "collection_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "controllers": ["oracle_greedy", "oracle_ema"], "uses_future_labels_for_selection": True,
            "objective": "minimum current standardized MAE, with or without per-action EMA",
            "ema_beta": beta,
            "test_start": start, "test_stop": stop, "candidate_count": len(catalog.actions)}
    save_json(output / "manifest.json", plan)
    summaries = []
    try:
        for name, policy_beta in (("oracle_greedy", None), ("oracle_ema", beta)):
            restore_rng(saved["rng"])
            oracle = OracleExecutor(copy.deepcopy(saved["executor"]), dataset, catalog, stop, ema_beta=policy_beta)
            directory = output / name
            summary = evaluate_policy(oracle, dataset, start, stop, directory, name=name,
                                      target_mean=np.asarray(collection["target_mean"]),
                                      target_std=np.asarray(collection["target_std"]))
            summary.update(candidate_search_seconds=oracle.candidate_search_seconds,
                           candidate_evaluations=oracle.candidate_evaluations,
                           candidate_search_role="decision", uses_future_labels_for_selection=True)
            with (directory / "forecasts.csv").open(encoding="utf-8", newline="") as handle:
                forecasts = list(csv.DictReader(handle))
            with (directory / "decisions.jsonl").open(encoding="utf-8") as decision_file, \
                    (directory / "action_outcomes.csv").open("w", encoding="utf-8", newline="") as outcome_file:
                writer = csv.DictWriter(outcome_file, fieldnames=["decision_index", "origin", "action_id",
                                                                 "scored_steps", "mae", "mse", "ema_mae", "selected"])
                writer.writeheader()
                for line in decision_file:
                    decision = json.loads(line)
                    offset = decision["index"] - start
                    for metric in ("mae", "mse"):
                        actual = float(forecasts[offset][metric])
                        if not np.isclose(actual, decision[f"selected_current_{metric}"], rtol=1e-7, atol=1e-10):
                            raise RuntimeError(f"Oracle selected branch differs from executed {metric} at {decision['index']}")
                    for candidate in decision["candidate_outcomes"]:
                        writer.writerow({"decision_index": decision["decision_index"], "origin": decision["index"],
                                         **candidate, "selected": candidate["action_id"] == decision["action_id"]})
            save_json(directory / "summary.json", summary)
            summaries.append(summary)
        with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
            fields = [key for key, value in summaries[0].items() if not isinstance(value, list)]
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summaries)
        plan.update(status="complete", results=summaries, selected_branches_match_trajectory=True)
    except BaseException as error:
        plan.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save_json(output / "manifest.json", plan)
    return summaries

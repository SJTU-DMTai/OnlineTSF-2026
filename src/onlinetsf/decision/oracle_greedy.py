"""Per-origin hindsight Oracles over the current update-action catalog."""

from __future__ import annotations

import csv
import copy
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from ..online import OnlineExecutor
from .catalog import ActionCatalog
from .experiment import evaluate_policy, restore_rng, save_json, score_action
from .selector import resolve_ema_beta


class GreedyOracleExecutor:
    """Select before each data clock, then let the ordinary executor train causally.

    Every candidate starts from the same pre-observation state and RNG. Search
    rolls back its branches; the winner is executed again on the real trajectory.
    The policy retains pending labels and optimizer state.
    """

    def __init__(self, executor: OnlineExecutor, dataset, catalog: ActionCatalog, stop: int,
                 *, ema_beta: float | None = None, terminal: bool = True) -> None:
        self.executor, self.dataset, self.catalog, self.stop = executor, dataset, catalog, stop
        self.terminal = terminal
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
            features, score = score_action(self.executor, self.dataset, index, action, steps=1, stop=self.stop if self.terminal else None)
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


def evaluate_greedy_oracle(data_directory: Path, output: Path, *, split: str = "test",
                           start_row: int | None = None, end_row: int | None = None,
                           ema_beta: float | None = None) -> dict:
    """Old one-step hindsight baselines, deliberately separate from beam search."""
    from .preparation import load_interval

    manifest, source, dataset, start, stop, state = load_interval(
        data_directory, split=split, start_row=start_row, end_row=end_row)
    beta = resolve_ema_beta(ema_beta if ema_beta is not None else
                            manifest["config"].get("evaluation", {}).get("ema_beta"),
                            manifest["config"].get("collection", {}).get("h_act", 1)
                            + manifest["config"].get("search", {}).get("h_tail", 128))
    catalog = ActionCatalog.from_dict(manifest["catalog"], source.method.capabilities)
    torch.set_num_threads(manifest["config"].get("threads", 1))
    output.mkdir(parents=True, exist_ok=False)
    plan = {"status": "evaluating", "kind": "greedy_oracle", "ema_beta": beta,
            "start": start, "stop": stop, "uses_future_labels_for_selection": True}
    save_json(output / "manifest.json", plan)
    summaries = []
    try:
        for name, policy_beta in (("oracle_greedy", None), ("oracle_ema", beta)):
            restore_rng(state)
            oracle = GreedyOracleExecutor(copy.deepcopy(source), dataset, catalog, stop, ema_beta=policy_beta,
                                           terminal=split != "history")
            summary = evaluate_policy(oracle, dataset, start, stop, output / name, name=name,
                                      target_mean=np.asarray(manifest["target_mean"]),
                                      target_std=np.asarray(manifest["target_std"]), terminal=split != "history")
            summary.update(candidate_search_seconds=oracle.candidate_search_seconds,
                           candidate_evaluations=oracle.candidate_evaluations)
            save_json(output / name / "summary.json", summary)
            summaries.append(summary)
        with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
            fields = [key for key, value in summaries[0].items() if not isinstance(value, list)]
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summaries)
        plan.update(status="complete", results=summaries)
    except BaseException as error:
        plan.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save_json(output / "manifest.json", plan)
    return plan

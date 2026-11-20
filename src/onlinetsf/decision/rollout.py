"""Isolated causal rollouts shared by search and teacher-data collection."""

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .catalog import ActionCatalog


def rng_state() -> tuple:
    return torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []


def restore_rng(state: tuple) -> None:
    torch.set_rng_state(state[0])
    if state[1]:
        torch.cuda.set_rng_state_all(state[1])


def step_dataset(executor, dataset, index: int, *, final: bool = False):
    context, target = dataset[index]
    values = dataset.decision_values
    decision_context = values[index:index + dataset.context_length] if values is not None else context
    return executor.step(index, context, target, decision_context=decision_context, final=final)


@dataclass
class SequencePolicy:
    catalog: ActionCatalog
    start: int
    indices: tuple[int, ...]

    def __call__(self, state):
        index = self.indices[state.index - self.start]
        self.last_selection = {"action_id": self.catalog.ids[index]}
        return self.catalog.actions[index]


@dataclass
class RecordingPolicy:
    action: object
    features: dict | None = None

    def __call__(self, state):
        if self.features is None:
            self.features = state.features
        return self.action


def advance(executor, dataset, start: int, stop: int, policy, random_state: tuple, *,
            terminal: int | None = None) -> tuple[np.ndarray, tuple]:
    """Advance an owned snapshot; restore the caller RNG even on failure.

    Only the real end of the experiment completes tail batches. Search and
    training-label boundaries are nonterminal unless they coincide with it.
    """
    saved = rng_state()
    restore_rng(random_state)
    executor.method.policy = policy
    errors = []
    try:
        for index in range(start, stop):
            final = terminal is not None and index == terminal - 1
            prediction = step_dataset(executor, dataset, index, final=final).emission.prediction.double()
            error = (prediction - dataset[index][1].double()).abs().mean().item()
            if not np.isfinite(error):
                raise FloatingPointError(f"nonfinite forecast at origin {index}")
            errors.append(error)
            executor._events.clear()
        return np.asarray(errors, dtype=np.float64), rng_state()
    finally:
        restore_rng(saved)


def load_sequence(directory: Path, catalog: ActionCatalog) -> tuple[dict, SequencePolicy]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or manifest.get("kind") != "beam_oracle":
        raise ValueError("teacher sequence search must complete before collection")
    if manifest["catalog"] != catalog.to_dict():
        raise ValueError("teacher and collection candidate catalogs differ")
    if hashlib.sha256((directory / "sequence.csv").read_bytes()).hexdigest() != manifest["sequence_sha256"]:
        raise ValueError("teacher sequence has changed since search")
    with (directory / "sequence.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if [int(row["origin"]) for row in rows] != list(range(manifest["start"], manifest["stop"])):
        raise ValueError("teacher sequence must cover consecutive forecast origins")
    return manifest, SequencePolicy(catalog, manifest["start"], tuple(catalog.ids.index(row["action_id"]) for row in rows))

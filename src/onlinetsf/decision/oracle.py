"""Persist and replay lookahead search sequences, independently of collection."""

import copy
import csv
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .beam_search import LookaheadBeamSearch
from .catalog import ActionCatalog
from .experiment import evaluate_policy
from .preparation import load_interval, save_json
from .rollout import SequencePolicy, restore_rng


def evaluate_oracle(data_directory: Path, output: Path, *, split: str = "test",
                    start_row: int | None = None, end_row: int | None = None,
                    search: dict | None = None) -> dict:
    manifest, source, dataset, start, stop, state = load_interval(
        data_directory, split=split, start_row=start_row, end_row=end_row)
    torch.set_num_threads(manifest["config"].get("threads", 1))
    catalog = ActionCatalog.from_dict(manifest["catalog"], source.method.capabilities)
    options = dict(manifest["config"].get("search", {}))
    if search is not None:
        options.update(search)
    # tail_policy controls label collection, not beam pruning.
    options.pop("tail_policy", None)
    engine = LookaheadBeamSearch(catalog, **options)
    output.mkdir(parents=True, exist_ok=False)
    plan = {"status": "searching", "kind": "beam_oracle", "prepared_data": str(data_directory.resolve()),
            "prepared_sha256": hashlib.sha256((data_directory / "manifest.json").read_bytes()).hexdigest(),
            "catalog": catalog.to_dict(), "search": options, "split": split,
            "start": start, "stop": stop, "target_start_row": start + dataset.context_length,
            "target_end_row": stop + dataset.context_length + dataset.horizon - 1,
            "uses_future_labels_for_selection": True,
            "terminal": split != "history"}
    save_json(output / "manifest.json", plan)
    torch.save({"executor": source, "rng": state}, output / "start.pt")
    try:
        began = perf_counter()
        result = engine.search(source, dataset, start, stop, state,
                               terminal=stop if plan["terminal"] else None)
        search_seconds = perf_counter() - began
        with (output / "sequence.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["origin", "action_id"])
            writer.writerows((start + offset, catalog.ids[index]) for offset, index in enumerate(result.actions))
        summaries = []
        for name, indices, expected in (("native", (catalog.native_indices[0],) * (stop - start), result.native_errors),
                                         ("oracle", result.actions, result.errors)):
            executor = copy.deepcopy(source)
            executor.method.policy = SequencePolicy(catalog, start, indices)
            restore_rng(state)
            summary = evaluate_policy(executor, dataset, start, stop, output / name, name=name,
                                      target_mean=np.asarray(manifest["target_mean"]),
                                      target_std=np.asarray(manifest["target_std"]), terminal=plan["terminal"])
            with (output / name / "forecasts.csv").open(encoding="utf-8", newline="") as handle:
                actual = np.asarray([float(row["mae"]) for row in csv.DictReader(handle)])
            if not np.allclose(actual, expected, rtol=1e-10, atol=1e-12):
                raise RuntimeError("persisted sequence differs from searched trajectory")
            summaries.append(summary)
        with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
            fields = [key for key, value in summaries[0].items() if not isinstance(value, list)]
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summaries)
        plan.update(status="complete", sweeps=result.sweeps, candidate_evaluations=result.candidate_evaluations,
                    search_seconds=search_seconds, results=summaries,
                    sequence_sha256=hashlib.sha256((output / "sequence.csv").read_bytes()).hexdigest())
    except BaseException as error:
        plan.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save_json(output / "manifest.json", plan)
    return plan

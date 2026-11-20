"""Counterfactual action-then-tail labels on a saved teacher trajectory."""

import copy
import csv
import hashlib
import io
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from .actions import positive_integer
from .beam_search import _executor_device
from .catalog import ActionCatalog
from .preparation import save_json
from .rollout import RecordingPolicy, advance, load_sequence


def _init_collection_worker(dataset, catalog: ActionCatalog, tail_policy, h_act: int, h_tail: int,
                            device: torch.device) -> None:
    global _collection_dataset, _collection_catalog, _collection_tail_policy, _collection_h_act, _collection_h_tail
    _collection_dataset = dataset
    _collection_catalog = catalog
    _collection_tail_policy = tail_policy
    _collection_h_act = h_act
    _collection_h_tail = h_tail
    torch.set_num_threads(1)
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _score_collection_job(job: tuple[bytes, int, int]) -> tuple[dict, dict]:
    snapshot, origin, action_index = job
    saved = torch.load(io.BytesIO(snapshot), weights_only=False)
    return score_candidate(saved["executor"], _collection_dataset, origin,
                           _collection_catalog.actions[action_index], _collection_tail_policy,
                           saved["rng"], h_act=_collection_h_act, h_tail=_collection_h_tail)


def score_candidate(source, dataset, origin: int, action, tail_policy, random_state: tuple, *,
                    h_act: int, h_tail: int) -> tuple[dict, dict]:
    branch = copy.deepcopy(source)
    probe = RecordingPolicy(action)
    steps, seconds = branch.method.gradient_steps, branch.method.update_seconds
    act, state = advance(branch, dataset, origin, origin + h_act, probe, random_state)
    tail, _ = advance(branch, dataset, origin + h_act, origin + h_act + h_tail, tail_policy, state)
    errors = np.concatenate((act, tail))
    return probe.features, {"mean_mae": float(errors.mean()), "act_mae": float(act.mean()),
                            "tail_mae": float(tail.mean()) if len(tail) else None,
                            "errors": errors, "gradient_steps": branch.method.gradient_steps - steps,
                            "update_seconds": branch.method.update_seconds - seconds}


def collect_experiment(data_directory: Path, oracle_directory: Path, output: Path, *,
                       config: dict | None = None) -> None:
    prepared = json.loads((data_directory / "manifest.json").read_text(encoding="utf-8"))
    if prepared["status"] != "complete" or prepared.get("kind") != "prepared":
        raise ValueError("prepare snapshots before collecting teacher labels")
    saved = torch.load(oracle_directory / "start.pt", weights_only=False)
    executor, state = saved["executor"], saved["rng"]
    catalog = ActionCatalog.from_dict(prepared["catalog"], executor.method.capabilities)
    teacher, policy = load_sequence(oracle_directory, catalog)
    if teacher["prepared_sha256"] != hashlib.sha256((data_directory / "manifest.json").read_bytes()).hexdigest():
        raise ValueError("teacher was searched from different prepared data")
    if teacher["target_end_row"] > prepared["train_end"] or teacher["terminal"]:
        raise ValueError("teacher labels must stay within history and use a nonterminal history search")
    options = copy.deepcopy(prepared["config"])
    options["search"] = {**options.get("search", {}), **teacher["search"]}
    if config is not None:
        for name in ("collection", "search", "selectors", "evaluation"):
            if name in config:
                options[name] = config[name]
    collection, search = options.get("collection", {}), options["search"]
    if "tau" in collection:
        raise ValueError("replace collection.tau with collection.h_act and search.h_tail")
    h_act, h_tail = collection.get("h_act", search.get("stride", 1)), search.get("h_tail", 128)
    positive_integer(h_act, "collection.h_act")
    if isinstance(h_tail, bool) or not isinstance(h_tail, int) or h_tail < 0:
        raise ValueError("search.h_tail must be a nonnegative integer")
    length = h_act + h_tail
    count = collection.get("train_states", 96)
    positive_integer(count, "collection.train_states")
    tail_mode = search.get("tail_policy", "native")
    if tail_mode not in {"native", "optimal_suffix"}:
        raise ValueError("tail_policy must be native or optimal_suffix")
    workers = search.get("workers", 1)
    positive_integer(workers, "search.workers")
    if (collection.get("feature_window", 32) != executor.method.features.feature_window
            or collection.get("error_window", 128) != executor.method.features.error_window):
        raise ValueError("feature windows must match the prepared snapshots; prepare again to change them")
    dataset = torch.load(data_directory / "online_start.pt", weights_only=False)["dataset"]
    stride = teacher["search"].get("stride", 1)
    first = max(teacher["start"], prepared["online_start"] + dataset.horizon)
    eligible = np.arange(first, teacher["stop"] - length + 1, stride)
    if len(eligible) < count:
        raise ValueError("too few teacher states with complete action and tail windows; reduce train_states or lookahead")
    origins = eligible[np.linspace(0, len(eligible) - 1, count, dtype=int)]
    requested = set(origins.tolist())
    tail_policy = policy if tail_mode == "optimal_suffix" else catalog.actions[catalog.native_indices[0]]
    output.mkdir(parents=True, exist_ok=False)
    manifest = {**prepared, "status": "collecting", "kind": "collection", "config": options,
                "prepared_data": Path(os.path.relpath(data_directory.resolve(), output.resolve())).as_posix(),
                "teacher": str(oracle_directory.resolve()),
                "teacher_sequence_sha256": teacher["sequence_sha256"], "h_act": h_act, "h_tail": h_tail,
                "tau": h_act, "tail_policy": tail_mode,
                "target": "mean action-then-tail MAE; regression uses J - J_native"}
    save_json(output / "manifest.json", manifest)
    torch.set_num_threads(options.get("threads", 1))
    states, maes, acts, tails, vectors, gradients, timings = [], [], [], [], [], [], []
    try:
        with (ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                  initializer=_init_collection_worker,
                                  initargs=(dataset, catalog, tail_policy, h_act, h_tail, _executor_device(executor)))
              if workers > 1 else nullcontext()) as pool, \
                (output / "states.csv").open("w", encoding="utf-8", newline="") as sf, \
                (output / "outcomes.csv").open("w", encoding="utf-8", newline="") as of:
            sw = csv.DictWriter(sf, fieldnames=["origin", *prepared["feature_names"]])
            ow = csv.DictWriter(of, fieldnames=["origin", "action_id", "h_act", "h_tail", "label_end",
                                                "mean_mae", "act_mae", "tail_mae", "gradient_steps", "update_seconds"])
            sw.writeheader()
            ow.writeheader()
            for origin in range(teacher["start"], int(origins[-1]) + 1):
                if origin in requested:
                    if pool is None:
                        results = [score_candidate(executor, dataset, origin, action, tail_policy, state,
                                                   h_act=h_act, h_tail=h_tail) for action in catalog.actions]
                    else:
                        snapshot = io.BytesIO()
                        torch.save({"executor": executor, "rng": state}, snapshot)
                        payload = snapshot.getvalue()
                        results = list(pool.map(_score_collection_job,
                                                [(payload, origin, i) for i in range(len(catalog.actions))]))
                    features = results[0][0]
                    states.append([features[name] for name in prepared["feature_names"]])
                    sw.writerow({"origin": origin, **{name: features[name] for name in prepared["feature_names"]}})
                    maes.append([row[1]["mean_mae"] for row in results])
                    acts.append([row[1]["act_mae"] for row in results])
                    tails.append([row[1]["tail_mae"] for row in results])
                    vectors.append([row[1]["errors"] for row in results])
                    gradients.append([row[1]["gradient_steps"] for row in results])
                    timings.append([row[1]["update_seconds"] for row in results])
                    for action_id, (_, outcome) in zip(catalog.ids, results):
                        ow.writerow({"origin": origin, "action_id": action_id, "h_act": h_act, "h_tail": h_tail,
                                     "label_end": origin + length + dataset.horizon - 1,
                                     **{key: value for key, value in outcome.items() if key != "errors"}})
                    sf.flush()
                    of.flush()
                    print(f"collected teacher origin={origin}, actions={len(catalog.ids)}", flush=True)
                _, state = advance(executor, dataset, origin, origin + 1, policy, state)
        values = np.asarray(maes)
        advantages = values - values[:, catalog.native_indices[:1]]
        np.savez(output / "training.npz", states=np.asarray(states, dtype=float), maes=values,
                 advantages=advantages, regrets=values - values.min(1, keepdims=True),
                 act_maes=np.asarray(acts), tail_maes=np.asarray(tails, dtype=float), errors=np.asarray(vectors),
                 gradient_steps=np.asarray(gradients), update_seconds=np.asarray(timings),
                 origins=origins, label_ends=origins + length + dataset.horizon - 1)
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save_json(output / "manifest.json", manifest)

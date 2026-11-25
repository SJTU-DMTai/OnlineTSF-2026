"""Counterfactual action-then-tail labels from winning search expansions or resampling."""

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
from time import perf_counter

import numpy as np
import torch

from .actions import positive_integer
from .beam_search import _executor_device, _search_copy
from .capabilities import UpdateCapabilities, UpdateUnit
from .catalog import ActionCatalog
from .preparation import save_json
from .profiling import SearchProfile
from .rollout import RecordingPolicy, SequencePolicy, advance, load_sequence
from .search_schedule import resolve_search_schedule


def anchor_teacher_costs(raw_maes: np.ndarray, teacher_actions: np.ndarray,
                         epsilon: float) -> tuple[np.ndarray, int]:
    """Clamp non-teacher action costs to at least J(teacher) + epsilon."""
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("collection.anchor_epsilon must be a finite nonnegative number")
    raw_maes = np.asarray(raw_maes, dtype=np.float64)
    teacher_actions = np.asarray(teacher_actions, dtype=np.int64)
    if raw_maes.ndim != 2 or teacher_actions.shape != (len(raw_maes),):
        raise ValueError("teacher_actions must provide one action index per state")
    rows = np.arange(len(raw_maes))
    teacher_costs = raw_maes[rows, teacher_actions]
    floor = teacher_costs[:, None] + float(epsilon)
    anchored = np.maximum(raw_maes, floor)
    anchored[rows, teacher_actions] = teacher_costs
    non_teacher = np.ones_like(raw_maes, dtype=bool)
    non_teacher[rows, teacher_actions] = False
    clipped = int(np.sum(non_teacher & (raw_maes < floor)))
    return anchored, clipped


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


def _init_fill_worker(dataset, catalog: ActionCatalog, tail_policy, h_act: int, h_tail: int,
                      device: torch.device) -> None:
    global _fill_dataset, _fill_catalog, _fill_tail_policy, _fill_h_act, _fill_h_tail
    _fill_dataset = dataset
    _fill_catalog = catalog
    _fill_tail_policy = tail_policy
    _fill_h_act = h_act
    _fill_h_tail = h_tail
    torch.set_num_threads(1)
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _score_missing_actions(source, dataset, origin: int, action_indices: tuple[int, ...], catalog: ActionCatalog,
                           tail_policy, random_state: tuple, *,
                           h_act: int, h_tail: int) -> dict[int, tuple[float, float, float]]:
    """Evaluate only pruned actions on a lightweight search copy (states are already cached)."""
    scores: dict[int, tuple[float, float, float]] = {}
    end = origin + h_act
    tail_end = end + h_tail
    for action_index in action_indices:
        branch = copy.deepcopy(source)
        act, state = advance(branch, dataset, origin, end, catalog.actions[action_index], random_state)
        act_sum = float(act.sum())
        if h_tail > 0:
            tail, _ = advance(branch, dataset, end, tail_end, tail_policy, state)
            tail_sum = float(tail.sum())
            scores[action_index] = ((act_sum + tail_sum) / (h_act + h_tail), act_sum / h_act, tail_sum / h_tail)
        else:
            act_mae = act_sum / h_act
            scores[action_index] = (act_mae, act_mae, float("nan"))
    return scores


def _fill_missing_job(job: tuple[bytes, int, tuple[int, ...], bool]) -> tuple[dict[int, tuple[float, float, float]], dict]:
    snapshot, origin, action_indices, profiling = job
    began = perf_counter() if profiling else 0.
    saved = torch.load(io.BytesIO(snapshot), weights_only=False)
    load_seconds = perf_counter() - began if profiling else 0.
    scores = _score_missing_actions(saved["executor"], _fill_dataset, origin, action_indices,
                                    _fill_catalog, _fill_tail_policy, saved["rng"],
                                    h_act=_fill_h_act, h_tail=_fill_h_tail)
    timings = ({"pid": os.getpid(), "actions": len(action_indices), "load_seconds": load_seconds,
                "job_seconds": perf_counter() - began, "input_bytes": len(snapshot)}
               if profiling else {})
    return scores, timings


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


_UNSET = object()


def _merge_collection_options(prepared: dict, teacher: dict, config: dict | None, *,
                              mode: str | None = None, anchor_epsilon=_UNSET,
                              workers: int | None = None) -> dict:
    options = copy.deepcopy(prepared["config"])
    prepared_collection = prepared["config"].get("collection", {})
    options["search"] = {**teacher["search"],
                         "tail_policy": options.get("search", {}).get("tail_policy", "native")}
    if config is not None:
        for name in ("collection", "selectors", "evaluation"):
            if name in config:
                options[name] = copy.deepcopy(config[name])
        if "collection" in config:
            for key in ("feature_window", "error_window"):
                if key not in options["collection"] and key in prepared_collection:
                    options["collection"][key] = prepared_collection[key]
        if "search" in config:
            merged_search = dict(options["search"])
            override = config["search"]
            if "k" in override:
                merged_search.pop("k_max", None)
                merged_search.pop("k_min", None)
            elif "k_max" in override or "k_min" in override:
                merged_search.pop("k", None)
            merged_search.update(override)
            options["search"] = merged_search
    if mode is not None:
        options.setdefault("collection", {})["mode"] = mode
        if mode == "reuse" and (config is None or "collection" not in config):
            options["collection"].pop("h_act", None)
            options["collection"].pop("h_tail", None)
    if anchor_epsilon is not _UNSET:
        options.setdefault("collection", {})["anchor_epsilon"] = anchor_epsilon
    if workers is not None:
        options["search"]["workers"] = workers
    return options


def _resolve_collection_mode(collection: dict, search: dict, teacher: dict, oracle_directory: Path) -> str:
    mode = collection.get("mode")
    if mode is not None:
        if mode not in {"reuse", "resample"}:
            raise ValueError("collection.mode must be 'reuse' or 'resample'")
        return mode
    expansions_file = teacher.get("expansions_file")
    if not expansions_file or not (oracle_directory / expansions_file).exists():
        return "resample"
    if "winning_k" not in teacher or "winning_h_tail" not in teacher:
        return "resample"
    if search.get("tail_policy", "native") == "optimal_suffix":
        return "resample"
    if "h_act" in collection and collection["h_act"] != teacher["winning_k"]:
        return "resample"
    if "h_tail" in collection and collection["h_tail"] != teacher["winning_h_tail"]:
        return "resample"
    return "reuse"


def _collect_from_expansions(data_directory: Path, oracle_directory: Path, output: Path, *,
                             prepared: dict, teacher: dict, catalog: ActionCatalog,
                             policy: SequencePolicy, options: dict, profile: bool = False) -> None:
    load_began = perf_counter()
    collection, search = options.get("collection", {}), options["search"]
    expansions_file = teacher.get("expansions_file")
    expansions_path = oracle_directory / expansions_file if expansions_file else None
    if expansions_path is None or not expansions_path.exists():
        raise ValueError("teacher directory does not contain expansions.npz; rerun oracle or use collection.mode='resample'")
    if "expansions_sha256" in teacher and hashlib.sha256(expansions_path.read_bytes()).hexdigest() != teacher["expansions_sha256"]:
        raise ValueError("teacher expansions have changed since search")
    h_act = int(teacher["winning_k"])
    h_tail = int(teacher["winning_h_tail"])
    if "h_act" in collection and collection["h_act"] is not None and collection["h_act"] != h_act:
        raise ValueError("collection.mode='reuse' requires h_act to match the winning search round; omit it or use mode='resample'")
    if "h_tail" in collection and collection["h_tail"] is not None and collection["h_tail"] != h_tail:
        raise ValueError("collection.mode='reuse' requires h_tail to match the winning search round; omit it or use mode='resample'")
    length = h_act + h_tail
    anchor_epsilon = collection.get("anchor_epsilon", 0.0)
    if anchor_epsilon is not None:
        if isinstance(anchor_epsilon, bool) or not isinstance(anchor_epsilon, (int, float)) or not np.isfinite(anchor_epsilon) or anchor_epsilon < 0:
            raise ValueError("collection.anchor_epsilon must be null or a finite nonnegative number")
        anchor_epsilon = float(anchor_epsilon)
    workers = search.get("workers", 1)
    positive_integer(workers, "search.workers")
    prepared_collection = prepared["config"].get("collection", {})
    if (collection.get("feature_window", prepared_collection.get("feature_window", 32)) != prepared_collection.get("feature_window", 32)
            or collection.get("error_window", prepared_collection.get("error_window", 128)) != prepared_collection.get("error_window", 128)):
        raise ValueError("feature windows must match the prepared snapshots; prepare again to change them")
    horizon = int(prepared["config"]["data"]["horizon"])
    first = max(teacher["start"], prepared["online_start"] + horizon)
    with np.load(expansions_path) as data:
        origins_all = np.asarray(data["origins"], dtype=np.int64)
        ends_all = np.asarray(data["ends"], dtype=np.int64)
        tail_ends_all = np.asarray(data["tail_ends"], dtype=np.int64)
        teacher_actions_all = np.asarray(data["teacher_actions"], dtype=np.int64)
        states_all = np.asarray(data["states"], dtype=np.float64)
        mean_maes_all = np.asarray(data["mean_maes"], dtype=np.float64)
        act_maes_all = np.asarray(data["act_maes"], dtype=np.float64)
        tail_maes_all = np.asarray(data["tail_maes"], dtype=np.float64)
        evaluated_all = np.asarray(data["evaluated"], dtype=bool)
        reference_actions = tuple(int(v) for v in data["reference_actions"].tolist())
    valid_mask = (origins_all >= first) & ((ends_all - origins_all) == h_act) & ((tail_ends_all - ends_all) == h_tail)
    valid_indices = np.flatnonzero(valid_mask)
    if not len(valid_indices):
        raise ValueError("no complete winning-round segments available for training; reduce search lookahead or interval")
    requested_count = collection.get("train_states")
    if requested_count is None or requested_count == "all":
        selected = valid_indices
    else:
        positive_integer(requested_count, "collection.train_states")
        if len(valid_indices) < requested_count:
            raise ValueError("too few complete winning-round segments; reduce train_states or lookahead")
        selected = valid_indices[np.linspace(0, len(valid_indices) - 1, requested_count, dtype=int)]
    origins = origins_all[selected]
    teacher_actions = teacher_actions_all[selected]
    states = states_all[selected]
    raw_maes = mean_maes_all[selected].copy()
    act_maes = act_maes_all[selected].copy()
    tail_maes = tail_maes_all[selected].copy()
    reused_mask = evaluated_all[selected].copy()
    label_ends = origins + length + horizon - 1
    load_seconds = perf_counter() - load_began
    reused_count = int(reused_mask.sum())
    total_count = int(reused_mask.size)
    filled_count = total_count - reused_count
    output.mkdir(parents=True, exist_ok=False)
    manifest = {**prepared, "status": "collecting", "kind": "collection", "config": options,
                "collection_mode": "reuse",
                "prepared_data": Path(os.path.relpath(data_directory.resolve(), output.resolve())).as_posix(),
                "teacher": str(oracle_directory.resolve()),
                "teacher_sequence_sha256": teacher["sequence_sha256"],
                "winning_round": teacher["winning_round"],
                "winning_reference_round": teacher.get("winning_reference_round", 0),
                "h_act": h_act, "h_tail": h_tail, "tau": h_act,
                "tail_policy": "winning_reference",
                "anchor_epsilon": anchor_epsilon,
                "target": "teacher-anchored action-then-tail MAE; regression uses J - J_native"
                if anchor_epsilon is not None else "mean action-then-tail MAE; regression uses J - J_native"}
    save_json(output / "manifest.json", manifest)
    recorder = SearchProfile(output / "profile.jsonl") if profile else None
    torch.set_num_threads(options.get("threads", 1))
    try:
        if recorder is not None:
            recorder.record("metadata", mode="reuse", winning_round=teacher["winning_round"],
                            h_act=h_act, h_tail=h_tail, workers=workers, state_count=len(origins),
                            action_count=len(catalog.ids), reused_candidates=reused_count,
                            filled_candidates=filled_count, total_candidates=total_count,
                            reuse_fraction=(reused_count / total_count) if total_count else 1.0,
                            load_expansions_seconds=load_seconds,
                            timing="host wall clock; no added CUDA synchronization; worker times overlap")
            recorder.record("phase_start", phase="fill_missing")
        fill_began = perf_counter() if recorder is not None else 0.
        load_snapshot_seconds = advance_seconds = eval_seconds = prepare_jobs_seconds = dispatch_wait_seconds = 0.
        snapshot_bytes = 0
        missing_by_origin: dict[int, tuple[int, tuple[int, ...]]] = {}
        for row_idx, origin in enumerate(origins.tolist()):
            missing = tuple(int(a) for a in np.flatnonzero(~reused_mask[row_idx]))
            if missing:
                missing_by_origin[origin] = (row_idx, missing)
        if missing_by_origin:
            snap_began = perf_counter() if recorder is not None else 0.
            saved = torch.load(oracle_directory / "start.pt", weights_only=False)
            executor, state = _search_copy(saved["executor"]), saved["rng"]
            dataset = torch.load(data_directory / "online_start.pt", weights_only=False)["dataset"]
            if recorder is not None:
                load_snapshot_seconds = perf_counter() - snap_began
            ref_tail_policy = SequencePolicy(catalog, teacher["start"], reference_actions)
            max_origin = max(missing_by_origin)
            use_pool = workers > 1 and any(len(actions) > 1 for _, actions in missing_by_origin.values())
            fill_step = 0
            with (ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                      initializer=_init_fill_worker,
                                      initargs=(dataset, catalog, ref_tail_policy, h_act, h_tail,
                                                _executor_device(executor)))
                  if use_pool else nullcontext()) as pool:
                for origin in range(teacher["start"], max_origin + 1):
                    if origin in missing_by_origin:
                        fill_step += 1
                        row_idx, missing_actions = missing_by_origin[origin]
                        origin_began = perf_counter() if recorder is not None else 0.
                        worker_jobs = []
                        if pool is None or len(missing_actions) == 1:
                            filled = _score_missing_actions(executor, dataset, origin, missing_actions,
                                                            catalog, ref_tail_policy, state,
                                                            h_act=h_act, h_tail=h_tail)
                            if recorder is not None:
                                eval_seconds += perf_counter() - origin_began
                        else:
                            prep_began = perf_counter() if recorder is not None else 0.
                            buffer = io.BytesIO()
                            torch.save({"executor": executor, "rng": state}, buffer)
                            payload = buffer.getvalue()
                            groups = min(len(missing_actions), workers)
                            chunk_size = (len(missing_actions) + groups - 1) // groups
                            jobs = [(payload, origin, missing_actions[offset:offset + chunk_size], recorder is not None)
                                    for offset in range(0, len(missing_actions), chunk_size)]
                            if recorder is not None:
                                snapshot_bytes += len(payload)
                                prepare_jobs_seconds += perf_counter() - prep_began
                                wait_began = perf_counter()
                            filled = {}
                            for chunk, timings in pool.map(_fill_missing_job, jobs):
                                filled.update(chunk)
                                if recorder is not None:
                                    worker_jobs.append(timings)
                            if recorder is not None:
                                wait_elapsed = perf_counter() - wait_began
                                dispatch_wait_seconds += wait_elapsed
                                eval_seconds += perf_counter() - origin_began
                        if recorder is not None:
                            recorder.record("fill_origin", step=fill_step, origin=origin,
                                            missing_count=len(missing_actions),
                                            wall_seconds=perf_counter() - origin_began,
                                            worker_jobs=worker_jobs)
                        for action_idx, (m_mae, a_mae, t_mae) in filled.items():
                            raw_maes[row_idx, action_idx] = m_mae
                            act_maes[row_idx, action_idx] = a_mae
                            tail_maes[row_idx, action_idx] = t_mae
                        print(f"reused origin={origin}, reused={len(catalog.ids) - len(missing_actions)}/{len(catalog.ids)}, "
                              f"filled={len(missing_actions)}", flush=True)
                    if origin < max_origin:
                        adv_began = perf_counter() if recorder is not None else 0.
                        _, state = advance(executor, dataset, origin, origin + 1, policy, state)
                        if recorder is not None:
                            advance_seconds += perf_counter() - adv_began
        if recorder is not None:
            recorder.record("phase_end", phase="fill_missing", wall_seconds=perf_counter() - fill_began,
                            origins_filled=len(missing_by_origin), filled_candidates=filled_count,
                            load_snapshot_seconds=load_snapshot_seconds, advance_seconds=advance_seconds,
                            eval_seconds=eval_seconds, prepare_jobs_seconds=prepare_jobs_seconds,
                            dispatch_wait_seconds=dispatch_wait_seconds, snapshot_bytes=snapshot_bytes)
            recorder.record("phase_start", phase="anchor_and_write")
        write_began = perf_counter() if recorder is not None else 0.
        if not np.isfinite(raw_maes).all():
            raise RuntimeError("incomplete action scores after expansion reuse")
        if anchor_epsilon is not None:
            values, anchored_entries = anchor_teacher_costs(raw_maes, teacher_actions, anchor_epsilon)
        else:
            values, anchored_entries = raw_maes.copy(), 0
        advantages = values - values[:, catalog.native_indices[:1]]
        regrets = values - values.min(1, keepdims=True)
        with (output / "states.csv").open("w", encoding="utf-8", newline="") as sf, \
                (output / "outcomes.csv").open("w", encoding="utf-8", newline="") as of:
            sw = csv.DictWriter(sf, fieldnames=["origin", *prepared["feature_names"]])
            ow = csv.DictWriter(of, fieldnames=["origin", "action_id", "h_act", "h_tail", "label_end",
                                                "mean_mae", "raw_mean_mae", "act_mae", "tail_mae",
                                                "teacher_action_id", "reused"])
            sw.writeheader()
            ow.writeheader()
            for row_idx, origin in enumerate(origins.tolist()):
                sw.writerow({"origin": origin, **{
                    name: ("" if not np.isfinite(states[row_idx, feat_idx]) else float(states[row_idx, feat_idx]))
                    for feat_idx, name in enumerate(prepared["feature_names"])}})
                teacher_id = catalog.ids[int(teacher_actions[row_idx])]
                for action_idx, action_id in enumerate(catalog.ids):
                    t_val = tail_maes[row_idx, action_idx]
                    ow.writerow({
                        "origin": origin, "action_id": action_id, "h_act": h_act, "h_tail": h_tail,
                        "label_end": int(label_ends[row_idx]),
                        "mean_mae": float(values[row_idx, action_idx]),
                        "raw_mean_mae": float(raw_maes[row_idx, action_idx]),
                        "act_mae": float(act_maes[row_idx, action_idx]),
                        "tail_mae": (float(t_val) if np.isfinite(t_val) else None),
                        "teacher_action_id": teacher_id,
                        "reused": bool(reused_mask[row_idx, action_idx]),
                    })
        np.savez(output / "training.npz", states=states, maes=values, raw_maes=raw_maes,
                 advantages=advantages, regrets=regrets, act_maes=act_maes, tail_maes=tail_maes,
                 teacher_actions=teacher_actions, reused_mask=reused_mask,
                 origins=origins, label_ends=label_ends)
        if recorder is not None:
            recorder.record("phase_end", phase="anchor_and_write",
                            wall_seconds=perf_counter() - write_began, anchored_entries=anchored_entries)
        manifest.update(status="complete", reused_candidates=reused_count,
                        filled_candidates=filled_count,
                        total_candidates=total_count,
                        reuse_fraction=(reused_count / total_count) if total_count else 1.0,
                        anchored_entries=anchored_entries,
                        state_count=len(origins))
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save_json(output / "manifest.json", manifest)
        if recorder is not None:
            recorder.record("run_end", status=manifest["status"], error=manifest.get("error"))
            recorder.close()


def collect_experiment(data_directory: Path, oracle_directory: Path, output: Path, *,
                       config: dict | None = None, mode: str | None = None,
                       anchor_epsilon=_UNSET, workers: int | None = None,
                       profile: bool = False) -> None:
    prepared = json.loads((data_directory / "manifest.json").read_text(encoding="utf-8"))
    if prepared["status"] != "complete" or prepared.get("kind") != "prepared":
        raise ValueError("prepare snapshots before collecting teacher labels")
    capabilities = UpdateCapabilities(tuple(UpdateUnit(**row) for row in prepared["capabilities"]))
    catalog = ActionCatalog.from_dict(prepared["catalog"], capabilities)
    teacher, policy = load_sequence(oracle_directory, catalog)
    if teacher["prepared_sha256"] != hashlib.sha256((data_directory / "manifest.json").read_bytes()).hexdigest():
        raise ValueError("teacher was searched from different prepared data")
    if teacher["target_end_row"] > prepared["train_end"] or teacher["terminal"]:
        raise ValueError("teacher labels must stay within history and use a nonterminal history search")
    options = _merge_collection_options(prepared, teacher, config,
                                        mode=mode, anchor_epsilon=anchor_epsilon, workers=workers)
    collection, search = options.get("collection", {}), options["search"]
    if "tau" in collection:
        raise ValueError("replace collection.tau with collection.h_act and collection.h_tail")
    mode = _resolve_collection_mode(collection, search, teacher, oracle_directory)
    if mode == "reuse":
        _collect_from_expansions(data_directory, oracle_directory, output,
                                 prepared=prepared, teacher=teacher, catalog=catalog,
                                 policy=policy, options=options, profile=profile)
        return
    load_began = perf_counter() if profile else 0.
    saved = torch.load(oracle_directory / "start.pt", weights_only=False)
    executor, state = saved["executor"], saved["rng"]
    schedule = resolve_search_schedule(**{name: search[name] for name in ("k", "k_max", "k_min", "h_tail") if name in search})
    finest_k, finest_tail = schedule[-1]
    h_act, h_tail = collection.get("h_act", finest_k), collection.get("h_tail", finest_tail)
    positive_integer(h_act, "collection.h_act")
    if isinstance(h_tail, bool) or not isinstance(h_tail, int) or h_tail < 0:
        raise ValueError("collection.h_tail must be a nonnegative integer")
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
    load_seconds = perf_counter() - load_began if profile else 0.
    stride = teacher["search"]["k"][-1]
    first = max(teacher["start"], prepared["online_start"] + dataset.horizon)
    eligible = np.arange(first, teacher["stop"] - length + 1, stride)
    if len(eligible) < count:
        raise ValueError("too few teacher states with complete action and tail windows; reduce train_states or lookahead")
    origins = eligible[np.linspace(0, len(eligible) - 1, count, dtype=int)]
    requested = set(origins.tolist())
    tail_policy = policy if tail_mode == "optimal_suffix" else catalog.actions[catalog.native_indices[0]]
    output.mkdir(parents=True, exist_ok=False)
    manifest = {**prepared, "status": "collecting", "kind": "collection", "config": options,
                "collection_mode": "resample",
                "prepared_data": Path(os.path.relpath(data_directory.resolve(), output.resolve())).as_posix(),
                "teacher": str(oracle_directory.resolve()),
                "teacher_sequence_sha256": teacher["sequence_sha256"], "h_act": h_act, "h_tail": h_tail,
                "tau": h_act, "tail_policy": tail_mode,
                "target": "mean action-then-tail MAE; regression uses J - J_native"}
    save_json(output / "manifest.json", manifest)
    recorder = SearchProfile(output / "profile.jsonl") if profile else None
    torch.set_num_threads(options.get("threads", 1))
    states, maes, acts, tails, vectors, gradients, timings = [], [], [], [], [], [], []
    try:
        if recorder is not None:
            recorder.record("metadata", mode="resample", h_act=h_act, h_tail=h_tail, tail_policy=tail_mode,
                            workers=workers, state_count=len(origins), action_count=len(catalog.ids),
                            load_snapshots_seconds=load_seconds,
                            timing="host wall clock; no added CUDA synchronization; worker times overlap")
            recorder.record("phase_start", phase="resample")
        resample_began = perf_counter() if recorder is not None else 0.
        advance_seconds = eval_seconds = 0.
        collect_step = 0
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
                    collect_step += 1
                    origin_began = perf_counter() if recorder is not None else 0.
                    if pool is None:
                        results = [score_candidate(executor, dataset, origin, action, tail_policy, state,
                                                   h_act=h_act, h_tail=h_tail) for action in catalog.actions]
                    else:
                        snapshot = io.BytesIO()
                        torch.save({"executor": executor, "rng": state}, snapshot)
                        payload = snapshot.getvalue()
                        results = list(pool.map(_score_collection_job,
                                                [(payload, origin, i) for i in range(len(catalog.actions))]))
                    if recorder is not None:
                        origin_elapsed = perf_counter() - origin_began
                        eval_seconds += origin_elapsed
                        recorder.record("collect_origin", step=collect_step, origin=origin,
                                        action_count=len(catalog.ids), wall_seconds=origin_elapsed)
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
                adv_began = perf_counter() if recorder is not None else 0.
                _, state = advance(executor, dataset, origin, origin + 1, policy, state)
                if recorder is not None:
                    advance_seconds += perf_counter() - adv_began
        if recorder is not None:
            recorder.record("phase_end", phase="resample", wall_seconds=perf_counter() - resample_began,
                            advance_seconds=advance_seconds, eval_seconds=eval_seconds,
                            state_count=len(origins))
            recorder.record("phase_start", phase="write_outputs")
        write_began = perf_counter() if recorder is not None else 0.
        values = np.asarray(maes)
        advantages = values - values[:, catalog.native_indices[:1]]
        np.savez(output / "training.npz", states=np.asarray(states, dtype=float), maes=values,
                 advantages=advantages, regrets=values - values.min(1, keepdims=True),
                 act_maes=np.asarray(acts), tail_maes=np.asarray(tails, dtype=float), errors=np.asarray(vectors),
                 gradient_steps=np.asarray(gradients), update_seconds=np.asarray(timings),
                 origins=origins, label_ends=origins + length + dataset.horizon - 1)
        if recorder is not None:
            recorder.record("phase_end", phase="write_outputs", wall_seconds=perf_counter() - write_began)
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save_json(output / "manifest.json", manifest)
        if recorder is not None:
            recorder.record("run_end", status=manifest["status"], error=manifest.get("error"))
            recorder.close()

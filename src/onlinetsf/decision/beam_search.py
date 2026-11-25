"""Coarse-to-fine lookahead beam search over independently owned snapshots."""

import copy
import io
import multiprocessing
import os
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from time import perf_counter

import numpy as np
import torch

from .actions import positive_integer
from .catalog import ActionCatalog
from .profiling import SearchProfile
from .rollout import SequencePolicy, advance
from .search_schedule import resolve_search_schedule


@dataclass(frozen=True)
class SegmentExpansion:
    origin: int
    end: int
    tail_end: int
    action_index: int
    # Action index -> (mean_mae, act_mae, tail_mae) for candidates that finished
    # evaluation at this parent state. Pruned actions are omitted.
    scores: dict[int, tuple[float, float, float]]


@dataclass
class BeamNode:
    executor: object
    random_state: tuple
    total: float
    actions: tuple[int, ...]
    errors: tuple[np.ndarray, ...] = ()
    expansions: tuple[SegmentExpansion, ...] = ()


@dataclass
class SearchResult:
    actions: tuple[int, ...]
    native_errors: np.ndarray
    errors: np.ndarray
    rounds: list[dict]
    winning_round: int
    candidate_evaluations: int
    winning_k: int = 1
    winning_h_tail: int = 0
    winning_reference_round: int = 0
    winning_reference_actions: tuple[int, ...] = ()
    winning_expansions: tuple[SegmentExpansion, ...] = ()


def _executor_device(executor) -> torch.device:
    inner = getattr(getattr(executor, "method", None), "method", None)
    return getattr(inner, "device", torch.device("cpu"))


def _init_search_worker(dataset, catalog: ActionCatalog, device: torch.device, terminal: int | None,
                        shared_scores) -> None:
    global _search_dataset, _search_catalog, _search_terminal, _search_scores
    _search_dataset = dataset
    _search_catalog = catalog
    _search_terminal = terminal
    _search_scores = shared_scores
    torch.set_num_threads(1)
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _search_copy(source):
    """Keep search instrumentation off its owned snapshots, not the caller's."""
    branch = copy.deepcopy(source)
    branch.record_events = False
    branch.method.features = None
    branch.method.measure_updates = False
    return branch


def _expand_actions(node: BeamNode, dataset, catalog: ActionCatalog, origin: int, end: int,
                    tail_end: int, actions: tuple[int, ...], tail_policy, beam_width: int, *,
                    terminal: int | None, score_limit: float | None = None,
                    shared_scores=None, metrics: dict | None = None,
                    recorded_scores: dict[int, tuple[float, float, float]] | None = None) -> list[tuple[float, BeamNode]]:
    """Keep local top-B; a discarded local candidate cannot be global top-B.

    Bounds use completed candidates only and strict > comparisons. Workers read
    the global bound once per candidate; a stale, higher bound only prunes less.
    Stable action order resolves exact ties just as in the serial search.
    """
    candidates = []
    act_len = end - origin
    tail_len = tail_end - end
    for action_index in actions:
        stats = metrics.setdefault(catalog.ids[action_index], Counter()) if metrics is not None else None
        if stats is not None:
            stats["candidates"] += 1
            stats["planned_action_origins"] += act_len
            stats["planned_tail_origins"] += tail_len
        limit = score_limit
        if len(candidates) == beam_width:
            limit = candidates[-1][0] if limit is None else min(limit, candidates[-1][0])
        if shared_scores is not None:
            began = perf_counter() if stats is not None else 0.
            with shared_scores.get_lock():
                global_limit = shared_scores[-1]
            if stats is not None:
                stats["bound_seconds"] += perf_counter() - began
            if global_limit < np.inf:
                limit = global_limit if limit is None else min(limit, global_limit)
        began = perf_counter() if stats is not None else 0.
        branch = copy.deepcopy(node.executor)
        if stats is not None:
            stats["clone_seconds"] += perf_counter() - began
            began = perf_counter()
        errors, state = advance(branch, dataset, origin, end, catalog.actions[action_index], node.random_state,
                                terminal=terminal, error_limit=limit, error_offset=node.total)
        if stats is not None:
            stats["action_seconds"] += perf_counter() - began
            stats["action_origins"] += len(errors)
        if len(errors) != act_len:
            continue
        act_sum = float(errors.sum())
        total = node.total + act_sum
        tail_total = 0.
        if tail_len == 0:
            if recorded_scores is not None:
                act_mae = act_sum / act_len
                recorded_scores[action_index] = (act_mae, act_mae, float("nan"))
            if stats is not None:
                stats["recorded_candidates"] += 1
            if limit is not None and total > limit:
                continue
        else:
            if limit is not None and total > limit:
                continue
            began = perf_counter() if stats is not None else 0.
            tail = copy.deepcopy(branch)
            if stats is not None:
                stats["tail_clone_seconds"] += perf_counter() - began
                began = perf_counter()
            tail_errors, _ = advance(tail, dataset, end, tail_end, tail_policy, state, terminal=terminal,
                                     error_limit=limit, error_offset=total)
            if stats is not None:
                stats["tail_seconds"] += perf_counter() - began
                stats["tail_origins"] += len(tail_errors)
            del tail
            if len(tail_errors) != tail_len:
                continue
            tail_total = float(tail_errors.sum())
            if recorded_scores is not None:
                recorded_scores[action_index] = (
                    (act_sum + tail_total) / (act_len + tail_len),
                    act_sum / act_len,
                    tail_total / tail_len,
                )
            if stats is not None:
                stats["recorded_candidates"] += 1
        score = total + tail_total
        if limit is not None and score > limit:
            continue
        if shared_scores is not None:
            # Only complete action-plus-tail scores can establish an upper bound.
            # Keep the lock out of model execution, cloning and serialization.
            began = perf_counter() if stats is not None else 0.
            with shared_scores.get_lock():
                shared_scores[:] = sorted([*shared_scores, score])[:beam_width]
            if stats is not None:
                stats["bound_seconds"] += perf_counter() - began
        if stats is not None:
            stats["completed_candidates"] += 1
        child = BeamNode(branch, state, total, node.actions + (action_index,) * act_len,
                         (*node.errors, errors), node.expansions)
        candidates.append((score, child))
        candidates.sort(key=lambda item: item[0])
        del candidates[beam_width:]
    return candidates


def _score_search_job(job: tuple) -> tuple:
    snapshot, actions, origin, end, tail_end, tail_actions, total, beam_width, profiling = job
    began = perf_counter() if profiling else 0.
    saved = torch.load(io.BytesIO(snapshot), weights_only=False)
    timings = {"load_seconds": perf_counter() - began} if profiling else {}
    metrics = {} if profiling else None
    recorded_scores: dict[int, tuple[float, float, float]] = {}
    node = BeamNode(saved["executor"], saved["random_state"], total, ())
    tail_policy = SequencePolicy(_search_catalog, end, tail_actions)
    candidates = _expand_actions(node, _search_dataset, _search_catalog, origin, end, tail_end,
                                 actions, tail_policy, beam_width, terminal=_search_terminal,
                                 shared_scores=_search_scores, metrics=metrics,
                                 recorded_scores=recorded_scores)
    results = []
    save_began = perf_counter() if profiling else 0.
    for score, child in candidates:
        buffer = io.BytesIO()
        torch.save({"executor": child.executor, "random_state": child.random_state}, buffer)
        results.append((score, child.actions[0], child.errors[0], buffer.getvalue()))
    if profiling:
        timings.update(pid=os.getpid(), save_seconds=perf_counter() - save_began,
                       job_seconds=perf_counter() - began, input_bytes=len(snapshot),
                       output_bytes=sum(len(row[3]) for row in results))
    return results, recorded_scores, metrics, timings


class CoarseToFineBeamSearch:
    def __init__(self, catalog: ActionCatalog, *, k: list[int] | tuple[int, ...] | None = None,
                 k_max: int | None = None, k_min: int | None = None, beam_width: int = 4,
                 h_tail: int | list[int] | tuple[int, ...] = 128, workers: int = 1) -> None:
        for name, value in (("beam_width", beam_width), ("workers", workers)):
            positive_integer(value, name)
        self.schedule = resolve_search_schedule(k=k, k_max=k_max, k_min=k_min, h_tail=h_tail)
        self.catalog, self.beam_width, self.workers = catalog, beam_width, workers

    def search(self, source, dataset, start: int, stop: int, random_state: tuple, *,
               terminal: int | None = None, native_errors: np.ndarray | None = None,
               profile: SearchProfile | None = None) -> SearchResult:
        if not 0 <= start < stop <= len(dataset) or source._last_index not in (None, start - 1):
            raise ValueError("search needs a consecutive source snapshot and a nonempty legal interval")
        native = self.catalog.native_indices[0]
        incumbent = (native,) * (stop - start)
        if native_errors is None:
            baseline = _search_copy(source)
            native_errors, _ = advance(baseline, dataset, start, stop,
                                       SequencePolicy(self.catalog, start, incumbent), random_state, terminal=terminal)
            del baseline
        else:
            native_errors = np.asarray(native_errors, dtype=np.float64)
            if native_errors.shape != (stop - start,) or not np.isfinite(native_errors).all():
                raise ValueError("native_errors must be a finite 1D array matching the search interval")
        best_errors = native_errors
        reports, evaluations, winning_round = [], 0, 0
        winning_k, winning_h_tail = self.schedule[-1]
        winning_reference_round = 0
        winning_reference_actions = incumbent
        winning_expansions: tuple[SegmentExpansion, ...] = ()
        finest_native_expansions: list[SegmentExpansion] = []
        device = _executor_device(source)
        order = [native] + [i for i in range(len(self.catalog.ids)) if i != native]
        action_ranks = {index: rank for rank, index in enumerate(order)}
        # The complete incumbent is protected outside the finite beam. This
        # preserves Native even for B=1 without occupying its only search slot.
        context = multiprocessing.get_context("spawn")
        shared_scores = context.Array("d", [np.inf] * self.beam_width) if self.workers > 1 else None
        with (ProcessPoolExecutor(max_workers=self.workers, mp_context=context,
                                  initializer=_init_search_worker,
                                  initargs=(dataset, self.catalog, device, terminal, shared_scores))
              if self.workers > 1 else nullcontext()) as pool:
            for round_index, (k, h_tail) in enumerate(self.schedule, 1):
                segments = (stop - start + k - 1) // k
                progress_every = max(1, segments // 10)
                round_began = perf_counter() if profile is not None else 0.
                round_evaluations_start = evaluations
                began = round_began
                reference_round = winning_round
                reference_actions = incumbent
                track_native_prefix = (round_index == len(self.schedule) and winning_round == 0)
                tail_policy = SequencePolicy(self.catalog, start, reference_actions)
                beam = [BeamNode(_search_copy(source), random_state, 0., ())]
                if profile is not None:
                    profile.record("round_start", round=round_index, k=k, h_tail=h_tail,
                                   reference_round=reference_round, initial_clone_seconds=perf_counter() - began)
                for step, origin in enumerate(range(start, stop, k), 1):
                    segment_began = perf_counter() if profile is not None else 0.
                    metrics = {} if profile is not None else None
                    parent_times, worker_jobs = Counter(), []
                    parent_count = len(beam)
                    end = min(origin + k, stop)
                    tail_end = min(end + h_tail, stop)
                    if pool is None:
                        candidates = []
                        parent_scores_list = []
                        for node_index, node in enumerate(beam):
                            parent_scores: dict[int, tuple[float, float, float]] = {}
                            parent_scores_list.append(parent_scores)
                            limit = candidates[-1][0] if len(candidates) == self.beam_width else None
                            expanded = _expand_actions(node, dataset, self.catalog, origin, end, tail_end,
                                                       tuple(order), tail_policy, self.beam_width,
                                                       terminal=terminal, score_limit=limit, metrics=metrics,
                                                       recorded_scores=parent_scores)
                            began = perf_counter() if profile is not None else 0.
                            for score, child in expanded:
                                candidates.append((score, node_index, action_ranks[child.actions[-1]], child))
                            candidates.sort(key=lambda item: item[:3])
                            del candidates[self.beam_width:]
                            if profile is not None:
                                parent_times["prune_seconds"] += perf_counter() - began
                            evaluations += len(order)
                        if track_native_prefix:
                            native_prefix = (native,) * (origin - start)
                            matched = next((parent_scores_list[i] for i, node in enumerate(beam)
                                            if node.actions == native_prefix), {})
                            finest_native_expansions.append(SegmentExpansion(origin, end, tail_end, native, matched))
                        beam = [
                            BeamNode(child.executor, child.random_state, child.total, child.actions, child.errors,
                                     (*child.expansions,
                                      SegmentExpansion(origin, end, tail_end, child.actions[-1], parent_scores_list[node_index])))
                            for _, node_index, _, child in candidates
                        ]
                    else:
                        began = perf_counter() if profile is not None else 0.
                        # The preceding map has finished all its jobs. Bounds
                        # belong only to this segment, including across rounds.
                        with shared_scores.get_lock():
                            shared_scores[:] = [np.inf] * self.beam_width
                        # Group actions so a parent is loaded once per group,
                        # rather than once per action. Limit groups to useful
                        # worker parallelism; return only each group's top-B.
                        groups = min(len(order), max(1, (self.workers + len(beam) - 1) // len(beam)))
                        chunk_size = (len(order) + groups - 1) // groups
                        jobs, parents = [], []
                        tail_slice = reference_actions[end - start:tail_end - start]
                        for node_index, node in enumerate(beam):
                            buffer = io.BytesIO()
                            torch.save({"executor": node.executor, "random_state": node.random_state}, buffer)
                            snapshot = buffer.getvalue()
                            if profile is not None:
                                parent_times["snapshot_bytes"] += len(snapshot)
                            for offset in range(0, len(order), chunk_size):
                                jobs.append((snapshot, tuple(order[offset:offset + chunk_size]), origin, end,
                                             tail_end, tail_slice, node.total, self.beam_width, profile is not None))
                                parents.append(node_index)
                        ranked = []
                        parent_scores_list = [{} for _ in range(len(beam))]
                        if profile is not None:
                            parent_times["prepare_jobs_seconds"] = perf_counter() - began
                            began = perf_counter()
                        for node_index, (results, chunk_scores, action_metrics, timings) in zip(parents, pool.map(_score_search_job, jobs), strict=True):
                            parent_scores_list[node_index].update(chunk_scores)
                            if profile is not None:
                                worker_jobs.append(timings)
                                for action_id, values in action_metrics.items():
                                    metrics.setdefault(action_id, Counter()).update(values)
                            for score, action_index, errors, snapshot in results:
                                ranked.append((score, node_index, action_ranks[action_index], action_index,
                                               errors, snapshot))
                        if profile is not None:
                            parent_times["dispatch_wait_seconds"] = perf_counter() - began
                            began = perf_counter()
                        evaluations += len(beam) * len(order)
                        ranked.sort(key=lambda item: item[:3])
                        if track_native_prefix:
                            native_prefix = (native,) * (origin - start)
                            matched = next((parent_scores_list[i] for i, node in enumerate(beam)
                                            if node.actions == native_prefix), {})
                            finest_native_expansions.append(SegmentExpansion(origin, end, tail_end, native, matched))
                        next_beam = []
                        for _, node_index, _, action_index, errors, snapshot in ranked[:self.beam_width]:
                            parent = beam[node_index]
                            saved = torch.load(io.BytesIO(snapshot), weights_only=False)
                            next_beam.append(BeamNode(saved["executor"], saved["random_state"],
                                                      parent.total + float(errors.sum()),
                                                      parent.actions + (action_index,) * (end - origin),
                                                      (*parent.errors, errors),
                                                      (*parent.expansions,
                                                       SegmentExpansion(origin, end, tail_end, action_index,
                                                                        parent_scores_list[node_index]))))
                        beam = next_beam
                        del jobs, ranked
                        if profile is not None:
                            parent_times["restore_prune_seconds"] = perf_counter() - began
                    if profile is not None:
                        profile.record("segment", round=round_index, k=k, h_tail=h_tail, segment=step, origin=origin,
                                       action_length=end - origin, tail_length=tail_end - end,
                                       parents=parent_count, survivors=len(beam),
                                       wall_seconds=perf_counter() - segment_began,
                                       parent=parent_times, actions=metrics, worker_jobs=worker_jobs)
                    if segments >= 20 and (step % progress_every == 0 or step == segments):
                        print(f"[{datetime.now().astimezone():%Y-%m-%d %H:%M:%S %z}] round {round_index}/{len(self.schedule)} k={k} "
                              f"segment {step}/{segments} (origin={origin})", flush=True)
                winner = min(beam, key=lambda node: node.total)
                winner_errors = np.concatenate(winner.errors)
                # Compare complete vectors with the same reduction shape across
                # rounds, independent of their different segment boundaries.
                winner_total = float(winner_errors.sum())
                previous_total = float(best_errors.sum())
                improved = (winner_total < previous_total
                            and not np.isclose(winner_total, previous_total, rtol=1e-12, atol=1e-12))
                reports.append({"round": round_index, "k": k, "h_tail": h_tail,
                                "reference_round": reference_round,
                                "candidate_mae": winner_total / (stop - start),
                                "incumbent_mae": previous_total / (stop - start), "improved": improved})
                if improved:
                    incumbent = winner.actions
                    best_errors = winner_errors
                    winning_round = round_index
                    winning_k = k
                    winning_h_tail = h_tail
                    winning_reference_round = reference_round
                    winning_reference_actions = reference_actions
                    winning_expansions = winner.expansions
                reports[-1]["best_mae"] = float(best_errors.mean())
                if profile is not None:
                    profile.record("round_end", **reports[-1],
                                   wall_seconds=perf_counter() - round_began,
                                   candidate_evaluations=evaluations - round_evaluations_start)
                print(f"[{datetime.now().astimezone():%Y-%m-%d %H:%M:%S %z}] round {round_index}/{len(self.schedule)} k={k} h_tail={h_tail} "
                      f"candidate_mae={reports[-1]['candidate_mae']:.6f} incumbent_mae={reports[-1]['incumbent_mae']:.6f} "
                      f"improved={improved}", flush=True)
                del beam, winner
        if winning_round == 0:
            winning_expansions = tuple(finest_native_expansions)
        return SearchResult(incumbent, native_errors, best_errors, reports, winning_round, evaluations,
                            winning_k, winning_h_tail, winning_reference_round,
                            winning_reference_actions, winning_expansions)

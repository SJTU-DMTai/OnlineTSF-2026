"""Scalar lookahead beam search over complete, independently owned snapshots."""

import copy
import io
import multiprocessing
from contextlib import nullcontext
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import numpy as np
import torch

from .actions import positive_integer
from .catalog import ActionCatalog
from .rollout import SequencePolicy, advance


@dataclass
class BeamNode:
    executor: object
    random_state: tuple
    total: float
    actions: tuple[int, ...]
    errors: tuple[np.ndarray, ...] = ()


@dataclass
class SearchResult:
    actions: tuple[int, ...]
    native_errors: np.ndarray
    errors: np.ndarray
    sweeps: list[dict]
    candidate_evaluations: int


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
                    shared_scores=None) -> list[tuple[float, BeamNode]]:
    """Keep local top-B; a discarded local candidate cannot be global top-B.

    Bounds use completed candidates only and strict > comparisons. Workers read
    the global bound once per candidate; a stale, higher bound only prunes less.
    Stable action order resolves exact ties just as in the serial search.
    """
    candidates = []
    for action_index in actions:
        limit = score_limit
        if len(candidates) == beam_width:
            limit = candidates[-1][0] if limit is None else min(limit, candidates[-1][0])
        if shared_scores is not None:
            with shared_scores.get_lock():
                global_limit = shared_scores[-1]
            if global_limit < np.inf:
                limit = global_limit if limit is None else min(limit, global_limit)
        branch = copy.deepcopy(node.executor)
        errors, state = advance(branch, dataset, origin, end, catalog.actions[action_index], node.random_state,
                                terminal=terminal, error_limit=limit, error_offset=node.total)
        if len(errors) != end - origin:
            continue
        total = node.total + float(errors.sum())
        if limit is not None and total > limit:
            continue
        tail_total = 0.
        if tail_end > end:
            tail = copy.deepcopy(branch)
            tail_errors, _ = advance(tail, dataset, end, tail_end, tail_policy, state, terminal=terminal,
                                     error_limit=limit, error_offset=total)
            del tail
            if len(tail_errors) != tail_end - end:
                continue
            tail_total = float(tail_errors.sum())
        score = total + tail_total
        if limit is not None and score > limit:
            continue
        if shared_scores is not None:
            # Only complete action-plus-tail scores can establish an upper bound.
            # Keep the lock out of model execution, cloning and serialization.
            with shared_scores.get_lock():
                shared_scores[:] = sorted([*shared_scores, score])[:beam_width]
        child = BeamNode(branch, state, total, node.actions + (action_index,) * (end - origin),
                         (*node.errors, errors))
        candidates.append((score, child))
        candidates.sort(key=lambda item: item[0])
        del candidates[beam_width:]
    return candidates


def _score_search_job(job: tuple) -> list[tuple]:
    snapshot, actions, origin, end, tail_end, tail_actions, total, beam_width = job
    saved = torch.load(io.BytesIO(snapshot), weights_only=False)
    node = BeamNode(saved["executor"], saved["random_state"], total, ())
    tail_policy = SequencePolicy(_search_catalog, end, tail_actions)
    candidates = _expand_actions(node, _search_dataset, _search_catalog, origin, end, tail_end,
                                 actions, tail_policy, beam_width, terminal=_search_terminal,
                                 shared_scores=_search_scores)
    results = []
    for score, child in candidates:
        buffer = io.BytesIO()
        torch.save({"executor": child.executor, "random_state": child.random_state}, buffer)
        results.append((score, child.actions[0], child.errors[0], buffer.getvalue()))
    return results


class LookaheadBeamSearch:
    def __init__(self, catalog: ActionCatalog, *, stride: int = 1, beam_width: int = 4,
                 h_tail: int = 128, sweeps: int = 1, workers: int = 1) -> None:
        for name, value in (("stride", stride), ("beam_width", beam_width), ("sweeps", sweeps), ("workers", workers)):
            positive_integer(value, name)
        if isinstance(h_tail, bool) or not isinstance(h_tail, int) or h_tail < 0:
            raise ValueError("h_tail must be a nonnegative integer")
        self.catalog, self.stride, self.beam_width = catalog, stride, beam_width
        self.h_tail, self.sweeps, self.workers = h_tail, sweeps, workers

    def search(self, source, dataset, start: int, stop: int, random_state: tuple, *,
               terminal: int | None = None, native_errors: np.ndarray | None = None) -> SearchResult:
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
        reports, evaluations = [], 0
        device = _executor_device(source)
        segments = (stop - start + self.stride - 1) // self.stride
        progress_every = max(1, segments // 10)
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
            for sweep in range(self.sweeps):
                tail_policy = SequencePolicy(self.catalog, start, incumbent)
                beam = [BeamNode(_search_copy(source), random_state, 0., ())]
                for step, origin in enumerate(range(start, stop, self.stride), 1):
                    end = min(origin + self.stride, stop)
                    tail_end = min(end + self.h_tail, stop)
                    if pool is None:
                        candidates = []
                        for node_index, node in enumerate(beam):
                            limit = candidates[-1][0] if len(candidates) == self.beam_width else None
                            expanded = _expand_actions(node, dataset, self.catalog, origin, end, tail_end,
                                                       tuple(order), tail_policy, self.beam_width,
                                                       terminal=terminal, score_limit=limit)
                            for score, child in expanded:
                                candidates.append((score, node_index, action_ranks[child.actions[-1]], child))
                            candidates.sort(key=lambda item: item[:3])
                            del candidates[self.beam_width:]
                            evaluations += len(order)
                        beam = [item[3] for item in candidates]
                    else:
                        # The preceding map has finished all its jobs. Bounds
                        # belong only to this segment, including across sweeps.
                        with shared_scores.get_lock():
                            shared_scores[:] = [np.inf] * self.beam_width
                        # Group actions so a parent is loaded once per group,
                        # rather than once per action. Limit groups to useful
                        # worker parallelism; return only each group's top-B.
                        groups = min(len(order), max(1, self.workers // len(beam)))
                        chunk_size = (len(order) + groups - 1) // groups
                        jobs, parents = [], []
                        tail_slice = incumbent[end - start:tail_end - start]
                        for node_index, node in enumerate(beam):
                            buffer = io.BytesIO()
                            torch.save({"executor": node.executor, "random_state": node.random_state}, buffer)
                            snapshot = buffer.getvalue()
                            for offset in range(0, len(order), chunk_size):
                                jobs.append((snapshot, tuple(order[offset:offset + chunk_size]), origin, end,
                                             tail_end, tail_slice, node.total, self.beam_width))
                                parents.append(node_index)
                        ranked = []
                        for node_index, results in zip(parents, pool.map(_score_search_job, jobs), strict=True):
                            for score, action_index, errors, snapshot in results:
                                ranked.append((score, node_index, action_ranks[action_index], action_index,
                                               errors, snapshot))
                        evaluations += len(beam) * len(order)
                        ranked.sort(key=lambda item: item[:3])
                        next_beam = []
                        for _, node_index, _, action_index, errors, snapshot in ranked[:self.beam_width]:
                            parent = beam[node_index]
                            saved = torch.load(io.BytesIO(snapshot), weights_only=False)
                            next_beam.append(BeamNode(saved["executor"], saved["random_state"],
                                                      parent.total + float(errors.sum()),
                                                      parent.actions + (action_index,) * (end - origin),
                                                      (*parent.errors, errors)))
                        beam = next_beam
                        del jobs, ranked
                    if segments >= 20 and (step % progress_every == 0 or step == segments):
                        print(f"[{datetime.now().astimezone():%Y-%m-%d %H:%M:%S %z}] sweep {sweep + 1}/{self.sweeps} "
                              f"segment {step}/{segments} (origin={origin})", flush=True)
                winner = min(beam, key=lambda node: node.total)
                previous_total = float(best_errors.sum())
                improved = (winner.actions != incumbent and winner.total < previous_total
                            and not np.isclose(winner.total, previous_total, rtol=1e-12, atol=1e-12))
                reports.append({"sweep": sweep + 1, "candidate_mae": winner.total / (stop - start),
                                "incumbent_mae": previous_total / (stop - start), "improved": improved})
                print(f"[{datetime.now().astimezone():%Y-%m-%d %H:%M:%S %z}] sweep {sweep + 1}/{self.sweeps} "
                      f"candidate_mae={reports[-1]['candidate_mae']:.6f} incumbent_mae={reports[-1]['incumbent_mae']:.6f} "
                      f"improved={improved}", flush=True)
                if not improved:
                    break
                incumbent = winner.actions
                best_errors = np.concatenate(winner.errors)
                del beam, winner
        return SearchResult(incumbent, native_errors, best_errors, reports, evaluations)

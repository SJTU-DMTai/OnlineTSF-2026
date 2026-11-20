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


def _init_search_worker(dataset, catalog: ActionCatalog, device: torch.device, terminal: int | None) -> None:
    global _search_dataset, _search_catalog, _search_terminal
    _search_dataset = dataset
    _search_catalog = catalog
    _search_terminal = terminal
    torch.set_num_threads(1)
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _score_search_job(job: tuple[bytes, int, int, int, int, tuple[int, ...]]) -> tuple[float, float]:
    snapshot, action_index, origin, end, tail_end, tail_actions = job
    saved = torch.load(io.BytesIO(snapshot), weights_only=False)
    branch = saved["executor"]
    errors, state = advance(branch, _search_dataset, origin, end,
                            _search_catalog.actions[action_index], saved["random_state"], terminal=_search_terminal)
    tail_total = 0.
    if tail_end > end:
        tail_policy = SequencePolicy(_search_catalog, end, tail_actions)
        tail_errors, _ = advance(branch, _search_dataset, end, tail_end, tail_policy, state, terminal=_search_terminal)
        tail_total = float(tail_errors.sum())
    return float(errors.sum()), tail_total


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
               terminal: int | None = None) -> SearchResult:
        if not 0 <= start < stop <= len(dataset) or source._last_index not in (None, start - 1):
            raise ValueError("search needs a consecutive source snapshot and a nonempty legal interval")
        native = self.catalog.native_indices[0]
        incumbent = (native,) * (stop - start)
        baseline = copy.deepcopy(source)
        native_errors, _ = advance(baseline, dataset, start, stop,
                                   SequencePolicy(self.catalog, start, incumbent), random_state, terminal=terminal)
        del baseline
        best_errors = native_errors
        reports, evaluations = [], 0
        device = _executor_device(source)
        segments = (stop - start + self.stride - 1) // self.stride
        progress_every = max(1, segments // 10)
        order = [native] + [i for i in range(len(self.catalog.ids)) if i != native]
        # The complete incumbent is protected outside the finite beam. This
        # preserves Native even for B=1 without occupying its only search slot.
        with (ProcessPoolExecutor(max_workers=self.workers, mp_context=multiprocessing.get_context("spawn"),
                                  initializer=_init_search_worker, initargs=(dataset, self.catalog, device, terminal))
              if self.workers > 1 else nullcontext()) as pool:
            for sweep in range(self.sweeps):
                tail_policy = SequencePolicy(self.catalog, start, incumbent)
                beam = [BeamNode(copy.deepcopy(source), random_state, 0., ())]
                for step, origin in enumerate(range(start, stop, self.stride), 1):
                    end = min(origin + self.stride, stop)
                    tail_end = min(end + self.h_tail, stop)
                    if pool is None:
                        candidates = []
                        for node in beam:
                            for action_index in order:
                                branch = copy.deepcopy(node.executor)
                                errors, state = advance(branch, dataset, origin, end,
                                                        self.catalog.actions[action_index], node.random_state, terminal=terminal)
                                total = node.total + float(errors.sum())
                                tail_total = 0.
                                if tail_end > end:
                                    tail = copy.deepcopy(branch)
                                    tail_errors, _ = advance(tail, dataset, end, tail_end, tail_policy, state, terminal=terminal)
                                    tail_total = float(tail_errors.sum())
                                    del tail
                                child = BeamNode(branch, state, total,
                                                 node.actions + (action_index,) * (end - origin))
                                candidates.append((total + tail_total, child))
                                evaluations += 1
                                # Bound live snapshots to B, rather than B*|U|.
                                candidates.sort(key=lambda item: item[0])
                                del candidates[self.beam_width:]
                        beam = [item[1] for item in candidates]
                    else:
                        snapshots = []
                        for node in beam:
                            buffer = io.BytesIO()
                            torch.save({"executor": node.executor, "random_state": node.random_state}, buffer)
                            snapshots.append(buffer.getvalue())
                        tail_slice = incumbent[end - start:tail_end - start]
                        jobs = [(snapshots[node_idx], action_index, origin, end, tail_end, tail_slice)
                                for node_idx in range(len(beam)) for action_index in order]
                        scores = list(pool.map(_score_search_job, jobs))
                        evaluations += len(jobs)
                        ranked = []
                        cursor = 0
                        for node_idx, node in enumerate(beam):
                            for rank, action_index in enumerate(order):
                                act_total, tail_total = scores[cursor]
                                ranked.append((node.total + act_total + tail_total, node_idx, rank, action_index))
                                cursor += 1
                        ranked.sort(key=lambda item: (item[0], item[1], item[2]))
                        next_beam = []
                        for _, node_idx, _, action_index in ranked[:self.beam_width]:
                            parent = beam[node_idx]
                            branch = copy.deepcopy(parent.executor)
                            errors, state = advance(branch, dataset, origin, end,
                                                    self.catalog.actions[action_index], parent.random_state, terminal=terminal)
                            next_beam.append(BeamNode(branch, state, parent.total + float(errors.sum()),
                                                      parent.actions + (action_index,) * (end - origin)))
                        beam = next_beam
                        del snapshots
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
                replay = copy.deepcopy(source)
                best_errors, _ = advance(replay, dataset, start, stop,
                                         SequencePolicy(self.catalog, start, incumbent), random_state, terminal=terminal)
                if not np.isclose(best_errors.sum(), winner.total, rtol=1e-10, atol=1e-12):
                    raise RuntimeError("search winner does not match sequence replay")
                del replay, beam, winner
        return SearchResult(incumbent, native_errors, best_errors, reports, evaluations)

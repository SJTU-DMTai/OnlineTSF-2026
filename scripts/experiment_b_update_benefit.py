# -*- coding: utf-8 -*-
"""Experiment B: paired update-versus-skip branches from identical causal states."""

from __future__ import annotations

import copy
import math
import random
from collections import defaultdict
from statistics import mean

from drift_experiment_data import aggregate_metrics, common_parser, initialize_experiment, phase_at, prepared_runs, write_rows
from drift_experiment_reporting import ReplayReport, finish_experiment, run_replay, save_run
from drift_experiment_runtime import UpdatePolicy, clone_with_policy, restore_rng, rng_state, update_opportunities
from onlinetsf.methods import DSOFMethod, UnderCaliMethod
from onlinetsf.online import OnlineExecutor


def select_probes(opportunities, labels, context_length, args):
    groups = defaultdict(list)
    for arrival, origin in opportunities:
        phase = phase_at(arrival + context_length - 1, labels, args.pre_rows, args.post_rows)
        groups[(phase["phase"], phase["drift_type"])].append((arrival, origin))
    generator = random.Random(args.selection_seed)
    selected = []
    for key in sorted(groups):
        candidates = groups[key]
        selected.extend(generator.sample(candidates, min(args.probes_per_phase, len(candidates))))
    return set(selected)


class ProbeExecutor(OnlineExecutor):
    """Probe just before feedback consumption, after it is removed from the heap.

    A branch receives that feedback explicitly once. It then consumes any other
    labels due at the same time before issuing the first evaluated prediction.
    """

    def __init__(self, controlled, dataset, stop, stream, probes, args):
        delay = list(range(1, dataset.horizon + 1)) if isinstance(controlled.method, DSOFMethod) else dataset.horizon
        super().__init__(controlled, feedback_delay=delay, keep_predictions=False)
        self.dataset, self.stop, self.stream, self.probes, self.args = dataset, stop, stream, probes, args
        self.effects = []

    def _consume_feedback(self, pending, current_context):
        if current_context is not None and (pending.available_at, pending.index) in self.probes:
            self.probe(pending, current_context)
        return super()._consume_feedback(pending, current_context)

    def probe(self, pending, current_context):
        method = self.method.method
        would_update = True
        update_kind = "complete_window"
        if isinstance(method, DSOFMethod):
            fast = bool(pending.observed_mask[0].all())
            slow = bool(pending.observed_mask[-1].all()) and len(method.replay) + 1 >= method.replay_batch_size and method._complete_count % method.replay_frequency == 0
            update_kind = "fast_and_replay" if fast and slow else "fast" if fast else "replay"
        if isinstance(method, UnderCaliMethod):
            score = method._pending_routes[pending.index][0]
            would_update = method.trigger_mean is not None and score > method.trigger_mean + method.trigger_std_k * math.sqrt(method.trigger_variance)
        clock_state = self.state_dict()
        initial_rng = rng_state()
        available = min(max(self.args.lookaheads), self.stop - pending.available_at)
        phase = phase_at(pending.available_at + self.dataset.context_length - 1, self.stream["labels"], self.args.pre_rows, self.args.post_rows)
        try:
            for protocol in self.args.branch_protocols:
                losses, steps = {}, {}
                for action in ("force", "skip"):
                    restore_rng(initial_rng)
                    controlled = copy.deepcopy(self.method)
                    controlled.policy = UpdatePolicy(name="no_grad" if protocol == "isolated" else "native")
                    controlled.records.clear()
                    controlled.override_next = action
                    branch = OnlineExecutor(controlled, feedback_delay=self.feedback_delay, keep_predictions=False)
                    branch.load_state_dict(clock_state)
                    branch._consume_feedback(copy.deepcopy(pending), current_context)
                    steps[action] = controlled.records[-1]["optimizer_steps"]
                    losses[action] = []
                    controlled.records.clear()
                    for index in range(pending.available_at, pending.available_at + available):
                        context, target = self.dataset[index]
                        prediction = branch.step(index, context, target).emission.prediction
                        losses[action].append((prediction - target).abs().mean().item())
                        branch._events.clear()
                        controlled.records.clear()
                for lookahead in self.args.lookaheads:
                    count = min(available, lookahead)
                    update_mae, skip_mae = mean(losses["force"][:count]), mean(losses["skip"][:count])
                    benefit = skip_mae - update_mae
                    self.effects.append({"forecast_index": pending.index, "available_at": pending.available_at,
                                         "raw_available_index": pending.available_at + self.dataset.context_length - 1,
                                         **phase, "protocol": protocol, "lookahead": lookahead,
                                         "update_kind": update_kind,
                                         "first_evaluated_target_index": pending.available_at + self.dataset.context_length,
                                         "evaluated_forecasts": count, "right_censored": count < lookahead,
                                         "native_would_update": would_update, "update_optimizer_steps": steps["force"],
                                         "skip_optimizer_steps": steps["skip"], "update_mae": update_mae,
                                         "skip_mae": skip_mae, "benefit": benefit,
                                         "harmful": benefit < -self.args.benefit_epsilon,
                                         "helpful": benefit > self.args.benefit_epsilon,
                                         "missed_helpful_update": not would_update and benefit > self.args.benefit_epsilon})
        finally:
            # Neither branch may advance the main stream's dropout/replay random state.
            restore_rng(initial_rng)


def main():
    parser = common_parser(__doc__)
    parser.add_argument("--lookaheads", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument("--probes-per-phase", type=int, default=8, help="random opportunities per phase/type stratum")
    parser.add_argument("--branch-protocols", nargs="+", choices=("isolated", "continuation"), default=["isolated", "continuation"])
    parser.add_argument("--benefit-epsilon", type=float, default=1e-8)
    args = parser.parse_args()
    if args.probes_per_phase <= 0 or any(value <= 0 for value in args.lookaheads) or args.benefit_epsilon < 0:
        parser.error("probe counts and lookaheads must be positive; benefit epsilon must be nonnegative")
    if not math.isfinite(args.benefit_epsilon) or len(set(args.lookaheads)) != len(args.lookaheads) or len(set(args.branch_protocols)) != len(args.branch_protocols):
        parser.error("epsilon must be finite; lookaheads and protocols must not contain duplicates")
    streams = initialize_experiment(args)
    summaries, phases, effects = [], [], []
    for stream, dataset, method, start, stop, directory, metadata in prepared_runs(args, streams):
        opportunities = update_opportunities(method, start, stop, dataset.horizon)
        probes = select_probes(opportunities, stream["labels"], dataset.context_length, args)
        controlled = clone_with_policy(method, UpdatePolicy(), dataset.context_length, args.parameter_metrics)
        executor = ProbeExecutor(controlled, dataset, stop, stream, probes, args)
        report = ReplayReport(stream, dataset, args)
        summary = run_replay(controlled, dataset, start, stop, report, executor)
        summary, rows = save_run(directory / "native_with_probes", metadata, "native_with_probes", report, summary)
        summaries.append(summary)
        phases.extend(rows)
        records = [{**metadata, **effect} for effect in executor.effects]
        write_rows(directory / "update_effects.csv", records)
        effects.extend(records)
    write_rows(args.output / "update_effects.csv", effects)
    complete = [row for row in effects if not row["right_censored"]]
    aggregate_metrics(complete, args.output / "benefit_by_drift_type_phase.csv",
                      ("family", "strategy", "horizon", "drift_type", "phase", "update_kind", "protocol", "lookahead"),
                      ("benefit", "harmful", "helpful", "missed_helpful_update"))
    finish_experiment(args.output, summaries, phases)


if __name__ == "__main__":
    main()

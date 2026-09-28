# -*- coding: utf-8 -*-
"""Experiment D: frequency, inner update rounds, and learning-rate sensitivity.

Periodic treatment reuses the latest eligible feedback; it does not buffer skipped
windows into a batch. DSOF independently retains all causally completed windows.
"""

from __future__ import annotations

from itertools import product
import math

from drift_experiment_data import common_parser, initialize_experiment, prepared_runs
from drift_experiment_reporting import ReplayReport, finish_experiment, run_replay, save_run
from drift_experiment_runtime import UpdatePolicy, clone_with_policy, restore_rng, rng_state, update_opportunities


def main():
    parser = common_parser(__doc__)
    parser.add_argument("--periods", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--rounds", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--lr-scales", nargs="+", type=float, default=[0.3, 1.0, 3.0])
    parser.add_argument("--design", choices=("grid", "matched_rounds"), default="grid")
    parser.add_argument("--respect-native-trigger", action="store_true", help="retain Under-Cali's gate; default forces selected opportunities")
    args = parser.parse_args()
    if any(not math.isfinite(value) or value <= 0 for value in [*args.periods, *args.rounds, *args.lr_scales]):
        parser.error("periods, rounds, and learning-rate scales must be positive")
    if any(len(values) != len(set(values)) for values in (args.periods, args.rounds, args.lr_scales)):
        parser.error("periods, rounds, and learning-rate scales must not contain duplicates")
    streams = initialize_experiment(args)
    summaries, phases = [], []
    for stream, dataset, method, start, stop, directory, metadata in prepared_runs(args, streams):
        initial_rng = rng_state()
        opportunity_count = len(update_opportunities(method, start, stop, dataset.horizon))
        round_budget = opportunity_count if args.design == "matched_rounds" else None
        settings = list(product(args.periods, args.rounds, args.lr_scales)) if args.design == "grid" else [(period, period, scale) for period in args.periods for scale in args.lr_scales]
        variants = [("native", UpdatePolicy(), 1.0)]
        for period, rounds, scale in settings:
            name = f"period{period}_rounds{rounds}_lr{scale:g}"
            policy = UpdatePolicy(name="periodic", period=period, rounds=rounds, force=not args.respect_native_trigger,
                                  round_budget=round_budget, total_opportunities=opportunity_count)
            variants.append((name, policy, scale))
        for name, policy, scale in variants:
            restore_rng(initial_rng)
            controlled = clone_with_policy(method, policy, dataset.context_length, args.parameter_metrics, scale)
            report = ReplayReport(stream, dataset, args)
            summary = run_replay(controlled, dataset, start, stop, report)
            summary.update(period=policy.period, rounds=policy.rounds, lr_scale=scale,
                           forced_selected_updates=policy.force, design=args.design,
                           control_rounds=controlled.rounds_used if policy.rounds is not None else None,
                           requested_round_budget=policy.round_budget,
                           budget_fill_events=sum(row["budget_fill"] for row in report.updates))
            if policy.round_budget is not None and controlled.rounds_used != policy.round_budget:
                raise RuntimeError("matched-round treatment did not spend its requested round budget")
            summary, rows = save_run(directory / name, metadata, name, report, summary)
            summaries.append(summary)
            phases.extend(rows)
    finish_experiment(args.output, summaries, phases)


if __name__ == "__main__":
    main()

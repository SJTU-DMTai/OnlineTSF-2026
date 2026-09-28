# -*- coding: utf-8 -*-
"""Experiment C: place a fixed number of update opportunities at different times.

The matched budget is feedback opportunities, NOT FLOPs or optimizer steps.
Component step counts are reported separately. Oracle schedules are privileged.
"""

from __future__ import annotations

import random
import math

from drift_experiment_data import common_parser, initialize_experiment, prepared_runs, write_rows
from drift_experiment_reporting import ReplayReport, finish_experiment, run_replay, save_run
from drift_experiment_runtime import UpdatePolicy, clone_with_policy, restore_rng, rng_state, update_opportunities


def timing_schedule(opportunities, name, budget, labels, context_length, delay=0, seed=0):
    if not 0 < budget <= len(opportunities):
        raise ValueError("timing budget must fit the available opportunities")
    if name == "uniform":
        return {opportunities[index * len(opportunities) // budget] for index in range(budget)}
    if name == "random":
        return set(random.Random(seed).sample(opportunities, budget))
    if name != "oracle":
        raise ValueError(f"unknown schedule: {name}")
    ranked = []
    for arrival, origin in opportunities:
        raw_time = arrival + context_length - 1
        distances = [raw_time - label["start_index"] - delay for label in labels
                     if raw_time >= label["start_index"] + delay]
        if distances:
            ranked.append((min(distances), arrival, origin))
    if len(ranked) < budget:
        raise ValueError(f"oracle delay {delay}: only {len(ranked)} post-trigger opportunities for budget {budget}")
    return {(arrival, origin) for _, arrival, origin in sorted(ranked)[:budget]}


def main():
    parser = common_parser(__doc__)
    parser.add_argument("--budget-fraction", type=float, default=0.25)
    parser.add_argument("--budget", type=int, help="exact opportunity count, overriding budget-fraction")
    parser.add_argument("--oracle-delays", type=int, nargs="+", default=[0, 8, 32, 64])
    parser.add_argument("--residual-threshold", type=float, default=1.0, help="absolute forecast error in fixed standardized units")
    parser.add_argument("--rounds", type=int, default=1)
    args = parser.parse_args()
    if not 0 < args.budget_fraction <= 1 or args.budget is not None and args.budget <= 0 or args.rounds <= 0:
        parser.error("budget-fraction must be in (0,1]; budget and rounds must be positive")
    if any(delay < 0 for delay in args.oracle_delays) or not math.isfinite(args.residual_threshold) or args.residual_threshold <= 0:
        parser.error("oracle delays must be nonnegative; residual threshold must be positive")
    if len(args.oracle_delays) != len(set(args.oracle_delays)):
        parser.error("oracle delays must not contain duplicates")
    streams = initialize_experiment(args)
    summaries, phases, exclusions = [], [], []
    for stream, dataset, method, start, stop, directory, metadata in prepared_runs(args, streams):
        opportunities = update_opportunities(method, start, stop, dataset.horizon)
        if not opportunities:
            exclusions.append({**metadata, "policy": "all", "reason": "no_training_opportunities"})
            continue
        budget = args.budget if args.budget is not None else max(1, int(len(opportunities) * args.budget_fraction))
        if budget > len(opportunities):
            raise ValueError(f"budget {budget} exceeds {len(opportunities)} available opportunities")
        initial_rng = rng_state()
        specifications = [("uniform", 0), ("random", 0), ("residual", 0)] + [("oracle", delay) for delay in args.oracle_delays]
        for name, delay in specifications:
            policy_name = f"oracle_delay_{delay}" if name == "oracle" else name
            if name == "residual":
                policy = UpdatePolicy(name="residual", force=True, rounds=args.rounds, budget=budget,
                                      total_opportunities=len(opportunities), residual_threshold=args.residual_threshold)
            else:
                try:
                    online_labels = [label for label in stream["labels"] if label["start_index"] >= start + dataset.context_length - 1]
                    selected = timing_schedule(opportunities, name, budget, online_labels, dataset.context_length, delay, metadata["seed"])
                except ValueError as error:
                    exclusions.append({**metadata, "policy": policy_name, "reason": str(error)})
                    continue
                policy = UpdatePolicy(name="scheduled", selected=selected, force=True, rounds=args.rounds)
            restore_rng(initial_rng)
            controlled = clone_with_policy(method, policy, dataset.context_length, args.parameter_metrics)
            report = ReplayReport(stream, dataset, args)
            summary = run_replay(controlled, dataset, start, stop, report)
            if summary["selected_opportunities"] != budget or summary["opportunities"] != len(opportunities):
                raise RuntimeError("actual timing schedule differs from its causal opportunity plan")
            summary.update(requested_budget=budget, budget_unit="feedback_opportunities", oracle=name == "oracle",
                           budget_fill_updates=sum(row["budget_fill"] and row["selected"] for row in report.updates))
            summary, rows = save_run(directory / policy_name, metadata, policy_name, report, summary)
            summaries.append(summary)
            phases.extend(rows)
    write_rows(args.output / "excluded_policies.csv", exclusions)
    finish_experiment(args.output, summaries, phases)


if __name__ == "__main__":
    main()

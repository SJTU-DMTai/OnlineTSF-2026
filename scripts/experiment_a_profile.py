# -*- coding: utf-8 -*-
"""Experiment A: native, no-gradient, and frozen behavior by drift type and phase."""

from __future__ import annotations

import torch

from drift_experiment_data import common_parser, initialize_experiment, prepared_runs
from drift_experiment_reporting import ReplayReport, finish_experiment, run_replay, save_run
from drift_experiment_runtime import UpdatePolicy, clone_with_policy, restore_rng, rng_state


class PersistenceMethod:
    """Repeat the last observed season; period 1 is ordinary persistence."""

    def __init__(self, dataset, period=1):
        self.model = torch.nn.Identity()
        self.device = torch.device("cpu")
        self.horizon, self.targets, self.period = dataset.horizon, dataset.target_indices, period

    def predict(self, context):
        positions = torch.arange(self.horizon) % self.period + len(context) - self.period
        return context[positions][:, self.targets]

    def on_feedback(self, feedback):
        return None


def main():
    parser = common_parser(__doc__)
    parser.add_argument("--policies", nargs="+", choices=("native", "no_grad", "frozen"), default=["native", "no_grad", "frozen"])
    parser.add_argument("--seasonal-period", type=int, help="also run seasonal persistence; period must fit the context")
    args = parser.parse_args()
    if len(set(args.policies)) != len(args.policies):
        parser.error("policies must not contain duplicates")
    if args.seasonal_period is not None and not 1 <= args.seasonal_period <= args.context_length:
        parser.error("seasonal-period must be between 1 and context-length")
    streams = initialize_experiment(args)
    summaries, phases = [], []
    for stream, dataset, method, start, stop, directory, metadata in prepared_runs(args, streams):
        initial_rng = rng_state()
        variants = [(name, method, metadata) for name in args.policies]
        if metadata["strategy"] == args.strategies[0]:
            variants.append(("persistence", PersistenceMethod(dataset), {**metadata, "strategy": "persistence"}))
            if args.seasonal_period is not None:
                variants.append(("seasonal_persistence", PersistenceMethod(dataset, args.seasonal_period),
                                 {**metadata, "strategy": "seasonal_persistence"}))
        for name, base_method, variant_metadata in variants:
            restore_rng(initial_rng)
            policy = UpdatePolicy(name=name if name in args.policies else "native")
            controlled = clone_with_policy(base_method, policy, dataset.context_length, args.parameter_metrics)
            controlled.collect_components = True
            report = ReplayReport(stream, dataset, args)
            summary = run_replay(controlled, dataset, start, stop, report)
            summary, rows = save_run(directory / name, variant_metadata, name, report, summary)
            summaries.append(summary)
            phases.extend(rows)
    finish_experiment(args.output, summaries, phases)


if __name__ == "__main__":
    main()

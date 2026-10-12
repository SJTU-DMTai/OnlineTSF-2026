"""Absolute errors over every completed policy test origin/segment; no torch."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from update_policy_common import DEFAULT_RUNS, find_job


def save_figure(fig, axes, path):
    for ax in axes:
        ax.grid(axis="y", color="#E1E5E9", linewidth=.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#CDD3D8")
        ax.tick_params(length=3, color="#8C969F")
        ax.set_xlabel("Steps")
    fig.tight_layout(rect=(0, 0, 1, .87))
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("runs/policy-exp1-exp2-results"))
    parser.add_argument("--source-runs", type=Path, default=DEFAULT_RUNS,
                        help="original same-state forks, used to read experiment 1 forecast errors")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.results / "overall_figures"
    output.mkdir(parents=True, exist_ok=True)
    first = args.results / "runs/policy-exp1/seed2"
    second = args.results / "runs/policy-exp2/seed2"
    outcomes = pd.read_csv(first / "test_outcomes.csv")
    bundle = json.loads((first / "selectors.json").read_text(encoding="utf-8"))
    job = find_job(args.source_runs, bundle["dataset"], bundle["method"], bundle["horizon"], bundle["test_seed"])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 11, "axes.labelsize": 10,
                         "legend.fontsize": 9, "pdf.fonttype": 42})
    controllers = [("native", "Native", "#285B86"),
                   ("policy_data", "Data policy", "#399A80"),
                   ("policy_full", "Full policy", "#D27735")]
    summaries = []
    for experiment, directory, filename in [(1, first, "01_experiment1_all_origins"),
                                             (2, second, "02_experiment2_all_segments")]:
        summary = pd.read_csv(directory / "summary.csv")
        summary = summary[(summary.lookahead == 128) & summary.controller.isin([item[0] for item in controllers])]
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        reference_indices = None
        for controller, label, color in controllers:
            trajectories, origin_indices = [], []
            if experiment == 1:
                selected = outcomes[(outcomes.controller == controller) &
                                    (outcomes.lookahead == 128)].sort_values("clock")
                if len(selected) != len(bundle["test_clocks"]):
                    raise ValueError("experiment 1 must include every test origin")
                for row in selected.itertuples():
                    frame = pd.read_csv(job / "probes" / row.probe_id / row.selected_action / "forecasts.csv")
                    if len(frame) != 128:
                        raise ValueError("expected complete 128-step source forks")
                    trajectories.append(frame[["mae", "mse"]].to_numpy())
                    origin_indices.append(frame.origin_raw_index.to_numpy())
            else:
                manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
                for clock in manifest["start_clocks"]:
                    frame = pd.read_csv(directory / f"start_{clock}" / controller / "forecasts.csv")
                    if len(frame) != manifest["steps"]:
                        raise ValueError("expected complete continuous segment")
                    trajectories.append(frame[["mae", "mse"]].to_numpy())
                    origin_indices.append(frame.origin_raw_index.to_numpy())
            indices = np.stack(origin_indices)
            if reference_indices is None:
                reference_indices = indices
            elif not np.array_equal(reference_indices, indices):
                raise ValueError("controllers must be evaluated at identical forecast origins")
            errors = np.stack(trajectories)
            steps = np.arange(1, errors.shape[1] + 1)
            cumulative = errors.mean(axis=0).cumsum(axis=0) / steps[:, None]
            expected = summary[summary.controller == controller][["mae", "mse"]].iloc[0].to_numpy()
            np.testing.assert_allclose(cumulative[-1], expected, rtol=1e-10, atol=1e-12)
            for column, (ax, metric) in enumerate(zip(axes, ["mae", "mse"])):
                ax.plot(steps, cumulative[:, column], color=color, linewidth=1.8, label=label)
                ax.set_ylabel("Cumulative " + metric.upper())
                ax.set_xlim(1, len(steps))
                ax.set_xticks([1, 32, 64, 96, 128] if experiment == 1 else [1, 128, 256, 384, 512])
            curve = pd.DataFrame(cumulative, columns=["mae", "mse"])
            curve.insert(0, "steps", steps)
            curve.to_csv(output / f"experiment{experiment}_{controller}_curve.csv", index=False)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, .99))
        save_figure(fig, axes, output / filename)
        summary = summary.copy()
        summary.insert(0, "experiment", experiment)
        summaries.append(summary[["experiment", "controller", "mae", "mse"]])
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(output / "summary.csv", index=False)
    print(summary.round(6).to_string(index=False))
    print(output)


if __name__ == "__main__":
    main()

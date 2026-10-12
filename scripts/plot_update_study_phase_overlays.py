"""Redraw the two full-stream frequency comparisons with aligned drift annotations."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import numpy as np
import pandas as pd

from plot_update_study import COLORS, draw_curve, policy_label


PHASE_COLORS = {"stable": "#DDEAE2", "transition": "#EACB83", "drift": "#C65D68"}
PHASE_LABELS = {"stable": "平稳候选区", "transition": "过渡区", "drift": "漂移区"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/update-study-completed/update-study-main-4gpu"))
    parser.add_argument("--annotations", type=Path, default=Path("runs/real-drift-electricity-5pct-20261010"))
    parser.add_argument("--output", type=Path, default=Path("reports/update_study_completed/figures"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": ["DejaVu Sans", "Microsoft YaHei"], "font.size": 10,
                         "axes.titlesize": 12, "axes.labelsize": 10, "pdf.fonttype": 42})
    policies = ["native", "every4_lr1", "every16_lr1", "every1_lr0.3"]
    annotations = {}
    for dataset in ["ETTh2", "ETTm1", "Electricity"]:
        folder = args.annotations / dataset
        labels = pd.read_csv(folder / "labeled_data.csv", usecols=["annotation_index", "annotation_phase", "is_drift_point"])
        intervals = pd.read_csv(folder / "intervals.csv")
        events = pd.read_csv(folder / "drift_points.csv")
        assert np.array_equal(labels.annotation_index, np.arange(len(labels)))
        assert np.array_equal(intervals.start_index.iloc[1:], intervals.end_index_exclusive.iloc[:-1])
        assert intervals.start_index.iloc[0] == 0 and intervals.end_index_exclusive.iloc[-1] == len(labels)
        for interval in intervals.itertuples():
            assert (labels.annotation_phase.iloc[interval.start_index:interval.end_index_exclusive] == interval.phase).all()
        assert np.array_equal(np.flatnonzero(labels.is_drift_point), events.detection_index.to_numpy())
        annotations[dataset] = (labels, intervals, events)

    audit = []
    for method, datasets in [("fsnet", ["ETTh2"]), ("onenet", ["ETTh2", "ETTm1", "Electricity"])]:
        rows = len(datasets)
        fig = plt.figure(figsize=(13.6, 3.7 * rows + .8))
        grid = fig.add_gridspec(rows, 2, left=.065, right=.985, bottom=.14 if rows == 1 else .065,
                                top=.75 if rows == 1 else .90, hspace=.48, wspace=.15)
        curve_handles = []
        for i, dataset in enumerate(datasets):
            labels, intervals, events = annotations[dataset]
            for j, horizon in enumerate([1, 24]):
                panel = grid[i, j].subgridspec(2, 1, height_ratios=[12, 1], hspace=.07)
                ax = fig.add_subplot(panel[0])
                phase_ax = fig.add_subplot(panel[1], sharex=ax)
                gpu = (0 if horizon == 1 else 1) if method == "fsnet" else (2 if horizon == 1 else 3)
                reference = None
                window = 672 if dataset == "ETTm1" else 168
                for color, policy in zip(COLORS, policies):
                    values = []
                    for seed in range(3):
                        path = args.runs / f"gpu{gpu}" / dataset / f"{method}_h{horizon}_seed{seed}" / policy / "forecasts.csv"
                        frame = pd.read_csv(path, usecols=["origin_raw_index", "target_start_index", "target_end_index_exclusive", "mae"])
                        assert (frame.target_start_index == frame.origin_raw_index + 1).all()
                        assert (frame.target_end_index_exclusive == frame.target_start_index + horizon).all()
                        if reference is None:
                            reference = frame.target_start_index.to_numpy()
                            assert np.array_equal(reference, np.arange(reference[0], reference[0] + len(reference)))
                        assert np.array_equal(frame.target_start_index, reference), path
                        values.append(frame.mae.to_numpy())
                    draw_curve(ax, np.stack(values), color, policy_label(policy), window=window)
                if not curve_handles:
                    curve_handles = ax.get_legend_handles_labels()[0]
                first, last = int(reference[0]), int(reference[-1])
                assert last + horizon <= len(labels)
                # Online step s corresponds to raw target row first + s - 1.
                for phase, color in PHASE_COLORS.items():
                    spans = []
                    for interval in intervals[intervals.phase == phase].itertuples():
                        left, right = max(first, interval.start_index), min(last + 1, interval.end_index_exclusive)
                        if left < right:
                            spans.append((left - first + .5, right - left))
                    phase_ax.broken_barh(spans, (0, 1), facecolors=color, edgecolors="none")
                event_rows = events.loc[events.detection_index.between(first, last), "detection_index"].to_numpy()
                event_steps = event_rows - first + 1
                # Point marks are above the phase strip so dense events do not hide its colors.
                phase_ax.vlines(event_steps, 1.08, 1.48, color="#613A42", linewidth=.55)
                phase_ax.set_ylim(0, 1.55)
                phase_ax.set_yticks([])
                phase_ax.set_xlim(.5, len(reference) + .5)
                phase_ax.set_xlabel("Online steps", labelpad=5)
                phase_ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
                phase_ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
                phase_ax.tick_params(axis="x", length=3, color="#B8C2CC", pad=4)
                phase_ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
                ax.tick_params(axis="x", bottom=False, labelbottom=False)
                ax.set_ylabel("Rolling MAE")
                ax.grid(axis="y", color="#E2E7EB", linewidth=.65)
                ax.spines[["top", "right"]].set_visible(False)
                ax.spines[["left", "bottom"]].set_color("#CDD4DA")
                ax.margins(x=0)
                method_label = "FSNet" if method == "fsnet" else "OneNet"
                ax.set_title(f"{dataset} · {method_label} · H={horizon}", pad=10)
                target_phases = labels.annotation_phase.iloc[first:last + 1]
                audit.append({"dataset": dataset, "method": method, "horizon": horizon,
                              "first_target_raw_index": first, "last_first_target_raw_index": last,
                              "online_steps": len(reference), "rolling_window": window,
                              "drift_points_shown": len(event_steps),
                              **{f"first_target_{phase}_rows": int((target_phases == phase).sum()) for phase in PHASE_COLORS}})
        fig.legend(curve_handles, [policy_label(p) for p in policies], loc="upper center",
                   bbox_to_anchor=(.5, .995), ncol=4, frameon=False, columnspacing=2.4,
                   handlelength=2.4, fontsize=10)
        phase_handles = [Patch(facecolor=PHASE_COLORS[p], label=PHASE_LABELS[p]) for p in PHASE_COLORS]
        phase_handles.append(Line2D([], [], color="#613A42", marker="|", markersize=9,
                                    linestyle="none", label="漂移点"))
        fig.legend(handles=phase_handles, loc="upper center", bbox_to_anchor=(.5, .91 if rows == 1 else .965),
                   ncol=4, frameon=False, columnspacing=2.4, handlelength=1.6, fontsize=10)
        path = args.output / f"{method}_frequency_phases"
        fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
        plt.close(fig)
    pd.DataFrame(audit).to_csv(args.output / "frequency_phase_alignment.csv", index=False, encoding="utf-8-sig")
    print(f"Saved two figures (PNG + PDF); checked annotation alignment for {len(audit)} panels.")


if __name__ == "__main__":
    main()

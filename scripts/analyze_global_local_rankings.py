"""Compare full-run winners in selected intervals with original drift annotations."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


GROUP = ["dataset", "method", "horizon"]
BLUE, ORANGE = "#285B86", "#D27735"


def policy_label(policy):
    if policy == "native":
        return "Every 1 / LR x1"
    if policy in ["frozen", "no_grad"]:
        return {"frozen": "Frozen", "no_grad": "No grad"}[policy]
    period, rate = policy.removeprefix("every").split("_lr")
    return f"Every {period} / LR x{rate}"


def save_plot(fig, axes, path):
    for ax in np.asarray(axes).flat:
        ax.grid(axis="y", color="#DDE2E6", linewidth=.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#CDD3D8")
        ax.set_xlabel("Online steps")
        ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -.22))
    fig.tight_layout(rect=(0, 0, 1, .91) if fig.legends else (0, 0, 1, 1))
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/update-study-completed/update-study-main-4gpu"))
    parser.add_argument("--tables", type=Path, default=Path("reports/update_study_completed/tables"))
    parser.add_argument("--annotations", type=Path, default=Path("runs/real-drift-electricity-5pct-20261010"))
    parser.add_argument("--output", type=Path, default=Path("reports/update_study_completed/global_local"))
    args = parser.parse_args()
    figures = args.output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 10, "pdf.fonttype": 42, "lines.linewidth": 1.6})
    full = pd.read_csv(args.tables / "full_policy_comparison.csv")
    assert (full.groupby(GROUP).size() == 11).all() and (full.seeds == 3).all()
    for metric in ["mae", "mse"]:
        full[f"{metric}_rank"] = full.groupby(GROUP)[metric].rank(method="min").astype(int)
    full.sort_values(GROUP + ["mae_rank"]).to_csv(args.output / "global_rankings.csv", index=False, encoding="utf-8-sig")
    slide_rows = []
    for key, group in full.groupby(GROUP):
        dataset, method, horizon = key
        if horizon == 24:
            top = group.nsmallest(3, "mae")
            mse_best = group.loc[group.mse.idxmin()]
            slide_rows.append([f"{dataset}\n{'FSNet' if method == 'fsnet' else 'OneNet'} / H=24"]
                              + [f"{policy_label(row.policy)}\n{row.mae:.4f}" for row in top.itertuples()]
                              + [f"{policy_label(mse_best.policy)}\n{mse_best.mse:.4f}"])

    # A plain ranking table for slides; ranking is within each dataset/method/H group.
    fig, ax = plt.subplots(figsize=(14.2, 3.6))
    ax.axis("off")
    table = ax.table(cellText=slide_rows, colLabels=["Dataset / Method", "MAE #1", "MAE #2", "MAE #3", "MSE #1"],
                     cellLoc="center", loc="center", colWidths=[.17, .2075, .2075, .2075, .2075])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, col), cell in table.get_celld().items():
        cell.set_height(.14 if row == 0 else .20)
        cell.set_edgecolor("#DEE4EA")
        cell.set_linewidth(.6)
        cell.set_facecolor("#EAF0F6" if row == 0 else "#F7F9FB" if row % 2 == 0 else "white")
        if row == 0 or col == 1:
            cell.set_text_props(weight="bold", color="#24445F")
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(figures / f"00_global_ranking.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    cases = [("ETTh2", "onenet", 4537, 4704), ("Electricity", "onenet", 12000, 12450),
             ("ETTh2", "fsnet", 10585, 10752)]
    score_rows, interval_rows = [], []
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 5.5), squeeze=False)
    for ax, (dataset, method, start, end) in zip(axes.flat, cases):
        group = full[(full.dataset == dataset) & (full.method == method) & (full.horizon == 24)]
        global_policy = group.loc[group.mae.idxmin(), "policy"]
        gpu = "gpu1" if method == "fsnet" else "gpu3"
        annotation = pd.read_csv(args.annotations / dataset / "labeled_data.csv", usecols=["annotation_phase", "is_drift_point"])
        statistics, smoothed = {}, {}
        origins = None
        for policy in group.policy:
            seed_losses, seed_curves = [], []
            for seed in range(3):
                path = args.runs / gpu / dataset / f"{method}_h24_seed{seed}" / policy / "forecasts.csv"
                trajectory = pd.read_csv(path)
                frame = trajectory.iloc[start - 1:end]
                assert len(frame) == end - start + 1 and (np.diff(frame.origin_raw_index) == 1).all()
                if origins is None:
                    origins = frame.origin_raw_index.to_numpy()
                assert np.array_equal(origins, frame.origin_raw_index.to_numpy())
                seed_losses.append(frame[["mae", "mse"]].to_numpy())
                # Smooth the existing trajectory before slicing, preserving past context.
                seed_curves.append(trajectory.mae.rolling(24, min_periods=24).mean().iloc[start - 1:end].to_numpy())
            seed_losses = np.stack(seed_losses)
            statistics[policy] = seed_losses.mean(axis=1)
            smoothed[policy] = np.stack(seed_curves)
        scores = pd.DataFrame({policy: values.mean(axis=0) for policy, values in statistics.items()}, index=["mae", "mse"]).T
        scores["local_mae_rank"] = scores.mae.rank(method="min").astype(int)
        local_policy = scores.mae.idxmin()
        for policy, row in scores.iterrows():
            score_rows.append({"dataset": dataset, "method": method, "horizon": 24,
                               "start_step": start, "end_step": end, "policy": policy,
                               "mae": row.mae, "mse": row.mse, "local_mae_rank": int(row.local_mae_rank)})
        segment = annotation.annotation_phase.to_numpy()[origins]
        changes = np.r_[0, np.flatnonzero(segment[1:] != segment[:-1]) + 1, len(segment)]
        for left, right in zip(changes[:-1], changes[1:]):
            if segment[left] != "stable":
                color = "#F2B678" if segment[left] == "drift" else "#CBD2D9"
                ax.axvspan(start + left - .5, start + right - .5, color=color, alpha=.3, linewidth=0)
        points = annotation.is_drift_point.to_numpy(dtype=bool)[origins]
        for step in np.arange(start, end + 1)[points]:
            ax.axvline(step, color="#A85151", linestyle="--", linewidth=.8, alpha=.8)
        runner_up = group.nsmallest(2, "mae").iloc[1].policy
        selected_policies = list(dict.fromkeys([global_policy, local_policy, runner_up, "native"]))
        for policy in selected_policies:
            if policy == global_policy:
                color, label = BLUE, "Overall #1"
            elif policy == local_policy:
                color, label = ORANGE, "Interval #1"
            elif policy == "native":
                color = "#65737E"
                label = "Native (overall #2)" if policy == runner_up else "Native"
            else:
                color, label = "#9467A6", "Overall #2"
            values = smoothed[policy]
            steps = np.arange(start, end + 1)
            short_label = policy_label(policy).replace("Every ", "P")
            ax.plot(steps, values.mean(axis=0), color=color, label=f"{label} | {short_label}",
                    linewidth=1.7 if policy in [global_policy, local_policy] else 1.2)
            if policy in [global_policy, local_policy]:
                ax.fill_between(steps, values.min(axis=0), values.max(axis=0), color=color, alpha=.09, linewidth=0)
        seed_gain = 100 * (1 - statistics[local_policy] / statistics[global_policy])
        assert (seed_gain[:, 0] > 0).all()
        indices = origins[:, None] + 1 + np.arange(24)
        phases = annotation.annotation_phase.to_numpy()
        assert indices.max() < len(phases)
        interval_rows.append({"dataset": dataset, "method": method, "horizon": 24, "start_step": start,
                           "end_step": end, "global_policy": global_policy, "local_policy": local_policy,
                           "global_policy_local_rank": int(scores.loc[global_policy, "local_mae_rank"]),
                           "global_mae": scores.loc[global_policy, "mae"], "local_mae": scores.loc[local_policy, "mae"],
                           "global_mse": scores.loc[global_policy, "mse"], "local_mse": scores.loc[local_policy, "mse"],
                           "raw_origin_start": origins[0], "raw_origin_end": origins[-1],
                           "drift_points": int(points.sum()),
                           "mae_gain_pct": 100 * (1 - statistics[local_policy][:, 0].mean() / statistics[global_policy][:, 0].mean()),
                           "mse_gain_pct": 100 * (1 - statistics[local_policy][:, 1].mean() / statistics[global_policy][:, 1].mean()),
                           "min_seed_mae_gain_pct": seed_gain[:, 0].min(),
                           "max_seed_mae_gain_pct": seed_gain[:, 0].max(),
                           **{f"target_{phase}_fraction": np.mean(phases[indices] == phase) for phase in ["stable", "transition", "drift"]}})
        ax.set_title(f"{dataset} / {'FSNet' if method == 'fsnet' else 'OneNet'} / H=24")
        ax.set_ylabel("MAE (24-step moving average)")
        ax.set_xlim(start, end)
        ax.set_xticks([start, (start + end) // 2, end])
    fig.legend(handles=[Patch(facecolor="#CBD2D9", alpha=.3, label="Transition"),
                        Patch(facecolor="#F2B678", alpha=.3, label="Drift"),
                        Line2D([0], [0], color="#A85151", linestyle="--", linewidth=.8, label="Drift event start")],
               loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, 1.01), fontsize=10)
    save_plot(fig, axes, figures / "01_drift_intervals")
    pd.DataFrame(score_rows).to_csv(args.output / "interval_scores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(interval_rows).to_csv(args.output / "selected_intervals.csv", index=False, encoding="utf-8-sig")
    print(f"Saved overall ranking table and 3 drift-annotated interval comparisons to {args.output}")


if __name__ == "__main__":
    main()

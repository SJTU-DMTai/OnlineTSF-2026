"""Analyze continuous policy replay and historical paired branches without torch."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from update_policy_common import file_sha256, select_actions


CONTROLLERS = ["native", "policy_full", "policy_data"]
LABELS = {"native": "Native",
          "policy_full": "Full policy", "policy_data": "Data policy"}
COLORS = {"native": "#363B43", "policy_full": "#0072B2", "policy_data": "#D55E00"}


def finish_figure(fig, axes, path):
    for ax in np.asarray(axes).ravel():
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#C7CDD1")
        ax.grid(axis="y", color="#E4E8EB", linewidth=.6)
        ax.tick_params(length=3, color="#8D969D")
    fig.savefig(path.with_suffix(".png"), dpi=190, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def shade_phases(ax, phases, first, last):
    """Shade forecast-origin phases; transitions can extend beyond event cores."""
    selected = phases[first:last]
    edges = np.r_[0, np.flatnonzero(selected[1:] != selected[:-1]) + 1, len(selected)]
    for left, right in zip(edges[:-1], edges[1:]):
        phase = selected[left]
        if phase in {"drift", "transition"}:
            ax.axvspan(left + 1, right + 1, color="#A5A5B2" if phase == "drift" else "#E5DED1",
                       alpha=.32, linewidth=0, zorder=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(
        "runs/policy-exp2-training16-seed1/runs/policy-exp2-training16/seed1"))
    parser.add_argument("--annotations", type=Path, default=Path("runs/real-drift-electricity-5pct-20261010"))
    parser.add_argument("--output", type=Path, default=Path("runs/policy-exp2-training16-seed1/analysis"))
    args = parser.parse_args()
    p, output = args.results, args.output
    manifest = json.loads((p / "manifest.json").read_text(encoding="utf-8"))
    selectors = json.loads((p / "selectors.json").read_text(encoding="utf-8"))
    annotation_dir = args.annotations / manifest["dataset"]
    assert manifest["status"] == "complete"
    assert file_sha256(annotation_dir / "labeled_data.csv") == manifest["input_sha256"]
    assert file_sha256(p / "selectors.json") == manifest["selector_sha256"]
    decisions = pd.read_csv(p / "decisions.csv")
    forecasts = {c: pd.read_csv(p / c / "forecasts.csv") for c in CONTROLLERS}
    summary = pd.read_csv(p / "summary.csv")
    summary = summary[summary.controller.isin(CONTROLLERS)].copy()
    native_seconds = summary.loc[summary.controller.eq("native"), "seconds"].iloc[0]
    summary["time_reduction_pct"] = 100 * (1 - summary.seconds / native_seconds)
    training = pd.read_csv(p / "training_states.csv")
    losses = pd.read_csv(p / "training_losses.csv")
    assert training[["seed", "probe_id"]].equals(losses[["seed", "probe_id"]])
    assert (training.clock + manifest["decision_interval"] + manifest["horizon"] <= manifest["test_start"]).all()
    assert training.seed.eq(manifest["test_seed"]).all()
    phases = pd.read_csv(annotation_dir / "labeled_data.csv", usecols=["annotation_phase"]).annotation_phase.to_numpy()
    events = pd.read_csv(annotation_dir / "drift_points.csv")
    first, size, interval = manifest["test_start"], manifest["steps"], manifest["decision_interval"]
    expected = np.arange(first, first + size)
    origin_phases = phases[expected]
    full_decisions = decisions[decisions.controller.eq("policy_full")].sort_values("clock")
    data_decisions = decisions[decisions.controller.eq("policy_data")].sort_values("clock")
    np.testing.assert_allclose(full_decisions[["channel_shift_fraction", "volatility_ratio"]],
                               data_decisions[["channel_shift_fraction", "volatility_ratio"]], rtol=1e-12)
    block_rows, phase_rows, action_rows, update_rows = [], [], [], []
    for c, frame in forecasts.items():
        assert np.array_equal(frame.origin_raw_index, expected)
        np.testing.assert_allclose(frame[["mae", "mse"]].mean(),
                                   summary.loc[summary.controller.eq(c), ["mae", "mse"]].iloc[0], rtol=1e-7)
        selected = decisions[decisions.controller.eq(c)].sort_values("clock")
        assert np.array_equal(selected.clock, expected[::interval])
        if c.startswith("policy_"):
            chosen, scores = select_actions(selectors["selectors"][c], selected)
            assert chosen == selected.selected_action.tolist()
            saved_scores = np.array([[json.loads(row)[a] for a in selectors["selectors"][c]["actions"]]
                                     for row in selected.predicted_gains])
            np.testing.assert_allclose(scores, saved_scores, rtol=1e-12, atol=1e-12)
        for block, offset in enumerate(range(0, size, interval)):
            end = min(offset + interval, size)
            record = selected.iloc[block]
            a, n = frame.iloc[offset:end], forecasts["native"].iloc[offset:end]
            block_rows.append({**record.to_dict(), "block": block, "steps": offset + 1,
                               "block_steps": end - offset, "complete": end - offset == interval,
                               "mae": a.mae.mean(), "mse": a.mse.mean(),
                               "native_mae": n.mae.mean(), "native_mse": n.mse.mean(),
                               "mae_gain_pct": 100 * (1 - a.mae.mean() / n.mae.mean()),
                               "mse_gain_pct": 100 * (1 - a.mse.mean() / n.mse.mean()),
                               "mae_saved_sum": (n.mae - a.mae).sum(),
                               "drift_origin_rows": int((origin_phases[offset:end] == "drift").sum()),
                               "transition_origin_rows": int((origin_phases[offset:end] == "transition").sum())})
        for phase in ["stable", "transition", "drift"]:
            mask = origin_phases == phase
            a, n = frame.loc[mask], forecasts["native"].loc[mask]
            phase_rows.append({"controller": c, "origin_phase": phase, "forecasts": int(mask.sum()),
                               "mae": a.mae.mean(), "mse": a.mse.mean(),
                               "mae_gain_pct": 100 * (1 - a.mae.mean() / n.mae.mean()),
                               "mse_gain_pct": 100 * (1 - a.mse.mean() / n.mse.mean()),
                               "mae_saved_sum": (n.mae - a.mae).sum()})
        updates = pd.read_csv(p / c / "updates.csv")
        groups = updates.groups.fillna("")
        row = {"controller": c, "feedback_rows": len(updates), "gradient_feedback_rows": int(groups.ne("").sum())}
        for group in ["time", "variable", "decision"]:
            row[group + "_updated_rows"] = int(groups.str.split("+").apply(list.__contains__, args=(group,)).sum())
        update_rows.append(row)
    blocks = pd.DataFrame(block_rows)
    full_blocks = blocks[blocks.complete]
    for (c, action), group in full_blocks.groupby(["controller", "selected_action"]):
        action_rows.append({"controller": c, "action": action, "blocks": len(group),
                            "mae_gain_pct": 100 * (1 - group.mae.mean() / group.native_mae.mean()),
                            "mae_win_fraction": float(group.mae.lt(group.native_mae).mean()),
                            "mae_saved_sum": group.mae_saved_sum.sum(),
                            "expert_gap_mean": group.expert_relative_gap.mean(),
                            "error_ratio_mean": group.error_ratio.mean()})
    within_rows = []
    complete_count = size // interval
    for c in CONTROLLERS:
        a = forecasts[c][["mae", "mse"]].iloc[:complete_count * interval].to_numpy().reshape(complete_count, interval, 2)
        n = forecasts["native"][["mae", "mse"]].iloc[:complete_count * interval].to_numpy().reshape(complete_count, interval, 2)
        # Same complete blocks at every offset: no changing sample population.
        cumulative = a.cumsum(axis=1).mean(axis=0) / np.arange(1, interval + 1)[:, None]
        native_cumulative = n.cumsum(axis=1).mean(axis=0) / np.arange(1, interval + 1)[:, None]
        for index in range(interval):
            within_rows.append({"controller": c, "steps_after_decision": index + 1, "blocks": complete_count,
                                "mae": cumulative[index, 0], "mse": cumulative[index, 1],
                                "mae_gain_pct": 100 * (1 - cumulative[index, 0] / native_cumulative[index, 0]),
                                "mse_gain_pct": 100 * (1 - cumulative[index, 1] / native_cumulative[index, 1])})
    within = pd.DataFrame(within_rows)
    phase_table = pd.DataFrame(phase_rows)
    update_table = pd.DataFrame(update_rows)
    # Define cases once on the shared time grid, then evaluate every controller
    # on those exact blocks. Phase cases use the entire 128-step block.
    reference = full_blocks[full_blocks.controller.eq("native")].sort_values("block")
    changing = reference.drift_origin_rows.add(reference.transition_origin_rows).gt(0).to_numpy()
    cases = [("all", np.ones(complete_count, dtype=bool)),
             ("stable_only", ~changing), ("drift_or_transition", changing),
             ("initial_384", np.arange(complete_count) < 3),
             ("following", np.arange(complete_count) >= 3)]
    window_rows = []
    for case, mask in cases:
        for c in CONTROLLERS:
            errors = forecasts[c][["mae", "mse"]].iloc[:complete_count * interval].to_numpy().reshape(complete_count, interval, 2)
            for lookahead in [8, 32, 128]:
                mean = errors[mask, :lookahead].mean(axis=(0, 1))
                window_rows.append({"case": case, "controller": c, "lookahead": lookahead,
                                    "blocks": int(mask.sum()), "forecasts": int(mask.sum()) * lookahead,
                                    "mae": mean[0], "mse": mean[1]})
    windows = pd.DataFrame(window_rows)
    for c in CONTROLLERS:
        a = windows[(windows.case.eq("all")) & windows.controller.eq(c)].sort_values("lookahead")
        b = within[within.controller.eq(c) & within.steps_after_decision.isin([8, 32, 128])].sort_values("steps_after_decision")
        np.testing.assert_allclose(a[["mae", "mse"]], b[["mae", "mse"]], rtol=1e-12)
    names = [a["name"] for a in selectors["actions"]]
    gains = (1 - losses[names].div(losses.native, axis=0)) * 100
    # Use the serialized thresholds, rather than rounded export_text thresholds.
    tree = selectors["selectors"]["policy_full"]["trees"]["only_variable+decision"]
    low_error_node = tree["left"][0]
    ratio_limit, gap_limit = tree["threshold"][0], tree["threshold"][low_error_node]
    assert selectors["selectors"]["policy_full"]["features"][tree["feature"][0]] == "error_ratio"
    assert selectors["selectors"]["policy_full"]["features"][tree["feature"][low_error_node]] == "expert_relative_gap"
    preferred = training.error_ratio.le(ratio_limit) & training.expert_relative_gap.le(gap_limit)
    condition_rows = []
    for label, mask in [("time_better_error_controlled", preferred), ("other_states", ~preferred)]:
        for action in ["only_variable+decision", "only_variable", "only_time+variable", "every16_lr0.3", "every16_lr1"]:
            condition_rows.append({"condition": label, "action": action, "states": int(mask.sum()),
                                   "mean_relative_mae_gain_pct": gains.loc[mask, action].mean(),
                                   "pooled_mae_gain_pct": 100 * (1 - losses.loc[mask, action].mean() / losses.loc[mask, "native"].mean())})
    part_rows = []
    for start, end in [(0, 384), (384, size), (0, size)]:
        for c in CONTROLLERS:
            a, n = forecasts[c].iloc[start:end], forecasts["native"].iloc[start:end]
            part_rows.append({"controller": c, "start_step": start + 1, "end_step_inclusive": end,
                              "forecasts": end - start, "mae": a.mae.mean(), "mse": a.mse.mean(),
                              "mae_gain_pct": 100 * (1 - a.mae.mean() / n.mae.mean()),
                              "mse_gain_pct": 100 * (1 - a.mse.mean() / n.mse.mean())})
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    for filename, table in [("decision_blocks", blocks), ("origin_phase_metrics", phase_table),
                            ("action_outcomes", pd.DataFrame(action_rows)), ("update_counts", update_table),
                            ("within_block_gains", within), ("historical_conditions", pd.DataFrame(condition_rows)),
                            ("interval_metrics", pd.DataFrame(part_rows)), ("overall_metrics", summary),
                            ("decision_window_metrics", windows)]:
        table.to_csv(output / (filename + ".csv"), index=False)
    table_text = []
    for case, mask in cases:
        table_text.append(f"{case}: {int(mask.sum())} blocks\n\n"
                          "每格为 MAE / MSE，粗体表示同一评价步数下该指标的最低值。\n\n"
                          "| 策略 | 前 8 步 | 前 32 步 | 前 128 步 |\n"
                          "|---|---:|---:|---:|")
        selected_window = windows[windows.case.eq(case)].set_index(["controller", "lookahead"])
        minima = windows[windows.case.eq(case)].groupby("lookahead")[["mae", "mse"]].min()
        for c in CONTROLLERS:
            cells = []
            for lookahead in [8, 32, 128]:
                values = []
                for metric in ["mae", "mse"]:
                    value = selected_window.loc[(c, lookahead), metric]
                    formatted = f"{value:.5f}"
                    values.append(f"**{formatted}**" if value == minima.loc[lookahead, metric] else formatted)
                cells.append(" / ".join(values))
            table_text.append(f"| {LABELS[c]} | " + " | ".join(cells) + " |")
        table_text.append("")
    (output / "decision_window_tables.md").write_text("\n".join(table_text), encoding="utf-8")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelsize": 10, "axes.titlesize": 11, "legend.fontsize": 9,
                         "pdf.fonttype": 42, "axes.axisbelow": True})
    steps = np.arange(1, size + 1)
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True, layout="constrained")
    for ax, metric in zip(axes, ["mae", "mse"]):
        for c in CONTROLLERS:
            ax.plot(steps, forecasts[c][metric].cumsum() / steps, color=COLORS[c], linewidth=1.5,
                    linestyle="--" if c == "native" else "-", label=LABELS[c])
        ax.set_ylabel("Cumulative " + metric.upper())
        ax.set_xlim(1, size)
    axes[0].legend(frameon=False, ncol=3)
    axes[1].set_xlabel("Steps")
    finish_figure(fig, axes, figures / "00_overall_errors")
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True, layout="constrained")
    for column, c in enumerate(["policy_full", "policy_data"]):
        for row, metric in enumerate(["mae", "mse"]):
            ax = axes[row, column]
            shade_phases(ax, phases, first, first + size)
            for name in ["native", c]:
                ax.plot(steps, forecasts[name][metric].rolling(64, min_periods=64).mean(),
                        color=COLORS[name], linewidth=1.3, linestyle="--" if name == "native" else "-",
                        label=LABELS[name])
            ax.set_ylabel(metric.upper() + " (64-step mean)")
            ax.set_xlim(1, size)
            if row == 0:
                ax.set_title(LABELS[c])
                ax.legend(frameon=False, loc="upper right")
            else:
                ax.set_xlabel("Steps")
    finish_figure(fig, axes, figures / "01_local_errors")

    fig, axes = plt.subplots(2, 1, figsize=(11, 5.8), sharex=True, layout="constrained")
    for ax, metric in zip(axes, ["mae", "mse"]):
        for c in CONTROLLERS:
            a = blocks[blocks.controller.eq(c)]
            ax.plot(a.steps, a[metric], marker="o", markersize=3, linewidth=1.3,
                    linestyle="--" if c == "native" else "-", color=COLORS[c], label=LABELS[c])
        ax.set_xlim(1, size)
        ax.set_ylabel("Block " + metric.upper())
    axes[0].legend(frameon=False, ncol=3)
    axes[1].set_xlabel("Steps at decision")
    finish_figure(fig, axes, figures / "02_decision_metrics")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), layout="constrained")
    for ax, metric in zip(axes, ["mae", "mse"]):
        for c in CONTROLLERS:
            a = within[within.controller.eq(c)]
            ax.plot(a.steps_after_decision, a[metric], color=COLORS[c],
                    linewidth=1.6, linestyle="--" if c == "native" else "-", label=LABELS[c])
        ax.set_xlim(1, interval)
        ax.set_xticks([1, 8, 24, 32, 64, 96, 128])
        ax.set_xlabel("Steps after decision")
        ax.set_ylabel("Mean " + metric.upper())
    axes[0].legend(frameon=False)
    finish_figure(fig, axes, figures / "03_response_windows")

    fig, axes = plt.subplots(4, 1, figsize=(12, 8.4), sharex=True, layout="constrained",
                             gridspec_kw={"height_ratios": [1.4, 1, 1, 1]})
    selected = decisions[decisions.controller.eq("policy_full")].copy()
    selected["steps"] = selected.clock - first + 1
    action_names = ["native", "only_time+variable", "only_variable+decision", "every4_lr0.3", "every16_lr1", "every16_lr0.3"]
    ticks = dict(zip(action_names, range(len(action_names))))
    axes[0].step(np.r_[selected.steps, size], np.r_[selected.selected_action.map(ticks), selected.selected_action.map(ticks).iloc[-1]],
                 where="post", color=COLORS["policy_full"], linewidth=1.4)
    axes[0].set_yticks(range(len(action_names)), action_names)
    axes[0].set_ylabel("Full policy")
    axes[1].plot(selected.steps, selected.volatility_ratio, color="#64748B", marker=".", linewidth=1)
    axes[1].axhline(1, color="#939BA1", linewidth=.7, linestyle="--")
    axes[1].set_ylabel("Volatility ratio")
    for c in ["native", "policy_full"]:
        a = decisions[decisions.controller.eq(c)]
        axes[2].plot(a.clock - first + 1, a.expert_relative_gap, color=COLORS[c], linewidth=1.1, marker=".", label=LABELS[c])
    axes[2].axhline(0, color="#939BA1", linewidth=.7)
    axes[2].set_ylabel("Expert MAE gap")
    axes[2].legend(frameon=False, ncol=2, loc="lower right")
    axes[3].plot(selected.steps, selected.error_ratio, color=COLORS["policy_full"], marker=".", linewidth=1)
    axes[3].axhline(ratio_limit, color="#939BA1", linewidth=.7, linestyle="--")
    axes[3].set_ylabel("MAE32 / MAE128")
    axes[3].set_xlabel("Steps")
    for ax in axes:
        shade_phases(ax, phases, first, first + size)
        ax.set_xlim(1, size)
    finish_figure(fig, axes, figures / "04_actions_and_states")

    fig, axes = plt.subplots(2, 2, figsize=(12, 5.8), layout="constrained")
    for col, (left, right) in enumerate([(1, 512), (3073, 3456)]):
        for c in ["native", "policy_full", "policy_data"]:
            axes[0, col].plot(steps[left - 1:right], forecasts[c].mae.rolling(32, min_periods=32).mean().iloc[left - 1:right],
                              color=COLORS[c], label=LABELS[c], linewidth=1.4)
        for c in ["native", "policy_full"]:
            for expert, style in [("time", "-"), ("variable", "--")]:
                values = forecasts[c][expert + "_expert_mae"].rolling(32, min_periods=32).mean()
                axes[1, col].plot(steps[left - 1:right], values.iloc[left - 1:right],
                                 color=COLORS[c], linestyle=style, linewidth=1.3,
                                 label=LABELS[c] + " / " + expert)
        for row in range(2):
            ax = axes[row, col]
            for event in events.itertuples():
                start, end = event.transition_start_index - first + 1, event.transition_end_index_exclusive - first + 1
                if end >= left and start <= right:
                    ax.axvspan(start, end, color="#E5DED1", alpha=.4, linewidth=0)
                    ax.axvspan(event.start_index - first + 1, event.end_index_exclusive - first + 1,
                               color="#A5A5B2", alpha=.4, linewidth=0)
            for clock in decisions[decisions.controller.eq("policy_full")].clock:
                x = clock - first + 1
                if left <= x <= right:
                    ax.axvline(x, linewidth=.65, color="#C0C7CC", linestyle=":")
            ax.set_xlim(left, right)
            ax.set_ylabel(("Forecast" if row == 0 else "Expert") + " MAE (32-step mean)")
            if row == 1:
                ax.set_xlabel("Steps")
        axes[0, col].set_title("Initial interval" if col == 0 else "Steps 3073–3456")
    axes[0, 0].legend(frameon=False, ncol=3, loc="upper right")
    axes[1, 0].legend(frameon=False, ncol=2, fontsize=8, loc="upper right")
    finish_figure(fig, axes, figures / "05_expert_cases")

    fig, axes = plt.subplots(2, 1, figsize=(11, 5.8), sharex=True, layout="constrained")
    for c, color in [("native", COLORS["native"]), ("only_variable+decision", COLORS["policy_full"])]:
        axes[0].plot(training.clock, losses[c], marker="o", markersize=4, color=color, linewidth=1.3, label=c)
    axes[0].set_ylabel("Branch MAE")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].plot(training.clock, training.error_ratio, marker="o", color="#64748B", linewidth=1.2, label="MAE32 / MAE128")
    axes[1].axhline(ratio_limit, color="#939BA1", linestyle="--", linewidth=.8)
    axes[1].set_ylabel("Error ratio")
    axes[1].set_xlabel("Raw steps (training interval)")
    axes[1].legend(frameon=False, loc="upper right")
    for ax in axes:
        for clock in training.loc[preferred, "clock"]:
            ax.axvspan(clock - 75, clock + 75, color="#B7D9D7", alpha=.4, linewidth=0)
    finish_figure(fig, axes, figures / "06_historical_branches")
    metadata = {"source_results": str(p.resolve()), "test_start": first, "steps": size,
                "decision_interval": interval, "complete_blocks": complete_count, "partial_block_steps": size % interval,
                "test_events": int(events.detection_index.between(first, first + size - 1).sum()),
                "training_rule_error_ratio_threshold": ratio_limit, "training_rule_expert_gap_threshold": gap_limit,
                "checks": ["input and selector hashes", "same forecast origins", "summary reproduced from forecasts",
                           "historical targets strictly before test start", "saved actions and scores reproduced",
                           "same blocks at every response offset"],
                "phase_definition": "forecast origin phase; targets may cross phases",
                "block_comparison": "continuous trajectories with their own accumulated states; historical branches share state",
                "figure_shading": {"light": "transition", "dark": "drift"}}
    (output / "analysis_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary[["controller", "mae", "mse", "gradient_steps", "seconds", "time_reduction_pct"]].round(6).to_string(index=False))
    print("Decision window metrics:")
    print(windows.round(5).to_string(index=False))
    print("Saved analysis:", output.resolve())


if __name__ == "__main__":
    main()

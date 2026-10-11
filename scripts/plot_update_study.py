"""Audit completed CSV logs and draw update-study figures without importing PyTorch.

Run from the repository root; all source results remain read-only.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import numpy as np
import pandas as pd


GROUP = ["dataset", "method", "horizon"]
JOB = GROUP + ["seed"]
PAIR = JOB + ["probe_id", "lookahead"]
DATASETS = ["ETTh2", "ETTm1", "Electricity"]
COLORS = ["#2878B5", "#E07A39", "#40967B", "#9B65A6", "#6E7785"]
LABELS = {"native": "Every 1 | LR x1", "frozen": "Frozen", "no_grad": "No gradient step"}
FEATURES = ["mean_shift_32", "mean_shift_128", "input_std_32", "input_trend_32",
            "autocorrelation_32", "cross_correlation", "mae_32", "mae_128",
            "current_feedback_mae", "residual_bias", "expert_disagreement"]


def policy_label(policy):
    if policy in LABELS:
        return LABELS[policy]
    if policy.startswith("only_"):
        return policy[5:].replace("+", " + ").title()
    period, rate = policy.removeprefix("every").split("_lr")
    return f"Every {period} | LR x{rate}"


def save_figure(fig, axes, path, ylabel, xlabel="Online steps"):
    for ax in np.asarray(axes).flat:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#DCE1E5", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#CBD1D6")
        ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.margins(x=0)
    handles, labels = np.asarray(axes).flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False,
                   bbox_to_anchor=(0.5, 1.005))
    fig.tight_layout(rect=(0, 0, 1, 0.94 if fig.get_figheight() > 6 else 0.87))
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def draw_curve(ax, values, color, label, *, window=1, cumulative=False, band=True):
    """Rows are independent seeds; bands show their range, not a confidence interval."""
    if cumulative:
        values = values.cumsum(axis=1) / np.arange(1, values.shape[1] + 1)
    elif window > 1:
        values = pd.DataFrame(values.T).rolling(window, min_periods=window).mean().to_numpy().T
    indices = np.unique(np.r_[np.arange(0, values.shape[1], max(1, values.shape[1] // 1800)), values.shape[1] - 1])
    x = indices + 1
    ax.plot(x, values.mean(axis=0)[indices], color=color, label=label, linewidth=1.25)
    if band:
        ax.fill_between(x, values.min(axis=0)[indices], values.max(axis=0)[indices], color=color, alpha=0.09, linewidth=0)


def temporal_rule_check(paired, states):
    """A small chronological diagnostic: fit a decision stump, then evaluate later probes.

    All seeds of a probe stay together; training targets cannot overlap test origins.
    This is offline policy evaluation on native-state forks, not a deployed controller.
    """
    results = []
    for key, frame in paired[paired.lookahead == 128].groupby(GROUP):
        features = states.set_index(JOB + ["probe_id"])
        wide = frame.pivot(index=JOB + ["probe_id"], columns="policy", values="mae").join(
            features[["clock"] + FEATURES]).reset_index()
        clocks = sorted(wide.clock.unique())
        cutoff = clocks[len(clocks) // 2]
        train = wide[wide.clock + 128 + key[2] < cutoff]
        test = wide[wide.clock >= cutoff]
        actions = ["native", "every1_lr0.3", "every4_lr0.3", "no_grad"]
        constant = train[actions].mean().idxmin()
        best = (train[constant].mean(), "constant", np.nan, constant, constant)
        for feature in FEATURES:
            values = sorted(train[feature].dropna().unique())
            for threshold in (np.asarray(values[:-1]) + np.asarray(values[1:])) / 2:
                left = train[feature] <= threshold
                if min(train[left].clock.nunique(), train[~left].clock.nunique()) < 2:
                    continue
                a, b = train.loc[left, actions].mean().idxmin(), train.loc[~left, actions].mean().idxmin()
                loss = (train.loc[left, a].sum() + train.loc[~left, b].sum()) / len(train)
                if loss < best[0]:
                    best = (loss, feature, threshold, a, b)
        _, feature, threshold, a, b = best
        decisions = np.full(len(test), a, dtype=object) if feature == "constant" else np.where(test[feature] <= threshold, a, b)
        selected = np.array([test.iloc[i][action] for i, action in enumerate(decisions)])
        results.append(dict(zip(GROUP, key)) | {
            "train_probes": train.clock.nunique(), "test_probes": test.clock.nunique(),
            "feature": feature, "threshold": threshold, "left_action": a, "right_action": b,
            "train_best_constant": constant, "test_native_mae": test.native.mean(),
            "test_constant_mae": test[constant].mean(), "test_rule_mae": selected.mean(),
            "test_rule_gain_pct": 100 * (1 - selected.mean() / test.native.mean()),
            "test_constant_gain_pct": 100 * (1 - test[constant].mean() / test.native.mean()),
            "test_oracle_subset_mae": test[actions].min(axis=1).mean()})
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/update_study_completed"))
    args = parser.parse_args()
    figures, tables = args.output / "figures", args.output / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 11, "axes.labelsize": 10, "pdf.fonttype": 42})
    summaries, outcomes, states, audit, phases, tails, experts, blocks = [], [], [], [], [], [], [], []
    full, branches = {}, {}
    for marker in sorted(args.input.rglob("policies.json")):
        directory = marker.parent
        summary = pd.read_csv(directory / "summary.csv")
        outcome = pd.read_csv(directory / "probe_outcomes.csv")
        state = pd.read_csv(directory / "probe_states.csv")
        summaries.append(summary)
        outcomes.append(outcome)
        states.append(state)
        identity = {column: summary.iloc[0][column] for column in JOB}
        key = tuple(identity[column] for column in JOB)
        print(f"Reading {directory.relative_to(args.input)}", flush=True)
        reference_indices = None
        assert len(summary) == 11 and len(state) == 12 and len(outcome) == 612, directory
        assert not outcome.duplicated(PAIR + ["policy"]).any(), directory
        assert state.probe_id.nunique() == 12, directory
        assert len(json.loads(marker.read_text(encoding="utf-8"))) == 17, directory
        for row in summary.to_dict("records"):
            policy = row["policy"]
            frame = pd.read_csv(directory / policy / "forecasts.csv")
            assert np.isfinite(frame[["mae", "mse"]]).all().all(), directory
            assert len(frame) == row["forecasts"], directory
            assert (np.diff(frame.forecast_index) == 1).all(), directory
            assert (frame.target_start_index == frame.origin_raw_index + 1).all(), directory
            assert (frame.target_end_index_exclusive == frame.target_start_index + key[2]).all(), directory
            if reference_indices is None:
                reference_indices = frame.forecast_index.to_numpy()
            assert np.array_equal(frame.forecast_index, reference_indices), directory
            delta = max(abs(frame.mae.mean() - row["mae"]), abs(frame.mse.mean() - row["mse"]))
            assert np.isclose(frame.mae.mean(), row["mae"], rtol=1e-6), directory
            assert np.isclose(frame.mse.mean(), row["mse"], rtol=1e-6), directory
            full[key + (policy,)] = frame
            width = 672 if key[0] == "ETTm1" else 168
            for start in range(0, len(frame), width):
                block = frame.iloc[start:start + width]
                blocks.append(identity | {"policy": policy, "start_step": start + 1,
                              "end_step": start + len(block), "windows": len(block),
                              "mae": block.mae.mean(), "mse": block.mse.mean()})
            phase = pd.read_csv(directory / policy / "phase_metrics.csv")
            phases.append(phase.assign(**identity, policy=policy))
            updates = pd.read_csv(directory / policy / "updates.csv")
            for counter in ["gradient_steps", "egd_steps"]:
                assert (updates[counter].iloc[-1] if len(updates) else 0) == row[counter], directory
            tails.append(identity | {"policy": policy, "mae_p95": frame.mae.quantile(.95),
                         "mae_p99": frame.mae.quantile(.99), "mae_max": frame.mae.max(),
                         "mse_top1pct_share": frame.mse.nlargest(int(np.ceil(len(frame) * .01))).sum() / frame.mse.sum()})
            audit.append(identity | {"policy": policy, "forecasts": len(frame),
                         "inferred_prefix_rows": int(frame.target_start_index.iloc[0]),
                         "summary_max_abs_difference": delta})
            if key[1] == "onenet" and policy in ["native", "no_grad"]:
                experts.append(identity | {"policy": policy, "mixture_mae": frame.mae.mean(),
                               "time_mae": frame.time_expert_mae.mean(), "variable_mae": frame.variable_expert_mae.mean(),
                               "window_oracle_mae": frame[["time_expert_mae", "variable_expert_mae"]].min(axis=1).mean(),
                               "mixture_worse_than_both_fraction": (frame.mae > frame[["time_expert_mae", "variable_expert_mae"]].max(axis=1) + 1e-6).mean()})
        max_replay_delta = 0.0
        for probe in state.to_dict("records"):
            block = outcome[outcome.probe_id == probe["probe_id"]]
            for policy, records in block.groupby("policy"):
                frame = pd.read_csv(directory / "probes" / probe["probe_id"] / policy / "forecasts.csv")
                assert len(frame) == 128 and frame.forecast_index.iloc[0] == probe["index"], directory
                assert (np.diff(frame.forecast_index) == 1).all(), directory
                assert np.isfinite(frame[["mae", "mse"]]).all().all(), directory
                for record in records.to_dict("records"):
                    for metric in ["mae", "mse"]:
                        assert np.isclose(frame[metric].iloc[:record["lookahead"]].mean(), record[metric], rtol=1e-6), directory
                branches[key + (probe["probe_id"], policy)] = frame.mae.to_numpy()
                if policy == "native":
                    original = full[key + ("native",)].set_index("forecast_index").loc[frame.forecast_index]
                    max_replay_delta = max(max_replay_delta, np.abs(original.mae.to_numpy() - frame.mae.to_numpy()).max())
        audit[-1]["native_branch_max_abs_difference"] = max_replay_delta

    summary = pd.concat(summaries, ignore_index=True)
    outcome = pd.concat(outcomes, ignore_index=True)
    state = pd.concat(states, ignore_index=True)
    assert not summary.duplicated(JOB + ["policy"]).any()
    paired = outcome.merge(outcome[outcome.policy == "native"][PAIR + ["mae"]].rename(columns={"mae": "native_mae"}), on=PAIR, validate="many_to_one")
    paired["benefit"] = paired.native_mae - paired.mae
    paired["relative_benefit_pct"] = 100 * paired.benefit / paired.native_mae
    aggregate = summary.groupby(GROUP + ["policy"]).agg(
        mae=("mae", "mean"), mae_sd=("mae", "std"), mse=("mse", "mean"),
        seconds=("seconds", "mean"), gradient_steps=("gradient_steps", "mean"), seeds=("seed", "nunique")).reset_index()
    aggregate = aggregate.merge(aggregate[aggregate.policy == "native"][GROUP + ["mae", "seconds"]].rename(
        columns={"mae": "native_mae", "seconds": "native_seconds"}), on=GROUP)
    aggregate["gain_pct"] = 100 * (1 - aggregate.mae / aggregate.native_mae)
    aggregate["time_ratio"] = aggregate.seconds / aggregate.native_seconds
    branch_stats = paired.groupby(GROUP + ["lookahead", "policy"]).agg(
        mae=("mae", "mean"), native_mae=("native_mae", "mean"), benefit=("benefit", "mean"),
        win_fraction=("benefit", lambda v: (v > 1e-8).mean())).reset_index()
    branch_stats["gain_pct"] = 100 * branch_stats.benefit / branch_stats.native_mae
    phase_stats = paired.groupby(GROUP + ["lookahead", "kind", "policy"]).agg(
        mae=("mae", "mean"), native_mae=("native_mae", "mean"), benefit=("benefit", "mean"),
        probes=("probe_id", "nunique")).reset_index()
    phase_stats["gain_pct"] = 100 * phase_stats.benefit / phase_stats.native_mae
    oracle_rows = []
    for key, frame in paired.groupby(GROUP + ["lookahead"]):
        wide = frame.pivot(index=["seed", "probe_id"], columns="policy", values="mae")
        best_constant = wide.mean().idxmin()
        oracle_rows.append(dict(zip(GROUP + ["lookahead"], key)) | {
            "native_mae": wide.native.mean(), "best_constant_policy": best_constant,
            "best_constant_mae": wide[best_constant].mean(), "oracle_mae": wide.min(axis=1).mean(),
            "oracle_gain_pct": 100 * (1 - wide.min(axis=1).mean() / wide.native.mean()),
            "oracle_vs_best_constant_pct": 100 * (1 - wide.min(axis=1).mean() / wide[best_constant].mean())})
    relations = []
    joined = paired[paired.lookahead == 128].merge(state, on=JOB + ["probe_id", "kind"], validate="many_to_one")
    for key, frame in joined.groupby(JOB + ["policy"]):
        for feature in FEATURES:
            values = frame[[feature, "benefit"]].dropna()
            if len(values) >= 4 and values[feature].std() > 1e-12 and values.benefit.std() > 1e-12:
                relations.append(dict(zip(JOB + ["policy"], key)) | {"feature": feature, "r": values.corr().iloc[0, 1]})
    relations = pd.DataFrame(relations)
    relation_summary = relations.groupby(GROUP + ["policy", "feature"]).agg(
        mean_r=("r", "mean"), min_r=("r", "min"), max_r=("r", "max"), seeds=("seed", "nunique")).reset_index()
    outputs = {"summary_per_seed": summary, "full_policy_comparison": aggregate,
               "audit": pd.DataFrame(audit), "probe_states": state, "paired_branches": paired,
               "branch_comparison": branch_stats, "branch_by_kind": phase_stats,
               "hindsight_oracle": pd.DataFrame(oracle_rows), "phase_metrics": pd.concat(phases),
               "tail_errors": pd.DataFrame(tails), "expert_comparison": pd.DataFrame(experts),
               "weekly_blocks": pd.DataFrame(blocks),
               "feature_correlations": relations, "feature_correlation_across_seeds": relation_summary,
               "temporal_rule_check": temporal_rule_check(paired, state)}
    for name, frame in outputs.items():
        frame.to_csv(tables / f"{name}.csv", index=False, encoding="utf-8-sig")

    for method in sorted(summary.method.unique()):
        datasets = [d for d in DATASETS if ((summary.method == method) & (summary.dataset == d)).any()]
        for family, policies in [
            ("controls", ["native", "frozen", "no_grad"]),
            ("frequency", ["native", "every4_lr1", "every16_lr1", "every1_lr0.3"]),
            ("cumulative", ["native", "every4_lr1", "every16_lr1", "every1_lr0.3"]),
            ("mse", ["native", "every1_lr0.3", "every4_lr0.3", "every16_lr0.3"])]:
            fig, axes = plt.subplots(len(datasets), 2, figsize=(12, 3.1 * len(datasets) + .5), squeeze=False)
            for i, dataset in enumerate(datasets):
                for j, horizon in enumerate([1, 24]):
                    ax = axes[i, j]
                    seeds = sorted(summary[(summary.dataset == dataset) & (summary.method == method) & (summary.horizon == horizon)].seed.unique())
                    for color, policy in zip(COLORS, policies):
                        values = np.stack([full[(dataset, method, horizon, seed, policy)]["mse" if family == "mse" else "mae"].to_numpy() for seed in seeds])
                        draw_curve(ax, values, color, policy_label(policy), window=672 if dataset == "ETTm1" else 168,
                                   cumulative=family == "cumulative")
                    ax.set_title(f"{dataset} · {method.upper()} · H={horizon}")
            save_figure(fig, axes, figures / f"{method}_{family}", "Cumulative MAE" if family == "cumulative" else "Rolling MSE" if family == "mse" else "Rolling MAE")

        for horizon in [1, 24]:
            fig, axes = plt.subplots(len(datasets), 3, figsize=(14, 3.0 * len(datasets) + .5), squeeze=False)
            for i, dataset in enumerate(datasets):
                seeds = sorted(summary[(summary.dataset == dataset) & (summary.method == method) & (summary.horizon == horizon)].seed.unique())
                for j, period in enumerate([1, 4, 16]):
                    for color, rate in zip(COLORS, ["0.3", "1", "3"]):
                        policy = "native" if period == 1 and rate == "1" else f"every{period}_lr{rate}"
                        values = np.stack([full[(dataset, method, horizon, seed, policy)].mae.to_numpy() for seed in seeds])
                        draw_curve(axes[i, j], values, color, f"LR x{rate}", window=672 if dataset == "ETTm1" else 168)
                    axes[i, j].set_title(f"{dataset} · H={horizon} · Every {period}")
            save_figure(fig, axes, figures / f"{method}_lr_h{horizon}", "Rolling MAE")

            groups = ["base", "adapter", "head"] if method == "fsnet" else ["time", "variable", "decision"]
            module_sets = [["only_" + group for group in groups],
                           ["only_" + "+".join([a, b]) for a, b in [(groups[0], groups[1]), (groups[0], groups[2]), (groups[1], groups[2])]]]
            fig, axes = plt.subplots(len(datasets), 2, figsize=(12, 3.1 * len(datasets) + .6), squeeze=False)
            for i, dataset in enumerate(datasets):
                subset = state[(state.dataset == dataset) & (state.method == method) & (state.horizon == horizon)]
                for j, policies in enumerate(module_sets):
                    ax = axes[i, j]
                    ax.axhline(0, color="#555555", linestyle="--", linewidth=.8)
                    for color, policy in zip(COLORS, policies):
                        by_seed = []
                        for seed, probes in subset.groupby("seed"):
                            by_seed.append(np.mean([branches[(dataset, method, horizon, seed, p, policy)] - branches[(dataset, method, horizon, seed, p, "native")] for p in probes.probe_id], axis=0))
                        draw_curve(ax, np.stack(by_seed), color, policy_label(policy), cumulative=True)
                    ax.set_title(f"{dataset} · H={horizon} · {'Single module' if j == 0 else 'Two modules'}")
            # Shared legends keep labels outside the data area.
            left_handles, left_labels = axes[0, 0].get_legend_handles_labels()
            right_handles, right_labels = axes[0, 1].get_legend_handles_labels()
            fig.legend(left_handles + right_handles, left_labels + right_labels,
                       loc="upper center", bbox_to_anchor=(0.5, 1.005), ncol=3,
                       frameon=False, fontsize=9)
            for ax in axes.flat:
                for line in ax.lines:
                    line.set_label("_nolegend_")
            save_figure(fig, axes, figures / f"{method}_modules_h{horizon}", "Δ cumulative MAE", "Steps after branch")

            fig, axes = plt.subplots(len(datasets), 4, figsize=(16, 3.0 * len(datasets) + .7), squeeze=False)
            for i, dataset in enumerate(datasets):
                for j, kind in enumerate(["stable", "pre_event", "event", "post_event"]):
                    ax = axes[i, j]
                    subset = state[(state.dataset == dataset) & (state.method == method) & (state.horizon == horizon) & (state.kind == kind)]
                    ax.axhline(0, color="#555555", linestyle="--", linewidth=.8)
                    for color, policy in zip(COLORS, ["every1_lr0.3", "every4_lr0.3", "every16_lr0.3", "no_grad"]):
                        by_seed = []
                        for seed, probes in subset.groupby("seed"):
                            by_seed.append(np.mean([branches[(dataset, method, horizon, seed, p, policy)] - branches[(dataset, method, horizon, seed, p, "native")] for p in probes.probe_id], axis=0))
                        draw_curve(ax, np.stack(by_seed), color, policy_label(policy), cumulative=True)
                    ax.set_title(f"{dataset} · {kind.replace('_', ' ')}")
            save_figure(fig, axes, figures / f"{method}_branch_phases_h{horizon}", "Δ cumulative MAE", "Steps after branch")

    for policy in ["native", "no_grad"]:
        fig, axes = plt.subplots(3, 2, figsize=(12, 9.8), squeeze=False)
        for i, dataset in enumerate(DATASETS):
            for j, horizon in enumerate([1, 24]):
                seeds = sorted(summary[(summary.dataset == dataset) & (summary.method == "onenet") & (summary.horizon == horizon)].seed.unique())
                for color, column, label in zip(COLORS, ["mae", "time_expert_mae", "variable_expert_mae"], ["Mixture", "Time expert", "Variable expert"]):
                    values = np.stack([full[(dataset, "onenet", horizon, seed, policy)][column].to_numpy() for seed in seeds])
                    draw_curve(axes[i, j], values, color, label, window=672 if dataset == "ETTm1" else 168)
                axes[i, j].set_title(f"{dataset} · H={horizon}")
        save_figure(fig, axes, figures / f"onenet_experts_{policy}", "Rolling MAE")
    print(f"Done: {len(audit)} full trajectories, {len(branches)} branches, {len(list(figures.glob('*.png')))} figures.", flush=True)


if __name__ == "__main__":
    main()

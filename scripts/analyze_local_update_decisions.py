"""Local state/action/outcome analysis, joined to original annotations; no torch required."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GROUP = ["dataset", "method", "horizon"]
STATE = GROUP + ["seed", "probe_id"]
PHASES = ["stable", "transition", "drift"]
PHASE_COLORS = {"stable": "#FFFFFF", "transition": "#CAD3DD", "drift": "#F2B678"}
COLORS = ["#285B86", "#D27735", "#399A80", "#9666A4", "#BA4E59"]
FAMILIES = {
    "data": ["mean_shift_32", "input_std_32", "input_trend_32", "autocorrelation_32",
             "cross_correlation", "volatility_ratio", "daily_similarity", "channel_shift_fraction"],
    "feedback": ["mae_32", "error_ratio", "current_feedback_mae", "residual_bias",
                 "mature_mse_32", "mature_mae_variation"],
    "expert_onenet": ["long_time_weight", "combined_time_weight", "short_weight_shift",
                      "expert_disagreement", "expert_relative_gap", "mature_expert_gap_32"],
    "expert_fsnet": ["gradient_memory_norm", "memory_interactions"],
}


def short_label(policy):
    names = {"native": "Native", "no_grad": "No grad", "frozen": "Frozen"}
    return names.get(policy, policy.replace("every", "P").replace("_lr", " / LR ").replace("only_", "").replace("+", " + "))


def finish_figure(fig, axes, path, ylabel):
    for ax in np.asarray(axes).flat:
        ax.set_xlabel("Steps after decision")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#DDE2E6", linewidth=.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#CDD3D8")
        ax.set_xticks([1, 32, 64, 96, 128])
        ax.set_xlim(1, 128)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=170, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def causal_data_features(values, clock, daily_lag):
    """All slices end at the current observed row, never after the decision."""
    current, previous = values[clock - 31:clock + 1], values[clock - 63:clock - 31]
    current_std, previous_std = current.std(axis=0), previous.std(axis=0)
    lagged = values[clock - 127 - daily_lag:clock + 1 - daily_lag]
    recent = values[clock - 127:clock + 1]
    x, y = recent - recent.mean(axis=0), lagged - lagged.mean(axis=0)
    denom = np.sqrt((x * x).sum(axis=0) * (y * y).sum(axis=0))
    valid = denom > 1e-10
    similarity = ((x[:, valid] * y[:, valid]).sum(axis=0) / denom[valid]).mean()
    return {"volatility_ratio": current_std.mean() / max(previous_std.mean(), 1e-10),
            "daily_similarity": similarity,
            "channel_shift_fraction": np.mean(np.abs(current.mean(axis=0) - previous.mean(axis=0)) > previous_std + 1e-6)}


def association_checks(joined):
    """Descriptive associations and chronological direction checks, not fitted policies."""
    rows = []
    for key, group in joined.groupby(GROUP + ["lookahead", "policy", "metric"]):
        if key[4] == "native":
            continue
        features = FAMILIES["data"] + FAMILIES["feedback"] + FAMILIES["expert_" + key[1]]
        for feature in features:
            available = group.dropna(subset=[feature])
            if available.probe_id.nunique() < 8 or available[feature].std() < 1e-10:
                continue
            seed_r = []
            for _, seed_frame in available.groupby("seed"):
                seed_r.append(seed_frame[[feature, "relative_gain_pct"]].rank().corr().iloc[0, 1])
            probe = available.groupby("probe_id").agg(feature=(feature, "mean"),
                      gain=("relative_gain_pct", "mean"), clock=("clock", "first")).sort_values("clock")
            if probe.feature.std() < 1e-10 or probe.gain.std() < 1e-10:
                continue
            loo = [probe.drop(index)[["feature", "gain"]].rank().corr().iloc[0, 1] for index in probe.index]
            ranks = probe[["feature", "gain", "clock"]].rank().to_numpy()
            design = np.column_stack([np.ones(len(ranks)), ranks[:, 2]])
            residual = ranks[:, :2] - design @ np.linalg.lstsq(design, ranks[:, :2], rcond=None)[0]
            partial = np.corrcoef(residual.T)[0, 1] if np.all(residual.std(axis=0) > 1e-10) else np.nan
            cutoff = sorted(group.clock.unique())[6]
            early = probe[probe.clock + int(key[3]) + int(key[2]) < cutoff]
            late = probe[probe.clock >= cutoff]
            threshold = early.feature.median()
            early_low, early_high = early[early.feature <= threshold], early[early.feature > threshold]
            late_low, late_high = late[late.feature <= threshold], late[late.feature > threshold]
            early_gap = early_high.gain.mean() - early_low.gain.mean()
            late_gap = late_high.gain.mean() - late_low.gain.mean()
            rows.append(dict(zip(GROUP + ["lookahead", "policy", "metric"], key)) | {
                "feature": feature, "mean_seed_spearman": np.nanmean(seed_r),
                "min_seed_spearman": np.nanmin(seed_r), "max_seed_spearman": np.nanmax(seed_r),
                "loo_probe_r_min": np.nanmin(loo), "loo_probe_r_max": np.nanmax(loo),
                "clock_partial_spearman": partial, "available_probes": len(probe),
                "early_threshold": threshold, "early_high_minus_low_gain_pp": early_gap,
                "late_high_minus_low_gain_pp": late_gap, "late_low_probes": len(late_low),
                "late_high_probes": len(late_high), "early_late_same_direction": bool(early_gap * late_gap > 0),
                "late_low_gain_pct": late_low.gain.mean(), "late_high_gain_pct": late_high.gain.mean()})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/update-study-completed/update-study-main-4gpu"))
    parser.add_argument("--annotations", type=Path, default=Path("runs/real-drift-electricity-5pct-20261010"))
    parser.add_argument("--output", type=Path, default=Path("reports/update_study_completed/local"))
    args = parser.parse_args()
    figures, tables = args.output / "figures", args.output / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titlesize": 10, "axes.labelsize": 9, "pdf.fonttype": 42})
    streams, manifests, source_audit = {}, {}, []
    for dataset in ["ETTh2", "ETTm1", "Electricity"]:
        directory = args.annotations / dataset
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        frame = pd.read_csv(directory / "labeled_data.csv")
        digest = hashlib.sha256((directory / "labeled_data.csv").read_bytes()).hexdigest()
        assert np.array_equal(frame.annotation_index, np.arange(manifest["rows"]))
        intervals = pd.read_csv(directory / "intervals.csv")
        assert intervals.start_index.iloc[0] == 0 and intervals.end_index_exclusive.iloc[-1] == len(frame)
        assert np.array_equal(intervals.start_index.iloc[1:], intervals.end_index_exclusive.iloc[:-1])
        for row in intervals.itertuples():
            assert (frame.annotation_phase.iloc[row.start_index:row.end_index_exclusive] == row.phase).all()
        for path in args.runs.glob("*/manifest.json"):
            for job in json.loads(path.read_text(encoding="utf-8"))["jobs"]:
                if job["dataset"] == dataset:
                    assert digest == job["input_sha256"], (dataset, path)
        streams[dataset] = frame
        manifests[dataset] = manifest
        source_audit.append({"dataset": dataset, "rows": len(frame), "sha256": digest,
                             "intervals": len(intervals), "events": len(pd.read_csv(directory / "drift_points.csv"))})
    enriched, exposures, outcomes, target_phase, curves, policy_masks = [], [], [], [], {}, {}
    module_effects, conditional_effects, feature_variation = [], [], []
    for marker in sorted(args.runs.rglob("policies.json")):
        directory = marker.parent
        states = pd.read_csv(directory / "probe_states.csv")
        block_outcomes = pd.read_csv(directory / "probe_outcomes.csv")
        native_full = pd.read_csv(directory / "native/forecasts.csv")
        policies = json.loads(marker.read_text(encoding="utf-8"))
        outcomes.append(block_outcomes)
        first = states.iloc[0]
        dataset, method, horizon = first.dataset, first.method, int(first.horizon)
        stream = streams[dataset]
        phases = stream.annotation_phase.to_numpy()
        prefix = int(native_full.target_start_index.iloc[0])
        raw = stream[manifests[dataset]["feature_columns"]].to_numpy(dtype=float)
        scale = raw[:prefix].std(axis=0)
        scale[scale < 1e-6] = 1
        values = (raw - raw[:prefix].mean(axis=0)) / scale
        events = pd.read_csv(args.annotations / dataset / "drift_points.csv")
        names = ["base", "adapter", "head"] if method == "fsnet" else ["time", "variable", "decision"]
        masks = {frozenset(): "no_grad", frozenset(names): "native"}
        masks.update({frozenset(p["groups"]): p["name"] for p in policies if isinstance(p["groups"], list)})
        policy_masks[method] = masks
        print(f"Joining {directory.relative_to(args.runs)}", flush=True)
        for row in states.to_dict("records"):
            clock, probe_id = int(row["clock"]), row["probe_id"]
            identity = {name: row[name] for name in STATE}
            key = tuple(identity[name] for name in STATE)
            row.update(causal_data_features(values, clock, int(86400 / manifests[dataset]["sample_period_seconds"])))
            # At decision t, all forecasts with complete feedback <= t are observable.
            mature = native_full[native_full.feedback_raw_index <= clock].tail(128)
            assert mature.empty or mature.feedback_raw_index.max() <= clock
            row["mature_mse_32"] = mature.mse.tail(32).mean()
            row["mature_mae_variation"] = mature.mae.tail(32).std(ddof=0) / mature.mae.tail(32).mean()
            row["error_ratio"] = row["mae_32"] / row["mae_128"]
            if method == "onenet":
                row["short_weight_shift"] = row["combined_time_weight"] - row["long_time_weight"]
                row["expert_relative_gap"] = (row["time_expert_mae"] - row["variable_expert_mae"]) / max(row["time_expert_mae"] + row["variable_expert_mae"], 1e-10)
                row["mature_expert_gap_32"] = (mature.time_expert_mae.tail(32).mean() - mature.variable_expert_mae.tail(32).mean()) / max(mature.time_expert_mae.tail(32).mean() + mature.variable_expert_mae.tail(32).mean(), 1e-10)
            row["phase_at_decision"] = phases[clock]
            row["phase_at_feedback_origin"] = phases[clock - horizon]
            row["annotation_votes_at_decision"] = stream.majority_vote_count.iloc[clock]
            if row["event_id"]:
                event = events[events.event_id == row["event_id"]].iloc[0]
                row["offline_event_start_offset"] = int(event.start_index - clock)
                row["offline_event_end_offset"] = int(event.end_index_exclusive - clock)
                assert row["event_rows"] == event.end_index_exclusive - event.start_index
            for phase in PHASES:
                row[f"offline_context_{phase}_fraction"] = np.mean(phases[clock - 95:clock + 1] == phase)
                row[f"offline_mature_target_{phase}_fraction"] = np.mean(phases[clock - horizon + 1:clock + 1] == phase)
            enriched.append(row)
            for lookahead in [8, 32, 128]:
                indices = clock + 1 + np.arange(lookahead)[:, None] + np.arange(horizon)[None, :]
                target_phases = phases[indices]
                future = stream.iloc[clock + 1:clock + lookahead + horizon]
                exposure = identity | {"lookahead": lookahead, "kind": row["kind"], "phase_at_decision": phases[clock],
                            "future_distinct_events": len(set(future.loc[future.annotation_phase == "drift", "drift_event_id"]) - {0}),
                            "future_phase_changes": int(np.sum(np.diff(pd.factorize(phases[clock + 1:clock + lookahead + horizon])[0]) != 0))}
                for phase in PHASES:
                    exposure[f"target_{phase}_fraction"] = np.mean(target_phases == phase)
                exposures.append(exposure)
            for p in policies:
                policy = p["name"]
                branch_dir = directory / "probes" / probe_id / policy
                prediction = pd.read_csv(branch_dir / "forecasts.csv")
                assert len(prediction) == 128 and prediction.origin_raw_index.iloc[0] == clock
                assert np.array_equal(prediction.origin_raw_index, np.arange(clock, clock + 128))
                assert np.isfinite(prediction[["mae", "mse"]]).all().all()
                curves[key + (policy,)] = prediction[["mae", "mse"]].to_numpy()
                phase_frame = pd.read_csv(branch_dir / "phase_metrics.csv")
                phase_frame = phase_frame[(phase_frame.horizon_step == 0) & (phase_frame.phase != "all")]
                for phase_row in phase_frame.to_dict("records"):
                    expected = np.sum(phases[clock + 1 + np.arange(128)[:, None] + np.arange(horizon)[None, :]] == phase_row["phase"]) * raw.shape[1]
                    assert phase_row["values"] == expected
                    target_phase.append(identity | {"policy": policy, "kind": row["kind"], **phase_row})
            for module in names:
                without = [mask for mask in masks if module not in mask]
                off = np.mean([curves[key + (masks[mask],)] for mask in without], axis=0)
                on = np.mean([curves[key + (masks[mask | {module}],)] for mask in without], axis=0)
                curves[key + ("effect_" + module,)] = on - off
                for lookahead in [8, 32, 128]:
                    for idx, metric in enumerate(["mae", "mse"]):
                        module_effects.append(identity | {"module": module, "lookahead": lookahead, "metric": metric,
                            "off_loss": off[:lookahead, idx].mean(), "on_loss": on[:lookahead, idx].mean(),
                            "gain": (off[:lookahead, idx] - on[:lookahead, idx]).mean(),
                            "relative_gain_pct": 100 * (1 - on[:lookahead, idx].mean() / off[:lookahead, idx].mean())})
                        for mask in without:
                            off_loss = curves[key + (masks[mask],)][:lookahead, idx].mean()
                            on_loss = curves[key + (masks[mask | {module}],)][:lookahead, idx].mean()
                            conditional_effects.append(identity | {"module": module,
                                "off_policy": masks[mask], "on_policy": masks[mask | {module}],
                                "lookahead": lookahead, "metric": metric, "off_loss": off_loss,
                                "on_loss": on_loss, "gain": off_loss - on_loss,
                                "relative_gain_pct": 100 * (1 - on_loss / off_loss)})

    states, exposure, outcome = pd.DataFrame(enriched), pd.DataFrame(exposures), pd.concat(outcomes, ignore_index=True)
    assert not states.duplicated(STATE).any()
    long = outcome.melt(id_vars=STATE + ["kind", "policy", "lookahead", "branch_seconds", "branch_gradient_steps"], value_vars=["mae", "mse"], var_name="metric", value_name="loss")
    pair = STATE + ["lookahead", "metric"]
    long = long.merge(long[long.policy == "native"][pair + ["loss"]].rename(columns={"loss": "native_loss"}), on=pair, validate="many_to_one")
    long["gain"] = long.native_loss - long.loss
    long["relative_gain_pct"] = 100 * long.gain / long.native_loss
    joined = long.merge(states, on=STATE + ["kind"], validate="many_to_one").merge(exposure.drop(columns=["kind", "phase_at_decision"]), on=STATE + ["lookahead"], validate="many_to_one")
    summary = long.groupby(GROUP + ["lookahead", "metric", "policy"]).agg(loss=("loss", "mean"), native_loss=("native_loss", "mean"), mean_probe_gain_pct=("relative_gain_pct", "mean"), win_1pct_fraction=("relative_gain_pct", lambda v: (v >= 1).mean())).reset_index()
    summary["pooled_gain_pct"] = 100 * (1 - summary.loss / summary.native_loss)
    decision_phase = joined.groupby(GROUP + ["lookahead", "phase_at_decision", "metric", "policy"]).agg(
        loss=("loss", "mean"), native_loss=("native_loss", "mean"), probes=("probe_id", "nunique"),
        mean_probe_gain_pct=("relative_gain_pct", "mean")).reset_index()
    decision_phase["pooled_gain_pct"] = 100 * (1 - decision_phase.loss / decision_phase.native_loss)
    rankings, consistency = [], []
    for key, frame in long.groupby(STATE + ["lookahead", "metric"]):
        ordered = frame.sort_values("loss")
        best = ordered.iloc[0]
        rankings.append(dict(zip(STATE + ["lookahead", "metric"], key)) | {"best_policy": best.policy,
            "best_loss": best.loss, "native_loss": best.native_loss,
            "native_regret_pct": 100 * (best.native_loss / best.loss - 1),
            "near_best_1pct_actions": ";".join(ordered.loc[ordered.loss <= best.loss * 1.01, "policy"])})
    ranks = pd.DataFrame(rankings)
    for key, frame in long.groupby(GROUP + ["probe_id", "lookahead", "metric"]):
        wide = frame.pivot(index="seed", columns="policy", values="loss")
        winner = wide.mean().idxmin()
        native = wide.native
        consistency.append(dict(zip(GROUP + ["probe_id", "lookahead", "metric"], key)) | {
            "seed_mean_best_policy": winner, "best_mean_loss": wide[winner].mean(), "native_mean_loss": native.mean(),
            "gain_pct": 100 * (1 - wide[winner].mean() / native.mean()),
            "winner_beats_native_all_seeds": bool((wide[winner] < native).all()),
            "winner_within_1pct_of_best_all_seeds": bool((wide[winner] <= wide.min(axis=1) * 1.01).all()),
            "distinct_seed_winners": wide.idxmin(axis=1).nunique()})
    phase_table = pd.DataFrame(target_phase)
    phase_long = phase_table.melt(id_vars=STATE + ["policy", "kind", "phase", "values"], value_vars=["mae", "mse"], var_name="metric", value_name="loss")
    phase_long = phase_long.merge(phase_long[phase_long.policy == "native"][STATE + ["phase", "metric", "loss"]].rename(columns={"loss": "native_loss"}), on=STATE + ["phase", "metric"], validate="many_to_one")
    phase_long["gain"] = phase_long.native_loss - phase_long.loss
    phase_long["weighted_gain"] = phase_long.gain * phase_long["values"]
    phase_long["weighted_native"] = phase_long.native_loss * phase_long["values"]
    phase_summary = phase_long.groupby(GROUP + ["phase", "metric", "policy"]).agg(weighted_gain=("weighted_gain", "sum"), weighted_native=("weighted_native", "sum"), probes=("probe_id", "nunique"), target_values=("values", "sum")).reset_index()
    phase_summary["gain_pct"] = 100 * phase_summary.weighted_gain / phase_summary.weighted_native
    for key, group in states.groupby(GROUP):
        for feature in sum(FAMILIES.values(), []) + ["steps_since_gradient", "recent_update_fraction"]:
            if feature in group:
                feature_variation.append(dict(zip(GROUP, key)) | {"feature": feature, "distinct_values": group[feature].nunique(), "minimum": group[feature].min(), "maximum": group[feature].max()})
    effects = pd.DataFrame(module_effects)
    associations = association_checks(joined)
    output_tables = {"source_audit": pd.DataFrame(source_audit), "state_inputs": states,
                     "phase_exposure": exposure, "local_outcomes": long, "local_action_summary": summary,
                     "local_best_actions": ranks, "seed_consistency": pd.DataFrame(consistency),
                     "exact_target_phase_effects": phase_long, "exact_target_phase_summary": phase_summary,
                     "module_factorial_effects": effects, "state_gain_associations": associations,
                     "module_conditional_effects": pd.DataFrame(conditional_effects),
                     "decision_phase_summary": decision_phase,
                     "feature_coverage": pd.DataFrame(feature_variation)}
    for name, frame in output_tables.items():
        frame.to_csv(tables / (name + ".csv"), index=False, encoding="utf-8-sig")

    # Every sampled location is shown, so striking examples do not replace the full local evidence.
    for group_key, group in states.groupby(GROUP):
        dataset, method, horizon = group_key
        probes = group.drop_duplicates("probe_id").sort_values("clock")
        seeds = sorted(group.seed.unique())
        for column, metric in enumerate(["mae", "mse"]):
            fig, axes = plt.subplots(4, 3, figsize=(15, 13), squeeze=False)
            for ax, probe in zip(axes.flat, probes.itertuples()):
                candidates = outcome[(outcome.dataset == dataset) & (outcome.method == method) & (outcome.horizon == horizon) & (outcome.probe_id == probe.probe_id) & (outcome.lookahead == 128)]
                mean_losses = candidates.groupby("policy")[metric].mean()
                module = mean_losses[mean_losses.index.str.startswith("only_")].idxmin()
                when = mean_losses[~mean_losses.index.str.startswith("only_")].idxmin()
                policies = list(dict.fromkeys(["native", "every1_lr0.3", "no_grad", when, module, "every4_lr0.3"]))[:5]
                segment = streams[dataset].annotation_phase.iloc[probe.clock + 1:probe.clock + 129].to_numpy()
                changes = np.r_[0, np.flatnonzero(segment[1:] != segment[:-1]) + 1, 128]
                for left, right in zip(changes[:-1], changes[1:]):
                    if segment[left] != "stable":
                        ax.axvspan(left + .5, right + .5, color=PHASE_COLORS[segment[left]], alpha=.28, linewidth=0)
                for color, policy in zip(COLORS, policies):
                    losses = np.stack([curves[(dataset, method, horizon, seed, probe.probe_id, policy)][:, column] for seed in seeds])
                    average = losses.cumsum(axis=1) / np.arange(1, 129)
                    label = short_label(policy) + (" [best module]" if policy == module else " [best when]" if policy == when else "")
                    ax.plot(np.arange(1, 129), average.mean(axis=0), color=color, label=label, linewidth=1.15)
                    ax.fill_between(np.arange(1, 129), average.min(axis=0), average.max(axis=0), color=color, alpha=.07, linewidth=0)
                ax.set_title(probe.probe_id.replace("_", " "))
                ax.legend(fontsize=6.5, frameon=True, framealpha=.8, edgecolor="none", loc="best")
            finish_figure(fig, axes, figures / f"{dataset}_{method}_h{horizon}_local_{metric}", f"Cumulative {metric.upper()}")

    for method in ["fsnet", "onenet"]:
        datasets = [d for d in ["ETTh2", "ETTm1", "Electricity"] if ((states.dataset == d) & (states.method == method)).any()]
        modules = ["base", "adapter", "head"] if method == "fsnet" else ["time", "variable", "decision"]
        for horizon in [1, 24]:
            fig, axes = plt.subplots(len(datasets), 2, figsize=(12, 3.4 * len(datasets)), squeeze=False)
            for i, dataset in enumerate(datasets):
                group = states[(states.dataset == dataset) & (states.method == method) & (states.horizon == horizon)]
                for j, metric in enumerate(["mae", "mse"]):
                    ax = axes[i, j]
                    ax.axhline(0, color="#666666", linewidth=.7, linestyle="--")
                    for color, module in zip(COLORS, modules):
                        by_seed = []
                        for seed, probes in group.groupby("seed"):
                            effect = np.mean([curves[(dataset, method, horizon, seed, p, "effect_" + module)][:, j] for p in probes.probe_id], axis=0)
                            by_seed.append(effect.cumsum() / np.arange(1, 129))
                        values = np.stack(by_seed)
                        ax.plot(np.arange(1, 129), values.mean(axis=0), label=module.title(), color=color)
                        ax.fill_between(np.arange(1, 129), values.min(axis=0), values.max(axis=0), color=color, alpha=.1, linewidth=0)
                    ax.set_title(f"{dataset} · H={horizon} · {metric.upper()}")
                    ax.legend(frameon=False, fontsize=8)
            finish_figure(fig, axes, figures / f"{method}_h{horizon}_module_effects", "Loss: module on − off")

            data_feature = "channel_shift_fraction" if method == "onenet" and horizon == 24 else "volatility_ratio" if method == "fsnet" and horizon == 24 else "mean_shift_32"
            selections = [(data_feature, "every4_lr0.3"), ("mae_32", "every1_lr0.3")]
            selections += [("gradient_memory_norm", "no_grad"), ("daily_similarity", "only_base+adapter")] if method == "fsnet" else [("expert_relative_gap", "only_variable+decision"), ("short_weight_shift", "only_time+variable")]
            fig, axes = plt.subplots(len(datasets), 4, figsize=(16, 3.6 * len(datasets)), squeeze=False)
            for i, dataset in enumerate(datasets):
                group = states[(states.dataset == dataset) & (states.method == method) & (states.horizon == horizon)]
                for j, (feature, policy) in enumerate(selections):
                    ax = axes[i, j]
                    ax.axhline(0, color="#666666", linewidth=.7, linestyle="--")
                    for color, high in zip(COLORS, [False, True]):
                        by_seed = []
                        for seed, probes in group.groupby("seed"):
                            available = probes.dropna(subset=[feature])
                            subset = available[(available[feature] > available[feature].median()) == high]
                            pairs = []
                            for p in subset.probe_id:
                                candidate = curves[(dataset, method, horizon, seed, p, policy)][:, 0].cumsum()
                                native = curves[(dataset, method, horizon, seed, p, "native")][:, 0].cumsum()
                                pairs.append(100 * (candidate / native - 1))
                            by_seed.append(np.mean(pairs, axis=0))
                        values = np.stack(by_seed)
                        ax.plot(np.arange(1, 129), values.mean(axis=0), label="High" if high else "Low", color=color)
                        ax.fill_between(np.arange(1, 129), values.min(axis=0), values.max(axis=0), color=color, alpha=.09, linewidth=0)
                    ax.set_title(f"{dataset} · {feature}\n{short_label(policy)}", fontsize=9)
                    ax.legend(frameon=False, fontsize=8)
            finish_figure(fig, axes, figures / f"{method}_h{horizon}_state_strata", "Relative Δ MAE (%)")
    print(f"Complete: {len(states)} states, {len(long)} local outcomes, {len(list(figures.glob('*.png')))} figures", flush=True)


if __name__ == "__main__":
    main()

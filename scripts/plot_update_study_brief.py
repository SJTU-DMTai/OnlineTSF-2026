"""Plot selected local comparisons for the brief report, using saved forecasts only."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = ["#285B86", "#D27735", "#399A80", "#9666A4"]
LABELS = {
    "native": "Native",
    "every4_lr0.3": "Every 4 / LR x0.3",
    "every1_lr0.3": "Every 1 / LR x0.3",
    "only_variable+decision": "Variable + decision only",
    "frozen": "Frozen",
    "no_grad": "No grad",
}
POLICY_COLORS = {"native": COLORS[0], "every4_lr0.3": COLORS[1],
                 "every1_lr0.3": COLORS[2], "only_variable+decision": COLORS[2],
                 "frozen": COLORS[1], "no_grad": COLORS[2]}


def read_losses(runs, dataset, method, probe, policy):
    """Return three matched seed trajectories, each with 128 MAE/MSE pairs."""
    gpu = "gpu1" if method == "fsnet" else "gpu3"
    values = []
    for seed in range(3):
        path = runs / gpu / dataset / f"{method}_h24_seed{seed}" / "probes" / probe / policy / "forecasts.csv"
        frame = pd.read_csv(path)
        clock = int(probe.rsplit("_", 1)[1])
        assert np.array_equal(frame.origin_raw_index, np.arange(clock, clock + 128)), path
        values.append(frame[["mae", "mse"]].to_numpy())
    result = np.stack(values)
    assert result.shape == (3, 128, 2) and np.isfinite(result).all()
    return result


def save_figure(fig, axes, path):
    for ax in np.asarray(axes).flat:
        ax.set_xlabel("Steps after decision")
        ax.set_xticks([1, 32, 64, 96, 128])
        ax.set_xlim(1, 128)
        ax.grid(axis="y", color="#DDE2E6", linewidth=.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#CDD3D8")
        ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/update-study-completed/update-study-main-4gpu"))
    parser.add_argument("--analysis", type=Path, default=Path("reports/update_study_completed/local"))
    parser.add_argument("--output", type=Path, default=Path("reports/update_study_completed/brief"))
    args = parser.parse_args()
    figures = args.output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 10,
                         "axes.titlepad": 9, "lines.linewidth": 1.7, "pdf.fonttype": 42})
    steps = np.arange(1, 129)
    # Fixed selections match the four questions in the brief report.
    plans = [
        ("01_data_pattern", 1, [
            ("ETTh2", "onenet", "stable_5198", "mae", ["native", "every4_lr0.3", "no_grad"]),
            ("ETTh2", "onenet", "stable_8273", "mae", ["native", "every4_lr0.3", "no_grad"])]),
        ("02_expert_training", 2, [
            ("Electricity", "onenet", "event_17536", "mae", ["native", "every4_lr0.3", "only_variable+decision"]),
            ("Electricity", "onenet", "event_17536", "mse", ["native", "every4_lr0.3", "only_variable+decision"]),
            ("ETTm1", "onenet", "stable_31637", "mae", ["native", "only_variable+decision"]),
            ("ETTm1", "onenet", "stable_31637", "mse", ["native", "only_variable+decision"])]),
        ("03_action_duration", 1, [
            ("ETTh2", "onenet", "event_4095", "mae", ["native", "frozen", "every1_lr0.3"]),
            ("ETTh2", "onenet", "event_4095", "mse", ["native", "frozen", "every1_lr0.3"])]),
        ("04_fsnet_coordination", 1, [
            ("ETTh2", "fsnet", "post_event_13781", "mae", ["native", "frozen", "no_grad"])])]
    checkpoints = []
    for name, rows, panels in plans:
        fig, axes = plt.subplots(rows, 2, figsize=(10.8, 3.65 * rows), squeeze=False)
        for ax, (dataset, method, probe, metric, policies) in zip(axes.flat, panels):
            column = ["mae", "mse"].index(metric)
            for policy in policies:
                color = POLICY_COLORS[policy]
                cumulative = read_losses(args.runs, dataset, method, probe, policy)[:, :, column].cumsum(axis=1) / steps
                ax.plot(steps, cumulative.mean(axis=0), color=color, label=LABELS[policy])
                ax.fill_between(steps, cumulative.min(axis=0), cumulative.max(axis=0), color=color, alpha=.09, linewidth=0)
                for lookahead in [8, 32, 128]:
                    checkpoints.append({"dataset": dataset, "method": method, "horizon": 24,
                                        "probe_id": probe, "metric": metric, "policy": policy,
                                        "lookahead": lookahead, "mean_loss": cumulative[:, lookahead - 1].mean()})
            method_label = "OneNet" if method == "onenet" else "FSNet"
            ax.set_title(f"{dataset} / {method_label} / H=24 / {probe}")
            ax.set_ylabel(f"Cumulative {metric.upper()}")
        if name == "04_fsnet_coordination":
            ax = axes[0, 1]
            states = pd.read_csv(args.analysis / "tables/state_inputs.csv")
            conditional = pd.read_csv(args.analysis / "tables/module_conditional_effects.csv")
            probes = states[(states.dataset == "ETTh2") & (states.method == "fsnet") & (states.horizon == 24)].probe_id.unique()
            for color, off_policy, on_policy, label in [
                (COLORS[0], "no_grad", "only_head", "Base + adapter fixed"),
                (COLORS[1], "only_base+adapter", "native", "Base + adapter trained")]:
                effects = []
                for probe in probes:
                    off = read_losses(args.runs, "ETTh2", "fsnet", probe, off_policy)[:, :, 0]
                    on = read_losses(args.runs, "ETTh2", "fsnet", probe, on_policy)[:, :, 0]
                    effects.append((on - off).cumsum(axis=1) / steps)
                by_seed = np.mean(effects, axis=0)
                reference = conditional[(conditional.dataset == "ETTh2") & (conditional.method == "fsnet")
                                        & (conditional.horizon == 24) & (conditional.module == "head")
                                        & (conditional.lookahead == 128) & (conditional.metric == "mae")
                                        & (conditional.off_policy == off_policy)]
                assert np.isclose(by_seed[:, -1].mean(), (reference.on_loss - reference.off_loss).mean())
                ax.plot(steps, by_seed.mean(axis=0), color=color, label=label)
                ax.fill_between(steps, by_seed.min(axis=0), by_seed.max(axis=0), color=color, alpha=.09, linewidth=0)
            ax.axhline(0, color="#666666", linestyle="--", linewidth=.8)
            ax.set_title("ETTh2 / FSNet / H=24 / Head training")
            ax.set_ylabel("Cumulative MAE: head on - off")
        save_figure(fig, axes, figures / name)
    evidence = pd.DataFrame(checkpoints).drop_duplicates()
    reference = pd.read_csv(args.analysis / "tables/local_outcomes.csv")
    matched = evidence.merge(reference, on=["dataset", "method", "horizon", "probe_id", "metric", "policy", "lookahead"], validate="one_to_many")
    comparison = matched.groupby(["dataset", "method", "probe_id", "metric", "policy", "lookahead"])[["mean_loss", "loss"]].mean()
    assert np.allclose(comparison.mean_loss, comparison.loss, rtol=1e-10, atol=1e-12)
    evidence.to_csv(args.output / "selected_evidence.csv", index=False, encoding="utf-8-sig")
    print(f"Saved 4 figures (PNG + PDF) and {len(evidence)} checked local results to {args.output}")


if __name__ == "__main__":
    main()

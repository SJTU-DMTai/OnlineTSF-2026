"""Experiment I/O and hindsight diagnostics; not used by online inference."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


DEFAULT_RUNS = Path("runs/update-study-completed/update-study-main-4gpu")
DEFAULT_ANNOTATIONS = Path("runs/real-drift-electricity-5pct-20261010")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_job(runs: Path, dataset: str, method: str, horizon: int, seed: int) -> Path:
    matches = list(runs.glob(f"**/{dataset}/{method}_h{horizon}_seed{seed}/policies.json"))
    if len(matches) != 1:
        raise ValueError(f"expected one job for {dataset}/{method}/H={horizon}/seed={seed}, found {len(matches)}")
    return matches[0].parent


def save_json(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save_timings(directory: Path, timings: dict) -> None:
    """Export measured durations in seconds; overlapping totals are not additive."""
    save_json(directory / "timings.json", timings)
    pd.DataFrame(timings.items(), columns=["stage", "seconds"]).to_csv(directory / "timings.csv", index=False)


def load_existing_results(experiment: Path, controller: str, start: int, steps: int) -> tuple[pd.DataFrame, dict]:
    """Read measured experiment-2 results without executing any forecasting method."""
    forecasts = pd.read_csv(experiment / controller / "forecasts.csv")
    if len(forecasts) != steps or not np.array_equal(forecasts.forecast_index, np.arange(start, start + steps)):
        raise ValueError(f"{controller}: saved forecasts do not cover the test interval")
    summaries = pd.read_csv(experiment / "trajectory_summary.csv")
    selected = summaries[summaries.controller == controller]
    if len(selected) != 1:
        raise ValueError(f"expected one saved summary for {controller}")
    summary = selected.iloc[0].to_dict()
    for metric in ["mae", "mse"]:
        if not np.isclose(summary[metric], forecasts[metric].mean(), rtol=1e-7, atol=1e-9):
            raise ValueError(f"{controller}: saved summary and forecasts disagree")
    updates = pd.read_csv(experiment / controller / "updates.csv")
    update_clocks = 0
    if not updates.empty:
        increments = updates.gradient_steps.diff().fillna(updates.gradient_steps.iloc[0])
        update_clocks = updates.loc[increments > 0, "available_at"].nunique()
    summary.update(gradient_update_clocks=int(update_clocks), update_seconds=float(updates.seconds.sum()),
                   result_source="experiment2_saved", executed_in_comparison=False)
    summary.pop("seconds")
    return forecasts, summary


def plot_forecasts(frames: dict[str, pd.DataFrame], destination: Path, title: str) -> None:
    """Actual emitted forecast errors, with a short trailing average for readability."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    colors = ["#285B86", "#D27735", "#399A80", "#9666A4", "#9B7045", "#BD455C"]
    for ax, metric in zip(axes, ["mae", "mse"]):
        for (name, frame), color in zip(frames.items(), colors):
            ax.plot(np.arange(1, len(frame) + 1), frame[metric].rolling(32, min_periods=1).mean(),
                    label=name, color=color, linewidth=1.5)
        ax.set(xlabel="Steps", ylabel=metric.upper() + " (trailing 32)")
        ax.grid(axis="y", color="#DDE2E6", linewidth=.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(destination.with_suffix(".png"), dpi=170)
    fig.savefig(destination.with_suffix(".pdf"))
    plt.close(fig)


def oracle_decision(outcomes: list[dict], selected: str, metric: str) -> dict:
    """Score a choice against the best fixed action over this local window.

    ACC accepts all numerically tied optima (rtol=1e-7, atol=1e-10).
    Relative regret is (selected - optimum) / optimum, not a percentage.
    """
    losses = np.array([row[metric] for row in outcomes], dtype=float)
    if not np.isfinite(losses).all() or (losses < 0).any():
        raise ValueError("oracle losses must be finite and nonnegative")
    best = float(losses.min())
    if best == 0:
        raise ValueError("relative regret is undefined for a zero optimal loss")
    names = [row["action"] for row in outcomes]
    chosen = float(losses[names.index(selected)])
    optimal = np.isclose(losses, best, rtol=1e-7, atol=1e-10)
    relative = (chosen - best) / best
    return {"metric": metric, "selected_action": selected,
            "oracle_action": names[int(losses.argmin())],
            "optimal_actions": json.dumps([name for name, match in zip(names, optimal) if match]),
            "selected_loss": chosen, "oracle_loss": best,
            "native_loss_same_state": float(losses[names.index("native")]),
            "correct": bool(optimal[names.index(selected)]),
            "relative_regret": relative, "relative_regret_pct": relative * 100,
            "within_1pct": bool(chosen <= best * 1.01 + 1e-12)}


def summarize_decisions(decisions: pd.DataFrame, metric: str) -> dict:
    """Each full decision interval has equal weight in these metrics."""
    return {"metric": metric, "decisions": len(decisions),
            "correct_decisions": int(decisions.correct.sum()), "acc": float(decisions.correct.mean()),
            "acc_pct": float(decisions.correct.mean() * 100),
            "mean_relative_regret": float(decisions.relative_regret.mean()),
            "mean_relative_regret_pct": float(decisions.relative_regret.mean() * 100),
            "median_relative_regret_pct": float(decisions.relative_regret.median() * 100),
            "within_1pct_decisions": int(decisions.within_1pct.sum()),
            "within_1pct_fraction": float(decisions.within_1pct.mean()),
            "within_1pct_pct": float(decisions.within_1pct.mean() * 100),
            "mean_selected_loss": float(decisions.selected_loss.mean()),
            "mean_oracle_loss_same_states": float(decisions.oracle_loss.mean()),
            "mean_native_loss_same_states": float(decisions.native_loss_same_state.mean())}


def score_action(replay, stream, index: int, steps: int, policy) -> dict:
    """Preserve unchanged-policy cadence and score only newly emitted forecasts."""
    began = perf_counter()
    branch = copy.deepcopy(replay)
    before_gradient, before_egd = branch.method.gradient_steps, branch.method.egd_steps
    branch.method.switch_policy(policy)
    count, absolute, squared = 0, 0., 0.
    for current in range(index, index + steps):
        for record in branch.step(stream.dataset, current):
            if record.get("index", -1) >= index:
                count += 1
                absolute += record["mae"]
                squared += record["mse"]
    # Mature terminal predictions without applying future gradient updates.
    for record in branch.finish(stream.dataset):
        if record["index"] >= index:
            count += 1
            absolute += record["mae"]
            squared += record["mse"]
    if count != steps:
        raise RuntimeError(f"{policy.name}: scored {count} forecasts, expected {steps}")
    return {"action": policy.name, "mae": absolute / steps, "mse": squared / steps,
            "forecasts": count, "gradient_steps": branch.method.gradient_steps - before_gradient,
            "egd_steps": branch.method.egd_steps - before_egd, "seconds": perf_counter() - began}



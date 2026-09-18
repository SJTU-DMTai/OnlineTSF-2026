# -*- coding: utf-8 -*-
"""Plot summary accuracy and online error curves for ETTh1 and ETTh2."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


DISPLAY_NAMES = {
    "linear_ogd": "Linear",
    "lstm_ogd": "LSTM",
    "tcn_ogd": "TCN",
    "patchtst_ogd": "PatchTST",
    "fsnet": "FSNet",
    "onenet": "OneNet",
}

COLORS = {
    "linear_ogd": "#A37C73",
    "lstm_ogd": "#9A8F7A",
    "tcn_ogd": "#718096",
    "patchtst_ogd": "#8B7D93",
    "fsnet": "#6F8E7D",
    "onenet": "#4C6A92",
}

DATASET_NAMES = ("ETTh1", "ETTh2")
STRATEGY_ORDER = (
    "linear_ogd",
    "patchtst_ogd",
    "lstm_ogd",
    "tcn_ogd",
    "fsnet",
    "onenet",
)


def configure_style() -> None:
    """Configure a restrained style and use a Chinese font when available."""

    installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
    font_candidates = ("Microsoft YaHei", "DengXian", "SimHei", "Noto Sans CJK SC")
    selected_font = next(
        (font for font in font_candidates if font in installed_fonts),
        "DejaVu Sans",
    )
    plt.rcParams.update(
        {
            "font.family": selected_font,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#A8AFB7",
            "axes.labelcolor": "#30343B",
            "xtick.color": "#525861",
            "ytick.color": "#30343B",
            "text.color": "#252A31",
            "grid.color": "#D9DDE2",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.7,
            "axes.titleweight": "semibold",
            "axes.titlesize": 16,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "savefig.facecolor": "white",
        }
    )


def resolve_batch_directory(path: Path) -> Path:
    """Resolve a direct batch directory or the latest batch below a dataset directory."""

    if (path / "summary.csv").is_file() and (path / "forecast_steps.csv").is_file():
        return path
    candidates = sorted(
        directory
        for directory in path.glob("batch-*")
        if (directory / "summary.csv").is_file()
        and (directory / "forecast_steps.csv").is_file()
    )
    if not candidates:
        raise FileNotFoundError(
            f"No batch directory containing summary.csv and forecast_steps.csv under {path}"
        )
    return candidates[-1]


def load_batch(path: Path, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Load the summary and per-step results for one dataset."""

    batch_directory = resolve_batch_directory(path)
    summary = pd.read_csv(batch_directory / "summary.csv")
    steps = pd.read_csv(
        batch_directory / "forecast_steps.csv",
        usecols=["strategy", "seed", "forecast_index", "step_mae"],
    )
    summary.insert(0, "dataset", dataset)
    steps.insert(0, "dataset", dataset)
    return summary, steps, batch_directory


def plot_accuracy(summary: pd.DataFrame, output_path: Path, dpi: int) -> None:
    """Plot mean prequential MAE with standard deviation across seeds."""

    required = {"dataset", "strategy", "seed", "mae"}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"summary.csv is missing columns: {sorted(missing)}")

    stats = (
        summary.groupby(["dataset", "strategy"], as_index=False)
        .agg(mean_mae=("mae", "mean"), std_mae=("mae", "std"), seeds=("seed", "nunique"))
    )
    stats["std_mae"] = stats["std_mae"].fillna(0.0)
    unknown = set(stats["strategy"]).difference(DISPLAY_NAMES)
    if unknown:
        raise ValueError(f"Unknown strategies in summary.csv: {sorted(unknown)}")

    seed_counts = stats.groupby("dataset")["seeds"].min()
    if not (seed_counts == stats.groupby("dataset")["seeds"].max()).all():
        raise ValueError("Strategies within a dataset have different seed counts")
    if seed_counts.nunique() != 1:
        raise ValueError("Datasets have different seed counts")

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.8), sharey=True)
    y = np.arange(len(STRATEGY_ORDER))
    for ax, dataset in zip(axes, DATASET_NAMES, strict=True):
        data = (
            stats[stats["dataset"] == dataset]
            .set_index("strategy")
            .reindex(STRATEGY_ORDER)
            .reset_index()
        )
        if data["mean_mae"].isna().any():
            raise ValueError(f"{dataset} is missing one or more configured strategies")
        bars = ax.barh(
            y,
            data["mean_mae"],
            xerr=data["std_mae"],
            height=0.62,
            color=[COLORS[name] for name in data["strategy"]],
            edgecolor="none",
            error_kw={"ecolor": "#4C535C", "elinewidth": 1.1, "capsize": 3},
        )
        ax.set_yticks(y, [DISPLAY_NAMES[name] for name in STRATEGY_ORDER])
        ax.set_xlabel("Prequential MAE")
        ax.set_title(dataset, loc="left", pad=10)
        ax.xaxis.grid(True)
        ax.yaxis.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)

        largest = float(data["mean_mae"].max())
        ax.set_xlim(0, largest * 1.20)
        for bar, mean_value, std_value in zip(
            bars,
            data["mean_mae"],
            data["std_mae"],
            strict=True,
        ):
            ax.text(
                mean_value + largest * 0.018,
                bar.get_y() + bar.get_height() / 2,
                f"{mean_value:.3f} ± {std_value:.3f}",
                va="center",
                ha="left",
                fontsize=9.2,
                color="#30343B",
            )

    common_seed_count = int(seed_counts.min())
    fig.suptitle("不同模型的初步预测精度", x=0.07, y=0.995, ha="left", fontsize=18, fontweight="semibold")
    fig.text(
        0.07,
        0.915,
        f"{common_seed_count} 个随机种子的均值 ± 标准差",
        color="#69717B",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.12, right=0.98, top=0.80, bottom=0.13, wspace=0.18)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_online_error(
    steps: pd.DataFrame,
    output_path: Path,
    rolling_window: int,
    dpi: int,
) -> None:
    """Plot rolling step MAE, averaged across seeds at matching stream indices."""

    required = {"dataset", "strategy", "seed", "forecast_index", "step_mae"}
    missing = required.difference(steps.columns)
    if missing:
        raise ValueError(f"forecast_steps.csv is missing columns: {sorted(missing)}")
    unknown = set(steps["strategy"]).difference(DISPLAY_NAMES)
    if unknown:
        raise ValueError(f"Unknown strategies in forecast_steps.csv: {sorted(unknown)}")

    steps = steps.sort_values(["dataset", "strategy", "seed", "forecast_index"]).copy()
    min_periods = max(1, rolling_window // 4)
    steps["rolling_mae"] = (
        steps.groupby(["dataset", "strategy", "seed"], sort=False)["step_mae"]
        .rolling(rolling_window, min_periods=min_periods)
        .mean()
        .reset_index(level=[0, 1, 2], drop=True)
    )
    curves = (
        steps.groupby(["dataset", "strategy", "forecast_index"], as_index=False)
        .agg(
            mean_mae=("rolling_mae", "mean"),
            min_mae=("rolling_mae", "min"),
            max_mae=("rolling_mae", "max"),
        )
        .dropna()
    )

    main_strategies = ("onenet", "fsnet", "tcn_ogd", "lstm_ogd", "patchtst_ogd")
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15.0, 8.2),
        sharex="col",
        gridspec_kw={"height_ratios": [3.1, 1.0], "hspace": 0.14, "wspace": 0.14},
    )

    for column, dataset in enumerate(DATASET_NAMES):
        ax_main = axes[0, column]
        ax_linear = axes[1, column]
        dataset_curves = curves[curves["dataset"] == dataset]
        for strategy in main_strategies:
            data = dataset_curves[dataset_curves["strategy"] == strategy]
            x = data["forecast_index"].to_numpy()
            mean = data["mean_mae"].to_numpy()
            lower = data["min_mae"].to_numpy()
            upper = data["max_mae"].to_numpy()
            ax_main.plot(
                x,
                mean,
                label=DISPLAY_NAMES[strategy],
                color=COLORS[strategy],
                linewidth=1.7,
            )
            ax_main.fill_between(
                x,
                lower,
                upper,
                color=COLORS[strategy],
                alpha=0.08,
                linewidth=0,
            )

        linear = dataset_curves[dataset_curves["strategy"] == "linear_ogd"]
        linear_x = linear["forecast_index"].to_numpy()
        linear_mean = linear["mean_mae"].to_numpy()
        linear_lower = linear["min_mae"].to_numpy()
        linear_upper = linear["max_mae"].to_numpy()
        ax_linear.plot(
            linear_x,
            linear_mean,
            label=DISPLAY_NAMES["linear_ogd"],
            color=COLORS["linear_ogd"],
            linewidth=1.7,
        )
        ax_linear.fill_between(
            linear_x,
            linear_lower,
            linear_upper,
            color=COLORS["linear_ogd"],
            alpha=0.10,
            linewidth=0,
        )

        ax_main.set_title(dataset, loc="left", pad=9)
        ax_main.set_ylabel("Rolling MAE")
        ax_linear.set_ylabel("Rolling MAE")
        ax_linear.set_xlabel("Forecast index")
        ax_linear.legend(loc="upper right", frameon=False)

    for ax in axes.flat:
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("在线误差随时间的变化", x=0.07, y=0.995, ha="left", fontsize=18, fontweight="semibold")
    fig.text(
        0.07,
        0.925,
        f"{rolling_window} 步滚动 MAE；实线为 3 个随机种子的均值，阴影为种子范围",
        color="#69717B",
        fontsize=10,
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.62, 0.895),
        ncol=5,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.4,
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.81, bottom=0.09)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--etth1-dir",
        type=Path,
        default=Path("runs/etth1"),
        help="ETTh1 batch directory or its parent directory.",
    )
    parser.add_argument(
        "--etth2-dir",
        type=Path,
        default=Path("runs/etth2"),
        help="ETTh2 batch directory or its parent directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory for generated PNG files.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=336,
        help="Rolling window used for the online MAE plot.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    if args.rolling_window <= 0:
        raise ValueError("--rolling-window must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    etth1_summary, etth1_steps, etth1_batch = load_batch(args.etth1_dir, "ETTh1")
    etth2_summary, etth2_steps, etth2_batch = load_batch(args.etth2_dir, "ETTh2")
    summary = pd.concat((etth1_summary, etth2_summary), ignore_index=True)
    steps = pd.concat((etth1_steps, etth2_steps), ignore_index=True)

    accuracy_path = args.output_dir / "preliminary_forecast_accuracy.png"
    online_error_path = args.output_dir / "online_error_over_time.png"
    plot_accuracy(summary, accuracy_path, args.dpi)
    plot_online_error(steps, online_error_path, args.rolling_window, args.dpi)
    print(f"source=ETTh1:{etth1_batch.resolve()}")
    print(f"source=ETTh2:{etth2_batch.resolve()}")
    print(f"saved={accuracy_path.resolve()}")
    print(f"saved={online_error_path.resolve()}")


if __name__ == "__main__":
    main()

"""Create publication-ready plots for the full-data teacher-loss experiment."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import PercentFormatter


LAMBDA_LABELS = [
    "0",
    "0.01",
    "0.05",
    "0.10",
    "0.20",
    "0.40",
    "0.60",
    "0.80",
    "1.00",
    "1.20",
    "1.40",
    "1.60",
]
EXPECTED_LAMBDAS = [
    0.0,
    0.01,
    0.05,
    0.10,
    0.20,
    0.40,
    0.60,
    0.80,
    1.00,
    1.20,
    1.40,
    1.60,
]

COLORS = {
    "balanced_accuracy": "#0072B2",
    "accuracy": "#6B7280",
    "macro_f1": "#009E73",
    "overall": "#111827",
    "cs.AI": "#56B4E9",
    "cs.DS": "#E69F00",
    "cs.PL": "#CC79A7",
    "strong": "#D55E00",
    "grid": "#E5E7EB",
}

REQUIRED_COLUMNS = {
    "model",
    "lambda",
    "balanced_accuracy",
    "accuracy",
    "macro_f1",
    "mean_teacher_utility",
    "cs_ai_mean_teacher_utility",
    "cs_ds_mean_teacher_utility",
    "cs_pl_mean_teacher_utility",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "legend.fontsize": 9.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def padded_limits(
    values: list[float],
    minimum_pad: float,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> tuple[float, float]:
    low = min(values)
    high = max(values)
    span = high - low
    pad = max(minimum_pad, span * 0.28)

    if span == 0:
        pad = max(minimum_pad, abs(low) * 0.01)

    low -= pad
    high += pad

    if lower_bound is not None:
        low = max(lower_bound, low)

    if upper_bound is not None:
        high = min(upper_bound, high)

    return low, high


def style_axis(ax: Axes) -> None:
    ax.grid(
        axis="y",
        color=COLORS["grid"],
        linewidth=0.8,
    )
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D1D5DB")
    ax.spines["bottom"].set_color("#D1D5DB")

    ax.tick_params(
        color="#9CA3AF",
        labelcolor="#374151",
    )


def set_lambda_axis(ax: Axes) -> None:
    positions = list(range(len(LAMBDA_LABELS)))
    ax.set_xticks(positions)
    ax.set_xticklabels(
        LAMBDA_LABELS,
        rotation=35,
        ha="right",
    )
    ax.set_xlabel(r"Teacher-loss weight $\lambda$")

def overlay_baseline(
    ax: Axes,
    y: float,
    color: str,
) -> None:
    ax.scatter(
        [0],
        [y],
        s=65,
        facecolor="white",
        edgecolor=color,
        linewidth=1.8,
        zorder=5,
    )


def plot_performance(
    ax: Axes,
    frame: dict[str, list],
) -> None:
    x = list(range(len(frame["lambda"])))

    specifications = [
        (
            "balanced_accuracy",
            "Balanced accuracy",
            "-",
            2.8,
        ),
        (
            "accuracy",
            "Accuracy",
            "--",
            1.9,
        ),
        (
            "macro_f1",
            "Macro-F1",
            "-.",
            1.9,
        ),
    ]

    all_values: list[float] = []

    for column, label, linestyle, width in specifications:
        values = frame[column]
        all_values.extend(values)
        color = COLORS[column]

        ax.plot(
            x,
            values,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=width,
            marker="o",
            markersize=6.5,
            markeredgewidth=1.2,
            zorder=3,
        )

        overlay_baseline(
            ax,
            values[0],
            color,
        )

    balanced = frame["balanced_accuracy"]

    for position, value in zip(x, balanced):
        ax.annotate(
            f"{float(value):.3f}",
            (position, float(value)),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            color=COLORS["balanced_accuracy"],
            fontsize=9,
            fontweight="bold",
        )

    ax.set_ylim(
        *padded_limits(
            all_values,
            0.003,
            0.0,
            1.0,
        )
    )

    ax.yaxis.set_major_formatter(
        PercentFormatter(
            xmax=1.0,
            decimals=1,
        )
    )

    ax.set_ylabel("Held-out test performance")
    ax.set_title("A  Prediction performance")
    ax.legend(frameon=False, loc="best")

    set_lambda_axis(ax)
    style_axis(ax)


def plot_utility(
    ax: Axes,
    frame: dict[str, list],
) -> None:
    x = list(range(len(frame["lambda"])))

    specifications = [
        (
            "mean_teacher_utility",
            "Overall",
            "overall",
            "-",
            3.0,
        ),
        (
            "cs_ai_mean_teacher_utility",
            "cs.AI",
            "cs.AI",
            "--",
            1.9,
        ),
        (
            "cs_ds_mean_teacher_utility",
            "cs.DS",
            "cs.DS",
            "--",
            1.9,
        ),
        (
            "cs_pl_mean_teacher_utility",
            "cs.PL",
            "cs.PL",
            "--",
            1.9,
        ),
    ]

    all_values: list[float] = []

    for (
        column,
        label,
        color_key,
        linestyle,
        width,
    ) in specifications:
        values = frame[column]
        all_values.extend(values)
        color = COLORS[color_key]

        ax.plot(
            x,
            values,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=width,
            marker="o",
            markersize=6.5,
            markeredgewidth=1.2,
            zorder=3,
        )

        overlay_baseline(
            ax,
            values[0],
            color,
        )

    overall = frame["mean_teacher_utility"]

    for position, value in zip(x, overall):
        ax.annotate(
            f"{float(value):.3f}",
            (position, float(value)),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            color=COLORS["overall"],
            fontsize=9,
            fontweight="bold",
        )

    ax.set_ylim(
        *padded_limits(
            all_values,
            0.01,
        )
    )

    ax.set_ylabel("Mean Qwen teacher utility")
    ax.set_title("B  Teacher alignment")

    ax.legend(
        frameon=False,
        ncol=2,
        loc="best",
    )

    set_lambda_axis(ax)
    style_axis(ax)


def plot_tradeoff(
    ax: Axes,
    frame: dict[str, list],
) -> None:
    utility = frame["mean_teacher_utility"]
    performance = frame["balanced_accuracy"]

    point_colors = [
        "#9CA3AF",
        "#56B4E9",
        "#0072B2",
        "#009E73",
        "#F0E442",
        "#E69F00",
        "#D55E00",
        "#CC79A7",
        "#7A5195",
        "#003F5C",
        "#2F4B7C",
        "#665191",
    ]

    ax.plot(
        utility,
        performance,
        color="#9CA3AF",
        linewidth=1.6,
        zorder=1,
    )

    for index, (
        x_value,
        y_value,
        color,
        lambda_label,
    ) in enumerate(
        zip(
            utility,
            performance,
            point_colors,
            LAMBDA_LABELS,
        )
    ):
        ax.scatter(
            [x_value],
            [y_value],
            s=90,
            facecolor=(
                "white"
                if index == 0
                else color
            ),
            edgecolor=color,
            linewidth=2.0,
            zorder=3,
        )

        is_last = index == len(utility) - 1
        offset = (
            (-8, 8)
            if is_last
            else (6, 8 if index % 2 == 0 else -13)
        )
        alignment = "right" if is_last else "left"

        ax.annotate(
            rf"$\lambda$={lambda_label}",
            (x_value, y_value),
            xytext=offset,
            textcoords="offset points",
            ha=alignment,
            fontsize=9,
            color="#374151",
        )

    ax.set_xlim(
        *padded_limits(
            utility,
            0.005,
        )
    )

    ax.set_ylim(
        *padded_limits(
            performance,
            0.003,
            0.0,
            1.0,
        )
    )

    ax.yaxis.set_major_formatter(
        PercentFormatter(
            xmax=1.0,
            decimals=1,
        )
    )

    ax.set_xlabel("Mean Qwen teacher utility")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title(
        "C  Performance–alignment trade-off"
    )

    style_axis(ax)


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
) -> None:
    for suffix in ("png", "pdf"):
        path = output_dir / f"{stem}.{suffix}"

        fig.savefig(
            path,
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
        )

        print(f"Plot: {path}")


def save_single_panel(
    draw: Callable[
        [Axes, dict[str, list]],
        None,
    ],
    frame: dict[str, list],
    output_dir: Path,
    stem: str,
) -> None:
    fig, ax = plt.subplots(
        figsize=(7.2, 5.2),
        constrained_layout=True,
    )

    draw(ax, frame)
    save_figure(fig, output_dir, stem)
    plt.close(fig)


def load_comparison(
    comparison_path: Path,
) -> dict[str, list]:
    with comparison_path.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(handle)

        missing = (
            REQUIRED_COLUMNS
            - set(reader.fieldnames or [])
        )

        if missing:
            raise ValueError(
                "Comparison table is missing columns: "
                f"{sorted(missing)}"
            )

        rows = sorted(
            reader,
            key=lambda row: float(row["lambda"]),
        )

    if len(rows) != len(EXPECTED_LAMBDAS):
        raise ValueError(
            f"Expected {len(EXPECTED_LAMBDAS)} experiment variants; "
            f"found {len(rows)}"
        )

    return {
        column: (
            [row[column] for row in rows]
            if column == "model"
            else [
                float(row[column])
                for row in rows
            ]
        )
        for column in REQUIRED_COLUMNS
    }


def main() -> None:
    args = parse_args()
    frame = load_comparison(args.comparison)

    observed = frame["lambda"]
    expected = EXPECTED_LAMBDAS

    if not all(
        math.isclose(
            left,
            right,
            abs_tol=1e-12,
        )
        for left, right in zip(
            observed,
            expected,
        )
    ):
        raise ValueError(
            "Expected lambda values "
            f"{expected}; observed {observed}"
        )

    output_dir = (
        args.output_dir
        or args.comparison.parent
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    style()

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(18.0, 5.5),
        constrained_layout=True,
    )

    plot_performance(axes[0], frame)
    plot_utility(axes[1], frame)
    plot_tradeoff(axes[2], frame)

    figure.suptitle(
        "IRIS performance–teacher alignment trade-off",
        fontsize=18,
        fontweight="bold",
    )

    figure.text(
        0.5,
        -0.035,
        (
            "Teacher utility measures alignment with "
            "Qwen's preference; it is not independent "
            "evidence of explanation quality. "
            "Axes are scaled to show between-model "
            "differences."
        ),
        ha="center",
        fontsize=9,
        color="#6B7280",
    )

    save_figure(
        figure,
        output_dir,
        "primary_tradeoff",
    )

    plt.close(figure)

    save_single_panel(
        plot_performance,
        frame,
        output_dir,
        "performance_vs_lambda",
    )

    save_single_panel(
        plot_utility,
        frame,
        output_dir,
        "teacher_utility_vs_lambda",
    )

    save_single_panel(
        plot_tradeoff,
        frame,
        output_dir,
        "performance_utility_tradeoff",
    )


if __name__ == "__main__":
    main()

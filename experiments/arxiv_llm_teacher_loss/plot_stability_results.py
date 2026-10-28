"""Plot repeated-seed performance, AUC, utility, and trade-offs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import PercentFormatter


COLORS = {
    "balanced_accuracy": "#0072B2",
    "accuracy": "#6B7280",
    "macro_f1": "#009E73",
    "macro_ovr_roc_auc": "#D55E00",
    "cs_ai_roc_auc": "#56B4E9",
    "cs_ds_roc_auc": "#E69F00",
    "cs_pl_roc_auc": "#CC79A7",
    "mean_teacher_utility": "#111827",
    "cs_ai_mean_teacher_utility": "#56B4E9",
    "cs_ds_mean_teacher_utility": "#E69F00",
    "cs_pl_mean_teacher_utility": "#CC79A7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, float | str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: float(row["lambda"]))
    converted = []
    for row in rows:
        converted.append(
            {
                key: value
                if key in {"model", "variant"}
                else float(value)
                for key, value in row.items()
            }
        )
    return converted


def style_axis(ax: Axes) -> None:
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def lambda_labels(rows: list[dict[str, float | str]]) -> list[str]:
    return [f"{float(row['lambda']):g}" for row in rows]


def set_lambda_axis(
    ax: Axes,
    rows: list[dict[str, float | str]],
) -> None:
    positions = list(range(len(rows)))
    ax.set_xticks(positions)
    ax.set_xticklabels(lambda_labels(rows), rotation=35, ha="right")
    ax.set_xlabel(r"Teacher-loss weight $\lambda$ (log-spaced grid)")


def error_line(
    ax: Axes,
    rows: list[dict[str, float | str]],
    metric: str,
    label: str,
    linestyle: str = "-",
) -> None:
    x = list(range(len(rows)))
    values = [float(row[metric]) for row in rows]
    errors = [float(row[f"{metric}_sd"]) for row in rows]
    ax.errorbar(
        x,
        values,
        yerr=errors,
        label=label,
        color=COLORS[metric],
        linestyle=linestyle,
        marker="o",
        linewidth=2.0,
        markersize=5.5,
        capsize=3,
    )


def plot_performance(
    ax: Axes,
    rows: list[dict[str, float | str]],
) -> None:
    error_line(ax, rows, "balanced_accuracy", "Balanced accuracy")
    error_line(ax, rows, "accuracy", "Accuracy", "--")
    error_line(ax, rows, "macro_f1", "Macro-F1", "-.")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    ax.set_ylabel("Held-out test performance")
    ax.set_title("A  Classification performance")
    ax.legend(frameon=False)
    set_lambda_axis(ax, rows)
    style_axis(ax)


def plot_auc(
    ax: Axes,
    rows: list[dict[str, float | str]],
) -> None:
    error_line(ax, rows, "macro_ovr_roc_auc", "Macro OVR AUC")
    error_line(ax, rows, "cs_ai_roc_auc", "cs.AI AUC", "--")
    error_line(ax, rows, "cs_ds_roc_auc", "cs.DS AUC", "--")
    error_line(ax, rows, "cs_pl_roc_auc", "cs.PL AUC", "--")
    ax.set_ylabel("One-vs-rest ROC-AUC")
    ax.set_title("B  Discrimination")
    ax.legend(frameon=False, ncol=2)
    set_lambda_axis(ax, rows)
    style_axis(ax)


def plot_utility(
    ax: Axes,
    rows: list[dict[str, float | str]],
) -> None:
    error_line(ax, rows, "mean_teacher_utility", "Overall")
    error_line(
        ax,
        rows,
        "cs_ai_mean_teacher_utility",
        "cs.AI",
        "--",
    )
    error_line(
        ax,
        rows,
        "cs_ds_mean_teacher_utility",
        "cs.DS",
        "--",
    )
    error_line(
        ax,
        rows,
        "cs_pl_mean_teacher_utility",
        "cs.PL",
        "--",
    )
    ax.set_ylabel("Mean Qwen teacher utility")
    ax.set_title("C  Teacher alignment")
    ax.legend(frameon=False, ncol=2)
    set_lambda_axis(ax, rows)
    style_axis(ax)


def plot_tradeoff(
    ax: Axes,
    rows: list[dict[str, float | str]],
) -> None:
    utility = [float(row["mean_teacher_utility"]) for row in rows]
    utility_sd = [float(row["mean_teacher_utility_sd"]) for row in rows]
    performance = [float(row["balanced_accuracy"]) for row in rows]
    performance_sd = [float(row["balanced_accuracy_sd"]) for row in rows]
    ax.plot(utility, performance, color="#9CA3AF", linewidth=1.5)
    ax.errorbar(
        utility,
        performance,
        xerr=utility_sd,
        yerr=performance_sd,
        fmt="o",
        color="#0072B2",
        ecolor="#9CA3AF",
        capsize=3,
    )
    for x_value, y_value, label in zip(
        utility,
        performance,
        lambda_labels(rows),
        strict=True,
    ):
        ax.annotate(
            rf"$\lambda$={label}",
            (x_value, y_value),
            xytext=(5, 6),
            textcoords="offset points",
            fontsize=8.5,
        )
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    ax.set_xlabel("Mean Qwen teacher utility")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("D  Performance–alignment trade-off")
    style_axis(ax)


def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    for suffix in ("png", "pdf"):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        print(f"Plot: {path}")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.comparison)
    output_dir = args.output_dir or args.comparison.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(15.5, 10.5),
        constrained_layout=True,
    )
    plot_performance(axes[0, 0], rows)
    plot_auc(axes[0, 1], rows)
    plot_utility(axes[1, 0], rows)
    plot_tradeoff(axes[1, 1], rows)
    figure.suptitle(
        "Repeated-seed IRIS teacher-loss experiment",
        fontsize=17,
        fontweight="bold",
    )
    save(figure, output_dir, "stability_summary")
    plt.close(figure)


if __name__ == "__main__":
    main()

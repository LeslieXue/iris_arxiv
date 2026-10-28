"""Aggregate repeated-seed teacher-loss results by lambda."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import tomllib


METRIC_COLUMNS = [
    "balanced_accuracy",
    "accuracy",
    "macro_f1",
    "macro_ovr_roc_auc",
    "cs_ai_roc_auc",
    "cs_ds_roc_auc",
    "cs_pl_roc_auc",
    "mean_teacher_utility",
    "cs_ai_mean_teacher_utility",
    "cs_ds_mean_teacher_utility",
    "cs_pl_mean_teacher_utility",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def display_name(weight: float) -> str:
    if weight == 0:
        return "Baseline"
    return f"Teacher lambda {weight:g}"


def markdown_table(rows: list[dict]) -> str:
    columns = list(rows[0])
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = [
            f"{row[column]:.6f}"
            if isinstance(row[column], float)
            else str(row[column])
            for column in columns
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(statistics.stdev(values))


def main() -> None:
    args = parse_args()
    with args.experiment_config.open("rb") as handle:
        experiment = tomllib.load(handle)

    training = experiment["training"]
    seeds = [int(seed) for seed in training["seeds"]]
    results_dir = args.results_dir or Path(training["metadata_dir"])

    run_rows: list[dict] = []
    aggregate_rows: list[dict] = []

    for variant in training["variants"]:
        name = str(variant["name"])
        weight = float(variant["weight"])
        variant_rows = []

        for seed in seeds:
            run_name = f"{name}.seed-{seed}"
            test = load_json(results_dir / f"{run_name}.test_metrics.json")
            alignment = load_json(
                results_dir / f"{run_name}.teacher_alignment.json"
            )
            metadata = load_json(results_dir / f"{run_name}.training.json")
            overall = alignment["overall"]
            per_class = alignment["per_class"]

            row = {
                "model": display_name(weight),
                "variant": name,
                "lambda": weight,
                "seed": seed,
                "epochs_trained": int(metadata["epochs_trained"]),
                "best_epoch": int(metadata["best_epoch"]),
                "balanced_accuracy": float(test["balanced_accuracy"]),
                "accuracy": float(test["accuracy"]),
                "macro_f1": float(test["macro_f1"]),
                "macro_ovr_roc_auc": float(test["macro_ovr_roc_auc"]),
                "cs_ai_roc_auc": float(
                    test["per_class"]["cs.AI"]["roc_auc_ovr"]
                ),
                "cs_ds_roc_auc": float(
                    test["per_class"]["cs.DS"]["roc_auc_ovr"]
                ),
                "cs_pl_roc_auc": float(
                    test["per_class"]["cs.PL"]["roc_auc_ovr"]
                ),
                "mean_teacher_utility": float(
                    overall["mean_teacher_utility"]
                ),
                "cs_ai_mean_teacher_utility": float(
                    per_class["cs.AI"]["mean_teacher_utility"]
                ),
                "cs_ds_mean_teacher_utility": float(
                    per_class["cs.DS"]["mean_teacher_utility"]
                ),
                "cs_pl_mean_teacher_utility": float(
                    per_class["cs.PL"]["mean_teacher_utility"]
                ),
            }
            variant_rows.append(row)
            run_rows.append(row)

        aggregate = {
            "model": display_name(weight),
            "variant": name,
            "lambda": weight,
            "n_runs": len(variant_rows),
            "mean_epochs_trained": mean(
                [float(row["epochs_trained"]) for row in variant_rows]
            ),
            "mean_best_epoch": mean(
                [float(row["best_epoch"]) for row in variant_rows]
            ),
        }
        for metric in METRIC_COLUMNS:
            values = [float(row[metric]) for row in variant_rows]
            aggregate[metric] = mean(values)
            aggregate[f"{metric}_sd"] = sample_sd(values)
        aggregate_rows.append(aggregate)

    write_csv(results_dir / "comparison_runs.csv", run_rows)
    write_csv(results_dir / "comparison.csv", aggregate_rows)
    (results_dir / "comparison_runs.json").write_text(
        json.dumps(run_rows, indent=2) + "\n",
        encoding="utf-8",
    )
    (results_dir / "comparison.json").write_text(
        json.dumps(
            {
                "aggregation": (
                    "Mean and sample SD across fixed-split training seeds"
                ),
                "split_seed": int(training["split_seed"]),
                "training_seeds": seeds,
                "variants": aggregate_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (results_dir / "comparison.md").write_text(
        "# Repeated-seed IRIS teacher-loss comparison\n\n"
        "> Values are means across training seeds; `_sd` columns are sample "
        "standard deviations. The document split is fixed.\n\n"
        + markdown_table(aggregate_rows)
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps(aggregate_rows, indent=2))
    print(f"Run-level results: {results_dir / 'comparison_runs.csv'}")
    print(f"Aggregated results: {results_dir / 'comparison.csv'}")


if __name__ == "__main__":
    main()

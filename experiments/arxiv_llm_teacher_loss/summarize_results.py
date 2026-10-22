"""Create one comparison table for all teacher-loss variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl
import tomllib


DISPLAY_NAMES = {
    "baseline": "Baseline",
    "teacher_w001": "Teacher weak",
    "teacher_w005": "Teacher medium",
    "teacher_w010": "Teacher strong",
    "teacher_w020": "Teacher lambda 0.20",
    "teacher_w040": "Teacher lambda 0.40",
    "teacher_w060": "Teacher lambda 0.60",
    "teacher_w080": "Teacher lambda 0.80",
    "teacher_w100": "Teacher lambda 1.00",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def markdown_table(frame: pl.DataFrame) -> str:
    columns = frame.columns
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.iter_rows():
        values = [
            f"{value:.6f}" if isinstance(value, float) else str(value)
            for value in row
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    with args.experiment_config.open("rb") as handle:
        experiment = tomllib.load(handle)
    results_dir = args.results_dir or Path(
        experiment["training"]["metadata_dir"]
    )

    rows = []
    for variant in experiment["training"]["variants"]:
        name = variant["name"]
        test = load_json(results_dir / f"{name}.test_metrics.json")
        alignment = load_json(
            results_dir / f"{name}.teacher_alignment.json"
        )
        overall = alignment["overall"]
        per_class = alignment["per_class"]

        rows.append(
            {
                "model": DISPLAY_NAMES[name],
                "variant": name,
                "lambda": float(variant["weight"]),
                "balanced_accuracy": test["balanced_accuracy"],
                "accuracy": test["accuracy"],
                "macro_f1": test["macro_f1"],
                "mean_teacher_utility": overall[
                    "mean_teacher_utility"
                ],
                "cs_ai_mean_teacher_utility": per_class["cs.AI"][
                    "mean_teacher_utility"
                ],
                "cs_ds_mean_teacher_utility": per_class["cs.DS"][
                    "mean_teacher_utility"
                ],
                "cs_pl_mean_teacher_utility": per_class["cs.PL"][
                    "mean_teacher_utility"
                ],
            }
        )

    frame = pl.DataFrame(rows)
    json_path = results_dir / "comparison.json"
    csv_path = results_dir / "comparison.csv"
    md_path = results_dir / "comparison.md"
    json_path.write_text(
        json.dumps(
            {
                "important_caveat": (
                    "Performance metrics are held-out task metrics. Teacher "
                    "alignment measures compliance with Qwen's preference and "
                    "is not independent explanation-quality validation."
                ),
                "variants": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    frame.write_csv(csv_path)
    md_path.write_text(
        "# IRIS teacher-loss experiment comparison\n\n"
        "> Teacher alignment measures Qwen preference compliance; it is not "
        "independent proof of explanation quality.\n\n"
        + markdown_table(frame)
        + "\n",
        encoding="utf-8",
    )
    print(frame)
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")


if __name__ == "__main__":
    main()

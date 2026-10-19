"""Create one comparison table for all teacher-loss variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl
import tomllib


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
        alignment = load_json(results_dir / f"{name}.teacher_alignment.json")
        overall = alignment["overall"]
        rows.append(
            {
                "variant": name,
                "teacher_enabled": bool(variant["enabled"]),
                "score_mode": variant["score_mode"],
                "teacher_weight": float(variant["weight"]),
                "accuracy": test["accuracy"],
                "balanced_accuracy": test["balanced_accuracy"],
                "macro_f1": test["macro_f1"],
                "cross_entropy": test["cross_entropy"],
                "mean_teacher_utility": overall["mean_teacher_utility"],
                "mean_target_evidence": overall["mean_target_evidence"],
                "high_evidence_rate": overall["high_evidence_rate"],
                "zero_evidence_rate": overall["zero_evidence_rate"],
                "keyword_only_risk": overall["mean_keyword_only_risk"],
                "direct_label_disclosure": overall[
                    "mean_direct_label_disclosure"
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

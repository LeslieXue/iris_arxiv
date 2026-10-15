"""Download arXiv11 and export selected classes as individual text files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from datasets import load_dataset


SELECTED_CATEGORIES = {"cs.AI", "cs.DS", "cs.PL"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/arxiv11/data"),
    )
    parser.add_argument(
        "--dataset-config",
        choices=("default", "no_ref"),
        default="default",
        help="Use no_ref to remove explicit category references from article text",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = load_dataset(
        "ccdv/arxiv-classification",
        args.dataset_config,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output.parent / "manifest.csv"

    manifest_rows = []
    counts = {category: 0 for category in sorted(SELECTED_CATEGORIES)}

    for split_name, split in dataset.items():
        label_names = split.features["label"].names

        for row_index, row in enumerate(split):
            category = label_names[row["label"]]

            if category not in SELECTED_CATEGORIES:
                continue

            category_dir = args.output / category
            category_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{split_name}_{row_index:06d}.txt"
            output_path = category_dir / filename
            output_path.write_text(row["text"], encoding="utf-8")

            relative_path = output_path.relative_to(args.output).as_posix()

            manifest_rows.append(
                {
                    "source_path": relative_path,
                    "original_split": split_name,
                    "original_row_index": row_index,
                    "category": category,
                }
            )
            counts[category] += 1

    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "source_path",
                "original_split",
                "original_row_index",
                "category",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Exported {len(manifest_rows)} documents")
    for category, count in counts.items():
        print(f"{category}: {count}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
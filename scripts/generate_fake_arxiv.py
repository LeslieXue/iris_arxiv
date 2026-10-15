"""Generate synthetic long arXiv-style documents for local testing."""

from __future__ import annotations

import argparse
from pathlib import Path


CATEGORY_CONTENT = {
    "cs.AI": (
        "Artificial Intelligence",
        "artificial intelligence neural networks machine learning reasoning "
        "planning agents representation inference optimization prediction",
    ),
    "cs.DS": (
        "Data Structures",
        "data structures graph algorithms trees hashing dynamic programming "
        "complexity shortest paths indexing search computational efficiency",
    ),
    "cs.PL": (
        "Programming Languages",
        "programming languages type systems compilers semantics interpreters "
        "static analysis verification syntax runtime functional programming",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/fake_arxiv/raw/arxiv11/data"),
    )
    parser.add_argument(
        "--documents-per-class",
        type=int,
        default=20,
    )
    return parser.parse_args()


def build_document(category: str, title: str, vocabulary: str, index: int) -> str:
    sections = [
        (
            f"Section {section}. This is a synthetic {category} research document "
            f"number {index}. The paper studies {vocabulary}. "
            "The methodology, experiments, results, limitations, and conclusions "
            "are described for local IRIS pipeline testing. "
            f"Document marker {category}-{index}-{section}."
        )
        for section in range(30)
    ]

    return "\n\n".join(
        [
            "SYNTHETIC ARXIV DOCUMENT",
            f"Title: {title} Study {index}",
            f"Category: {category}",
            *sections,
        ]
    )


def main() -> None:
    args = parse_args()
    total = 0

    for category, (title, vocabulary) in CATEGORY_CONTENT.items():
        category_dir = args.output / category
        category_dir.mkdir(parents=True, exist_ok=True)

        for index in range(args.documents_per_class):
            text = build_document(
                category=category,
                title=title,
                vocabulary=vocabulary,
                index=index,
            )
            path = category_dir / f"fake_{index:03d}.txt"
            path.write_text(text, encoding="utf-8")
            total += 1

    print(f"Generated {total} synthetic documents in {args.output}")


if __name__ == "__main__":
    main()
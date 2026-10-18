import argparse
import json
import statistics
import tomllib
from collections import Counter
from pathlib import Path

import polars as pl


def parse_args():
    default_config = Path(__file__).with_name("config.target.toml")
    parser = argparse.ArgumentParser(
        description="Describe IRIS and Qwen selections in the top-32 pilot"
    )
    parser.add_argument(
        "--config",
        default=default_config,
        help="Target-conditioned experiment configuration",
    )
    return parser.parse_args()


def load_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def key(row: dict) -> tuple[int, int, int]:
    return (
        int(row["document_id"]),
        int(row["query_id"]),
        int(row["chunk_id"]),
    )


def compact_text(text: str, limit: int = 220) -> str:
    compact = " ".join(text.split()).replace("|", "\\|")
    return compact if len(compact) <= limit else compact[:limit] + "..."


def selection_block(title: str, row: dict) -> str:
    text = " ".join(row["chunk_text"].split())
    return "\n".join(
        [
            f"#### {title}",
            "",
            (
                f"- Chunk `{row['chunk_id']}`; original candidate rank "
                f"{row['candidate_rank']}"
            ),
            (
                f"- Scores: target evidence={row['target_evidence']}, "
                f"substantive={row['substantive_relevance']}, "
                f"specificity={row['specificity']}, "
                f"context={row['context_completeness']}"
            ),
            (
                f"- Risks: disclosure={row['direct_label_disclosure']}, "
                f"keyword={row['keyword_only_risk']}"
            ),
            f"- Teacher reason: {row['brief_reason']}",
            "",
            text,
            "",
        ]
    )


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path

    cfg = load_toml(config_path)
    prompt_version = str(cfg["teacher"]["prompt_version"])
    target_root = Path(cfg["paths"]["output_dir"])
    if not target_root.is_absolute():
        target_root = repo_root / target_root
    pilot_dir = target_root / str(cfg["pilot"]["output_subdir"])

    manifest_path = pilot_dir / "pilot_manifest.parquet"
    score_path = (
        pilot_dir / f"qwen_scores.{prompt_version}.pilot.jsonl"
    )
    metadata_path = pilot_dir / "pilot_metadata.json"
    scored_path = (
        pilot_dir / f"pilot_scored.{prompt_version}.parquet"
    )
    summary_path = (
        pilot_dir / f"pilot_comparison.{prompt_version}.json"
    )
    report_path = (
        pilot_dir / f"pilot_comparison.{prompt_version}.md"
    )

    manifest_rows = pl.read_parquet(manifest_path).to_dicts()
    score_rows = read_jsonl(score_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    manifest_by_key = {key(row): row for row in manifest_rows}
    score_by_key = {key(row): row for row in score_rows}
    if len(manifest_by_key) != len(manifest_rows):
        raise ValueError("Duplicate keys in pilot manifest")
    if len(score_by_key) != len(score_rows):
        raise ValueError("Duplicate keys in pilot scores")
    if set(manifest_by_key) != set(score_by_key):
        raise ValueError("Pilot score keys do not match manifest")

    rows = []
    for record_key in sorted(manifest_by_key):
        manifest = manifest_by_key[record_key]
        score = score_by_key[record_key]
        rows.append(
            {
                **manifest,
                "target_label": str(score["target_label"]),
                "target_evidence": int(score["target_evidence"]),
                "substantive_relevance": int(
                    score["substantive_relevance"]
                ),
                "specificity": int(score["specificity"]),
                "context_completeness": int(
                    score["context_completeness"]
                ),
                "direct_label_disclosure": int(
                    score["direct_label_disclosure"]
                ),
                "keyword_only_risk": int(score["keyword_only_risk"]),
                "brief_reason": str(score["brief_reason"]),
            }
        )

    query_summaries = []
    original_rows = []
    teacher_rows = []

    for query_id in sorted({int(row["query_id"]) for row in rows}):
        candidates = sorted(
            (row for row in rows if int(row["query_id"]) == query_id),
            key=lambda row: int(row["candidate_rank"]),
        )
        originals = [
            row for row in candidates if row["original_iris_selection"]
        ]
        if len(originals) != 1:
            raise ValueError(
                f"Query {query_id} has {len(originals)} original selections"
            )
        original = originals[0]
        maximum = max(row["target_evidence"] for row in candidates)
        maximum_rows = [
            row for row in candidates if row["target_evidence"] == maximum
        ]
        teacher = min(
            maximum_rows,
            key=lambda row: int(row["candidate_rank"]),
        )
        changed = int(teacher["chunk_id"]) != int(original["chunk_id"])

        for row in candidates:
            row["qwen_selected"] = (
                int(row["chunk_id"]) == int(teacher["chunk_id"])
            )
            row["qwen_max_score_tie_count"] = len(maximum_rows)

        original_rows.append(original)
        teacher_rows.append(teacher)
        query_summaries.append(
            {
                "query_id": query_id,
                "changed": changed,
                "distinct_target_scores": len(
                    {row["target_evidence"] for row in candidates}
                ),
                "maximum_target_evidence": maximum,
                "maximum_score_tie_count": len(maximum_rows),
                "original_chunk_id": int(original["chunk_id"]),
                "original_target_evidence": int(
                    original["target_evidence"]
                ),
                "qwen_chunk_id": int(teacher["chunk_id"]),
                "qwen_original_rank": int(teacher["candidate_rank"]),
                "qwen_target_evidence": int(teacher["target_evidence"]),
                "target_evidence_delta": int(
                    teacher["target_evidence"]
                    - original["target_evidence"]
                ),
            }
        )

    score_distribution = Counter(row["target_evidence"] for row in rows)
    original_distribution = Counter(
        row["target_evidence"] for row in original_rows
    )
    qwen_distribution = Counter(
        row["target_evidence"] for row in teacher_rows
    )
    summary = {
        "important_caveat": (
            "This report describes how Qwen changes IRIS selections. It does "
            "not infer relative selection quality. Qwen's own scores are not "
            "independent evaluation labels."
        ),
        "prompt_version": prompt_version,
        "document_id": int(metadata["document_id"]),
        "target_label": str(metadata["target_label"]),
        "source_path": str(metadata["source_path"]),
        "candidate_records": len(rows),
        "queries": len(query_summaries),
        "changed_queries": sum(row["changed"] for row in query_summaries),
        "unchanged_queries": sum(
            not row["changed"] for row in query_summaries
        ),
        "queries_with_unique_qwen_maximum": sum(
            row["maximum_score_tie_count"] == 1
            for row in query_summaries
        ),
        "queries_with_tied_qwen_maximum": sum(
            row["maximum_score_tie_count"] > 1
            for row in query_summaries
        ),
        "queries_with_multiple_score_levels": sum(
            row["distinct_target_scores"] > 1 for row in query_summaries
        ),
        "candidate_target_evidence_distribution": {
            str(score): score_distribution.get(score, 0)
            for score in range(5)
        },
        "original_selection_target_evidence_distribution": {
            str(score): original_distribution.get(score, 0)
            for score in range(5)
        },
        "qwen_selection_target_evidence_distribution": {
            str(score): qwen_distribution.get(score, 0)
            for score in range(5)
        },
        "mean_original_selection_target_evidence": statistics.mean(
            row["target_evidence"] for row in original_rows
        ),
        "mean_qwen_selection_target_evidence": statistics.mean(
            row["target_evidence"] for row in teacher_rows
        ),
        "query_comparisons": query_summaries,
    }

    pl.DataFrame(rows).sort(
        "query_id", "candidate_rank"
    ).write_parquet(scored_path)
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    report = [
        "# IRIS and Qwen top-32 pilot comparison",
        "",
        summary["important_caveat"],
        "",
        f"- Document: `{summary['document_id']}`",
        f"- Target: `{summary['target_label']}`",
        f"- Source: `{summary['source_path']}`",
        f"- Changed queries: {summary['changed_queries']}/8",
        (
            f"- Queries with tied Qwen maximum: "
            f"{summary['queries_with_tied_qwen_maximum']}/8"
        ),
        (
            f"- Queries with multiple score levels: "
            f"{summary['queries_with_multiple_score_levels']}/8"
        ),
        "",
    ]

    for comparison in query_summaries:
        query_id = comparison["query_id"]
        candidates = sorted(
            (row for row in rows if int(row["query_id"]) == query_id),
            key=lambda row: int(row["candidate_rank"]),
        )
        original = next(
            row for row in candidates if row["original_iris_selection"]
        )
        teacher = next(row for row in candidates if row["qwen_selected"])

        report.extend(
            [
                f"## Query {query_id}",
                "",
                f"- Selection changed: `{comparison['changed']}`",
                (
                    f"- Highest target-evidence tie count: "
                    f"{comparison['maximum_score_tie_count']}"
                ),
                "",
                "| IRIS rank | chunk | target | substantive | specificity | context | disclosure | keyword risk | Qwen selected | excerpt |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
            ]
        )
        for row in candidates:
            report.append(
                "| "
                + " | ".join(
                    [
                        str(row["candidate_rank"]),
                        str(row["chunk_id"]),
                        str(row["target_evidence"]),
                        str(row["substantive_relevance"]),
                        str(row["specificity"]),
                        str(row["context_completeness"]),
                        str(row["direct_label_disclosure"]),
                        str(row["keyword_only_risk"]),
                        "yes" if row["qwen_selected"] else "",
                        compact_text(row["chunk_text"]),
                    ]
                )
                + " |"
            )
        report.extend(["", selection_block("Original IRIS selection", original)])
        if comparison["changed"]:
            report.append(selection_block("Qwen selection", teacher))
        else:
            report.extend(
                [
                    "#### Qwen selection",
                    "",
                    "Qwen retained the original IRIS selection.",
                    "",
                ]
            )

    report_path.write_text("\n".join(report), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Scored pilot: {scored_path}")
    print(f"Comparison summary: {summary_path}")
    print(f"Comparison report: {report_path}")


if __name__ == "__main__":
    main()

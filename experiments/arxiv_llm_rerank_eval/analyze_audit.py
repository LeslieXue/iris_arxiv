import argparse
import json
import tomllib
from pathlib import Path

import polars as pl


EVIDENCE_FIELDS = {
    "cs.AI": "ai_evidence",
    "cs.DS": "ds_evidence",
    "cs.PL": "pl_evidence",
}

QUALITY_FIELDS = [
    "substantive_relevance",
    "specificity",
    "context_completeness",
    "direct_label_disclosure",
    "keyword_only_risk",
]

INSUFFICIENT_MAX_SCORE = 1


def parse_args():
    default_config = Path(__file__).with_name("config.toml")

    parser = argparse.ArgumentParser(
        description="Analyze the teacher-v2 validation audit"
    )
    parser.add_argument(
        "--config",
        default=default_config,
        help="Experiment configuration",
    )
    return parser.parse_args()


def load_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [
            json.loads(line)
            for line in handle
            if line.strip()
        ]


def key(record: dict) -> tuple[int, int, int]:
    return (
        int(record["document_id"]),
        int(record["query_id"]),
        int(record["chunk_id"]),
    )


def mean(values: list[float]) -> float | None:
    if not values:
        return None

    return sum(values) / len(values)


def summarize(rows: list[dict]) -> dict:
    unique_rows = [
        row
        for row in rows
        if row["prediction_status"] == "unique"
    ]

    aligned_unique = [
        row
        for row in unique_rows
        if row["document_label_aligned"]
    ]

    return {
        "records": len(rows),
        "unique_informative_predictions": len(unique_rows),
        "unique_informative_coverage": (
            len(unique_rows) / len(rows)
            if rows
            else None
        ),
        "document_label_alignment_all": (
            sum(row["document_label_aligned"] for row in rows)
            / len(rows)
            if rows
            else None
        ),
        "document_label_alignment_when_unique": (
            len(aligned_unique) / len(unique_rows)
            if unique_rows
            else None
        ),
        "insufficient_evidence": sum(
            row["prediction_status"] == "insufficient"
            for row in rows
        ),
        "ambiguous_top_score": sum(
            row["prediction_status"] == "ambiguous"
            for row in rows
        ),
        "mean_true_class_evidence": mean(
            [row["true_class_evidence"] for row in rows]
        ),
        "mean_true_class_margin": mean(
            [row["true_class_margin"] for row in rows]
        ),
        "mean_max_evidence": mean(
            [row["max_evidence"] for row in rows]
        ),
    }


def format_case(row: dict) -> str:
    text = " ".join(row["chunk_text"].split())

    if len(text) > 900:
        text = text[:900] + "..."

    return "\n".join(
        [
            (
                f"### Document {row['document_id']}, "
                f"query {row['query_id']}, "
                f"chunk {row['chunk_id']}"
            ),
            "",
            (
                f"- Document label: "
                f"`{row['document_label_name']}`"
            ),
            (
                f"- Predicted evidence label: "
                f"`{row['predicted_label']}`"
            ),
            (
                f"- Prediction status: "
                f"`{row['prediction_status']}`"
            ),
            (
                f"- Rank band: `{row['rank_band']}`; "
                f"candidate rank: {row['candidate_rank']}"
            ),
            (
                f"- Evidence: AI={row['ai_evidence']}, "
                f"DS={row['ds_evidence']}, "
                f"PL={row['pl_evidence']}"
            ),
            (
                f"- True-class margin: "
                f"{row['true_class_margin']}"
            ),
            (
                f"- Quality: substantive="
                f"{row['substantive_relevance']}, "
                f"specificity={row['specificity']}, "
                f"context={row['context_completeness']}"
            ),
            (
                f"- Risks: disclosure="
                f"{row['direct_label_disclosure']}, "
                f"keyword={row['keyword_only_risk']}"
            ),
            f"- Teacher reason: {row['brief_reason']}",
            "",
            "**Chunk excerpt**",
            "",
            text,
            "",
        ]
    )


def main():
    args = parse_args()

    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    config_path = Path(args.config)

    if not config_path.is_absolute():
        config_path = repo_root / config_path

    cfg = load_toml(config_path)
    output_dir = Path(cfg["paths"]["output_dir"])

    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir

    manifest_path = output_dir / "audit_manifest.parquet"
    input_path = output_dir / "teacher_inputs.audit.jsonl"
    score_path = (
        output_dir
        / "qwen_scores.teacher_v2.audit.jsonl"
    )

    scored_path = output_dir / "audit_scored.parquet"
    metrics_path = output_dir / "audit_metrics.json"
    review_path = output_dir / "audit_review.md"

    manifest_rows = pl.read_parquet(
        manifest_path
    ).to_dicts()
    input_rows = read_jsonl(input_path)
    score_rows = read_jsonl(score_path)

    manifest_by_key = {
        key(row): row
        for row in manifest_rows
    }
    input_by_key = {
        key(row): row
        for row in input_rows
    }
    score_by_key = {
        key(row): row
        for row in score_rows
    }

    expected_keys = set(manifest_by_key)

    if set(input_by_key) != expected_keys:
        raise ValueError(
            "Audit input keys do not match the manifest"
        )

    if set(score_by_key) != expected_keys:
        raise ValueError(
            "Audit score keys do not match the manifest"
        )

    joined_rows = []

    for record_key in sorted(expected_keys):
        manifest = manifest_by_key[record_key]
        teacher_input = input_by_key[record_key]
        score = score_by_key[record_key]

        evidence = {
            label_name: int(score[field])
            for label_name, field in EVIDENCE_FIELDS.items()
        }

        max_evidence = max(evidence.values())
        top_labels = sorted(
            label_name
            for label_name, value in evidence.items()
            if value == max_evidence
        )

        if max_evidence <= INSUFFICIENT_MAX_SCORE:
            prediction_status = "insufficient"
            predicted_label = "insufficient"
        elif len(top_labels) > 1:
            prediction_status = "ambiguous"
            predicted_label = "|".join(top_labels)
        else:
            prediction_status = "unique"
            predicted_label = top_labels[0]

        document_label = str(manifest["label_name"])
        true_class_evidence = evidence[document_label]
        maximum_other_evidence = max(
            value
            for label_name, value in evidence.items()
            if label_name != document_label
        )
        true_class_margin = (
            true_class_evidence
            - maximum_other_evidence
        )

        joined_rows.append(
            {
                "audit_id": int(manifest["audit_id"]),
                "split": str(manifest["split"]),
                "rank_band": str(manifest["rank_band"]),
                "document_id": int(manifest["document_id"]),
                "query_id": int(manifest["query_id"]),
                "chunk_id": int(manifest["chunk_id"]),
                "candidate_rank": int(
                    manifest["candidate_rank"]
                ),
                "vector_distance": float(
                    manifest["vector_distance"]
                ),
                "document_label_id": int(manifest["label"]),
                "document_label_name": document_label,
                "source_path": str(manifest["source_path"]),
                "chunk_text": str(
                    teacher_input["chunk_text"]
                ),
                "ai_evidence": evidence["cs.AI"],
                "ds_evidence": evidence["cs.DS"],
                "pl_evidence": evidence["cs.PL"],
                **{
                    field: int(score[field])
                    for field in QUALITY_FIELDS
                },
                "brief_reason": str(
                    score["brief_reason"]
                ),
                "max_evidence": max_evidence,
                "predicted_label": predicted_label,
                "prediction_status": prediction_status,
                "true_class_evidence": true_class_evidence,
                "maximum_other_evidence": (
                    maximum_other_evidence
                ),
                "true_class_margin": true_class_margin,
                "document_label_aligned": (
                    prediction_status == "unique"
                    and predicted_label == document_label
                ),
            }
        )

    scored = pl.DataFrame(joined_rows)
    scored.write_parquet(scored_path)

    per_class = {
        label_name: summarize(
            [
                row
                for row in joined_rows
                if row["document_label_name"] == label_name
            ]
        )
        for label_name in EVIDENCE_FIELDS
    }

    per_rank_band = {
        rank_band: summarize(
            [
                row
                for row in joined_rows
                if row["rank_band"] == rank_band
            ]
        )
        for rank_band in ["high", "medium", "low"]
    }

    evidence_matrix = {
        document_label: {
            evidence_label: mean(
                [
                    row[field]
                    for row in joined_rows
                    if row["document_label_name"]
                    == document_label
                ]
            )
            for evidence_label, field
            in EVIDENCE_FIELDS.items()
        }
        for document_label in EVIDENCE_FIELDS
    }

    quality_means = {
        field: mean(
            [row[field] for row in joined_rows]
        )
        for field in QUALITY_FIELDS
    }

    metrics = {
        "important_caveat": (
            "These are document-label alignment diagnostics. "
            "A document label is not a gold chunk label."
        ),
        "prompt_version": cfg["teacher"]["prompt_version"],
        "insufficient_max_score": (
            INSUFFICIENT_MAX_SCORE
        ),
        "overall": summarize(joined_rows),
        "per_document_class": per_class,
        "per_rank_band": per_rank_band,
        "mean_evidence_by_document_class": evidence_matrix,
        "mean_quality_and_risk": quality_means,
    }

    metrics_path.write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    lowest_margin = sorted(
        joined_rows,
        key=lambda row: (
            row["true_class_margin"],
            row["document_id"],
        ),
    )[:8]

    insufficient = [
        row
        for row in joined_rows
        if row["prediction_status"] == "insufficient"
    ][:8]

    strongest_alignment = sorted(
        [
            row
            for row in joined_rows
            if row["document_label_aligned"]
        ],
        key=lambda row: (
            -row["true_class_margin"],
            row["document_id"],
        ),
    )[:5]

    review_sections = [
        "# Teacher-v2 validation audit",
        "",
        (
            "Document-label alignment is a diagnostic only. "
            "The document category is not a gold label for "
            "every individual chunk."
        ),
        "",
        "## Lowest document-label margins",
        "",
    ]

    review_sections.extend(
        format_case(row)
        for row in lowest_margin
    )

    review_sections.extend(
        [
            "## Insufficient-evidence cases",
            "",
        ]
    )
    review_sections.extend(
        format_case(row)
        for row in insufficient
    )

    review_sections.extend(
        [
            "## Strongest aligned cases",
            "",
        ]
    )
    review_sections.extend(
        format_case(row)
        for row in strongest_alignment
    )

    review_path.write_text(
        "\n".join(review_sections),
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2))
    print("Scored audit table:", scored_path)
    print("Audit metrics:", metrics_path)
    print("Manual review report:", review_path)
    print("Teacher-v2 audit analysis completed.")


if __name__ == "__main__":
    main()
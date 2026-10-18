import argparse
import json
import tomllib
from collections import Counter
from pathlib import Path

import polars as pl


KEY_FIELDS = ("document_id", "query_id", "chunk_id")
STATIC_FIELDS = (
    "audit_id",
    "split",
    "rank_band",
    "candidate_rank",
    "document_label_name",
    "source_path",
    "chunk_text",
)
SCORE_FIELDS = (
    "ai_evidence",
    "ds_evidence",
    "pl_evidence",
    "substantive_relevance",
    "specificity",
    "context_completeness",
    "direct_label_disclosure",
    "keyword_only_risk",
    "brief_reason",
    "max_evidence",
    "predicted_label",
    "prediction_status",
    "true_class_evidence",
    "maximum_other_evidence",
    "true_class_margin",
    "document_label_aligned",
)


def parse_args():
    default_config = Path(__file__).with_name("config.toml")

    parser = argparse.ArgumentParser(
        description="Compare two teacher prompts on the same audit chunks"
    )
    parser.add_argument(
        "--config",
        default=default_config,
        help="Experiment configuration",
    )
    parser.add_argument(
        "--baseline",
        default="teacher_v2",
        help="Baseline prompt version",
    )
    parser.add_argument(
        "--candidate",
        default="teacher_v3",
        help="Candidate prompt version",
    )
    return parser.parse_args()


def load_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def row_key(row: dict) -> tuple[int, int, int]:
    return tuple(int(row[field]) for field in KEY_FIELDS)


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def count(rows: list[dict], field: str, value) -> int:
    return sum(row[field] == value for row in rows)


def summarize_pairs(rows: list[dict]) -> dict:
    return {
        "records": len(rows),
        "corrected": count(rows, "alignment_change", "corrected"),
        "regressed": count(rows, "alignment_change", "regressed"),
        "stayed_aligned": count(
            rows, "alignment_change", "stayed_aligned"
        ),
        "stayed_misaligned": count(
            rows, "alignment_change", "stayed_misaligned"
        ),
        "net_aligned_change": (
            count(rows, "alignment_change", "corrected")
            - count(rows, "alignment_change", "regressed")
        ),
        "resolved_insufficient": count(
            rows, "insufficient_change", "resolved"
        ),
        "introduced_insufficient": count(
            rows, "insufficient_change", "introduced"
        ),
        "mean_true_class_evidence_delta": mean(
            [row["true_class_evidence_delta"] for row in rows]
        ),
        "mean_true_class_margin_delta": mean(
            [row["true_class_margin_delta"] for row in rows]
        ),
        "mean_max_evidence_delta": mean(
            [row["max_evidence_delta"] for row in rows]
        ),
        "more_confident_but_still_misaligned": sum(
            row["alignment_change"] == "stayed_misaligned"
            and row["max_evidence_delta"] > 0
            for row in rows
        ),
        "candidate_ds_prediction_for_ai_or_pl": sum(
            row["candidate_predicted_label"] == "cs.DS"
            and row["document_label_name"] in {"cs.AI", "cs.PL"}
            for row in rows
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
                f"query {row['query_id']}, chunk {row['chunk_id']}"
            ),
            "",
            f"- Document label: `{row['document_label_name']}`",
            (
                f"- Prediction: `{row['baseline_predicted_label']}` "
                f"→ `{row['candidate_predicted_label']}`"
            ),
            (
                f"- Alignment result: `{row['alignment_change']}`; "
                f"insufficient result: `{row['insufficient_change']}`"
            ),
            (
                f"- True-class evidence: "
                f"{row['baseline_true_class_evidence']} → "
                f"{row['candidate_true_class_evidence']} "
                f"(delta {row['true_class_evidence_delta']:+})"
            ),
            (
                f"- True-class margin: "
                f"{row['baseline_true_class_margin']} → "
                f"{row['candidate_true_class_margin']} "
                f"(delta {row['true_class_margin_delta']:+})"
            ),
            (
                f"- Evidence v2: AI={row['baseline_ai_evidence']}, "
                f"DS={row['baseline_ds_evidence']}, "
                f"PL={row['baseline_pl_evidence']}"
            ),
            (
                f"- Evidence v3: AI={row['candidate_ai_evidence']}, "
                f"DS={row['candidate_ds_evidence']}, "
                f"PL={row['candidate_pl_evidence']}"
            ),
            f"- Baseline reason: {row['baseline_brief_reason']}",
            f"- Candidate reason: {row['candidate_brief_reason']}",
            "",
            "**Chunk excerpt**",
            "",
            text,
            "",
        ]
    )


def add_cases(sections: list[str], title: str, rows: list[dict]):
    sections.extend([f"## {title}", ""])
    if rows:
        sections.extend(format_case(row) for row in rows)
    else:
        sections.extend(["None.", ""])


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

    baseline_version = str(args.baseline)
    candidate_version = str(args.candidate)
    if baseline_version == candidate_version:
        raise ValueError("Baseline and candidate versions must differ")

    baseline_path = (
        output_dir / f"audit_scored.{baseline_version}.parquet"
    )
    candidate_path = (
        output_dir / f"audit_scored.{candidate_version}.parquet"
    )

    baseline_rows = pl.read_parquet(baseline_path).to_dicts()
    candidate_rows = pl.read_parquet(candidate_path).to_dicts()

    required = set(KEY_FIELDS + STATIC_FIELDS + SCORE_FIELDS)
    for version, rows in [
        (baseline_version, baseline_rows),
        (candidate_version, candidate_rows),
    ]:
        if not rows:
            raise ValueError(f"No scored rows for {version}")
        missing = required - set(rows[0])
        if missing:
            raise ValueError(
                f"Missing columns for {version}: {sorted(missing)}"
            )

    baseline_by_key = {row_key(row): row for row in baseline_rows}
    candidate_by_key = {row_key(row): row for row in candidate_rows}
    if len(baseline_by_key) != len(baseline_rows):
        raise ValueError("Duplicate keys in baseline audit")
    if len(candidate_by_key) != len(candidate_rows):
        raise ValueError("Duplicate keys in candidate audit")
    if set(baseline_by_key) != set(candidate_by_key):
        raise ValueError("Prompt versions do not contain the same audit keys")

    paired_rows = []
    for key in sorted(baseline_by_key):
        baseline = baseline_by_key[key]
        candidate = candidate_by_key[key]

        for field in STATIC_FIELDS:
            if baseline[field] != candidate[field]:
                raise ValueError(
                    f"Static field {field} differs for key {key}"
                )

        baseline_aligned = bool(baseline["document_label_aligned"])
        candidate_aligned = bool(candidate["document_label_aligned"])
        if not baseline_aligned and candidate_aligned:
            alignment_change = "corrected"
        elif baseline_aligned and not candidate_aligned:
            alignment_change = "regressed"
        elif baseline_aligned and candidate_aligned:
            alignment_change = "stayed_aligned"
        else:
            alignment_change = "stayed_misaligned"

        baseline_insufficient = (
            baseline["prediction_status"] == "insufficient"
        )
        candidate_insufficient = (
            candidate["prediction_status"] == "insufficient"
        )
        if baseline_insufficient and not candidate_insufficient:
            insufficient_change = "resolved"
        elif not baseline_insufficient and candidate_insufficient:
            insufficient_change = "introduced"
        elif baseline_insufficient and candidate_insufficient:
            insufficient_change = "stayed_insufficient"
        else:
            insufficient_change = "neither"

        paired = {
            **{field: baseline[field] for field in KEY_FIELDS},
            **{field: baseline[field] for field in STATIC_FIELDS},
            "alignment_change": alignment_change,
            "insufficient_change": insufficient_change,
        }
        for field in SCORE_FIELDS:
            paired[f"baseline_{field}"] = baseline[field]
            paired[f"candidate_{field}"] = candidate[field]

        for field in (
            "true_class_evidence",
            "true_class_margin",
            "max_evidence",
            "substantive_relevance",
            "specificity",
            "context_completeness",
            "keyword_only_risk",
        ):
            paired[f"{field}_delta"] = (
                candidate[field] - baseline[field]
            )

        paired_rows.append(paired)

    comparison_name = f"{baseline_version}_vs_{candidate_version}"
    paired_path = output_dir / f"prompt_comparison.{comparison_name}.parquet"
    metrics_path = output_dir / f"prompt_comparison.{comparison_name}.json"
    review_path = output_dir / f"prompt_comparison.{comparison_name}.md"

    pl.DataFrame(paired_rows).write_parquet(paired_path)

    transitions = Counter(
        (
            f"{row['baseline_predicted_label']} -> "
            f"{row['candidate_predicted_label']}"
        )
        for row in paired_rows
    )
    per_class = {
        label_name: summarize_pairs(
            [
                row
                for row in paired_rows
                if row["document_label_name"] == label_name
            ]
        )
        for label_name in ("cs.AI", "cs.DS", "cs.PL")
    }

    metrics = {
        "important_caveat": (
            "Document-label alignment is diagnostic only; a document "
            "label is not a gold chunk label. Prompt v3 was developed "
            "using this validation audit, so this is not a held-out "
            "estimate of generalization."
        ),
        "baseline_prompt_version": baseline_version,
        "candidate_prompt_version": candidate_version,
        "overall": summarize_pairs(paired_rows),
        "per_document_class": per_class,
        "prediction_transitions": dict(sorted(transitions.items())),
    }
    metrics_path.write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    corrected = [
        row
        for row in paired_rows
        if row["alignment_change"] == "corrected"
    ]
    regressed = [
        row
        for row in paired_rows
        if row["alignment_change"] == "regressed"
    ]
    unresolved_ds = [
        row
        for row in paired_rows
        if row["candidate_predicted_label"] == "cs.DS"
        and row["document_label_name"] in {"cs.AI", "cs.PL"}
    ]
    confidence_without_correction = sorted(
        [
            row
            for row in paired_rows
            if row["alignment_change"] == "stayed_misaligned"
            and row["max_evidence_delta"] > 0
        ],
        key=lambda row: (
            -row["max_evidence_delta"],
            row["document_id"],
        ),
    )
    largest_margin_declines = sorted(
        paired_rows,
        key=lambda row: (
            row["true_class_margin_delta"],
            row["document_id"],
        ),
    )[:8]

    overall = metrics["overall"]
    review_sections = [
        f"# Prompt comparison: {baseline_version} vs {candidate_version}",
        "",
        (
            "Document-label alignment is a diagnostic only. The same "
            "validation chunks were used for both prompts, and the "
            "candidate prompt was developed using this audit."
        ),
        "",
        "## Paired summary",
        "",
        f"- Records: {overall['records']}",
        f"- Corrected: {overall['corrected']}",
        f"- Regressed: {overall['regressed']}",
        f"- Net aligned change: {overall['net_aligned_change']}",
        f"- Resolved insufficient: {overall['resolved_insufficient']}",
        f"- Introduced insufficient: {overall['introduced_insufficient']}",
        (
            "- More confident but still misaligned: "
            f"{overall['more_confident_but_still_misaligned']}"
        ),
        (
            "- Candidate DS predictions for AI/PL documents: "
            f"{overall['candidate_ds_prediction_for_ai_or_pl']}"
        ),
        "",
    ]
    add_cases(review_sections, "Corrected cases", corrected)
    add_cases(review_sections, "Regressed cases", regressed)
    add_cases(
        review_sections,
        "Remaining DS predictions for AI or PL documents",
        unresolved_ds,
    )
    add_cases(
        review_sections,
        "More confident but still misaligned",
        confidence_without_correction,
    )
    add_cases(
        review_sections,
        "Largest true-class margin declines",
        largest_margin_declines,
    )

    review_path.write_text(
        "\n".join(review_sections),
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2))
    print("Paired table:", paired_path)
    print("Comparison metrics:", metrics_path)
    print("Comparison review:", review_path)
    print("Prompt comparison completed.")


if __name__ == "__main__":
    main()

import argparse
import json
import statistics
import tomllib
from pathlib import Path

import polars as pl


SCORE_FIELDS = (
    "target_evidence",
    "substantive_relevance",
    "specificity",
    "context_completeness",
    "direct_label_disclosure",
    "keyword_only_risk",
)

LABEL_CUE_PHRASES = (
    "category cue",
    "category connection",
    "category terminology",
    "category-specific terminology",
    "explicit category",
    "explicit ai",
    "explicit ds",
    "explicit pl",
    "target category",
)


def parse_args():
    default_config = Path(__file__).with_name("config.target.toml")
    parser = argparse.ArgumentParser(
        description="Analyze target-conditioned teacher audit scores"
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


def key(record: dict) -> tuple[int, int, int]:
    return (
        int(record["document_id"]),
        int(record["query_id"]),
        int(record["chunk_id"]),
    )


def mean(rows: list[dict], field: str) -> float | None:
    if not rows:
        return None
    return sum(float(row[field]) for row in rows) / len(rows)


def evidence_distribution(rows: list[dict]) -> dict[str, int]:
    return {
        str(score): sum(
            int(row["target_evidence"]) == score for row in rows
        )
        for score in range(5)
    }


def summarize(rows: list[dict]) -> dict:
    evidence = [int(row["target_evidence"]) for row in rows]
    return {
        "records": len(rows),
        "target_evidence_distribution": evidence_distribution(rows),
        "mean_target_evidence": mean(rows, "target_evidence"),
        "median_target_evidence": (
            statistics.median(evidence) if evidence else None
        ),
        "strong_or_clear_evidence": sum(value >= 3 for value in evidence),
        "strong_or_clear_evidence_rate": (
            sum(value >= 3 for value in evidence) / len(rows)
            if rows
            else None
        ),
        "weak_or_absent_evidence": sum(value <= 1 for value in evidence),
        "weak_or_absent_evidence_rate": (
            sum(value <= 1 for value in evidence) / len(rows)
            if rows
            else None
        ),
        "mean_substantive_relevance": mean(
            rows, "substantive_relevance"
        ),
        "mean_specificity": mean(rows, "specificity"),
        "mean_context_completeness": mean(
            rows, "context_completeness"
        ),
        "mean_direct_label_disclosure": mean(
            rows, "direct_label_disclosure"
        ),
        "mean_keyword_only_risk": mean(rows, "keyword_only_risk"),
        "label_cue_reason_flags": sum(
            bool(row["label_cue_reason_flag"]) for row in rows
        ),
        "high_evidence_shortcut_flags": sum(
            bool(row["high_evidence_shortcut_flag"]) for row in rows
        ),
        "substantive_but_weak_target_flags": sum(
            bool(row["substantive_but_weak_target_flag"])
            for row in rows
        ),
    }


def format_case(row: dict) -> str:
    excerpt = " ".join(str(row["chunk_text"]).split())
    if len(excerpt) > 1000:
        excerpt = excerpt[:1000] + "..."

    flags = []
    if row["label_cue_reason_flag"]:
        flags.append("label-cue wording")
    if row["high_evidence_shortcut_flag"]:
        flags.append("high-evidence shortcut risk")
    if row["substantive_but_weak_target_flag"]:
        flags.append("substantive but weak target evidence")

    flag_text = ", ".join(flags) if flags else "none"

    return "\n".join(
        [
            (
                f"### Document {row['document_id']}, "
                f"query {row['query_id']}, chunk {row['chunk_id']}"
            ),
            "",
            f"- Target: `{row['target_label']}`",
            (
                f"- Original IRIS rank band: `{row['rank_band']}`; "
                f"candidate rank: {row['candidate_rank']}"
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
            f"- Diagnostic flags: {flag_text}",
            f"- Teacher reason: {row['brief_reason']}",
            "",
            "**Chunk excerpt**",
            "",
            excerpt,
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
    output_dir = Path(cfg["paths"]["output_dir"])
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir

    manifest_path = output_dir / "audit_manifest.parquet"
    input_path = output_dir / "target_teacher_inputs.audit.jsonl"
    score_path = (
        output_dir / f"qwen_scores.{prompt_version}.audit.jsonl"
    )
    scored_path = (
        output_dir / f"target_audit_scored.{prompt_version}.parquet"
    )
    metrics_path = (
        output_dir / f"target_audit_metrics.{prompt_version}.json"
    )
    review_path = (
        output_dir / f"target_audit_review.{prompt_version}.md"
    )

    manifest_rows = pl.read_parquet(manifest_path).to_dicts()
    input_rows = read_jsonl(input_path)
    score_rows = read_jsonl(score_path)

    manifest_by_key = {key(row): row for row in manifest_rows}
    input_by_key = {key(row): row for row in input_rows}
    score_by_key = {key(row): row for row in score_rows}
    expected_keys = set(manifest_by_key)

    if len(manifest_by_key) != len(manifest_rows):
        raise ValueError("Duplicate keys in target audit manifest")
    if len(input_by_key) != len(input_rows):
        raise ValueError("Duplicate keys in target teacher inputs")
    if len(score_by_key) != len(score_rows):
        raise ValueError("Duplicate keys in target teacher scores")
    if set(input_by_key) != expected_keys:
        raise ValueError("Target teacher input keys do not match manifest")
    if set(score_by_key) != expected_keys:
        raise ValueError("Target teacher score keys do not match manifest")

    joined_rows = []
    for record_key in sorted(expected_keys):
        manifest = manifest_by_key[record_key]
        teacher_input = input_by_key[record_key]
        score = score_by_key[record_key]

        target_label = str(teacher_input["target_label"])
        if str(score["target_label"]) != target_label:
            raise ValueError(
                f"Target-label mismatch for {record_key}: "
                f"{score['target_label']} != {target_label}"
            )
        if str(manifest["label_name"]) != target_label:
            raise ValueError(
                f"Manifest-label mismatch for {record_key}"
            )

        values = {field: int(score[field]) for field in SCORE_FIELDS}
        reason = str(score["brief_reason"])
        reason_lower = reason.lower()
        label_cue_reason_flag = any(
            phrase in reason_lower for phrase in LABEL_CUE_PHRASES
        )
        high_evidence_shortcut_flag = (
            values["target_evidence"] >= 3
            and (
                values["direct_label_disclosure"] >= 3
                or values["keyword_only_risk"] >= 3
            )
        )
        substantive_but_weak_target_flag = (
            values["target_evidence"] <= 1
            and values["substantive_relevance"] >= 3
        )

        joined_rows.append(
            {
                "audit_id": int(manifest["audit_id"]),
                "split": str(manifest["split"]),
                "rank_band": str(manifest["rank_band"]),
                "document_id": int(manifest["document_id"]),
                "query_id": int(manifest["query_id"]),
                "chunk_id": int(manifest["chunk_id"]),
                "candidate_rank": int(manifest["candidate_rank"]),
                "vector_distance": float(manifest["vector_distance"]),
                "target_label": target_label,
                "target_description": str(
                    teacher_input["target_description"]
                ),
                "chunk_text": str(teacher_input["chunk_text"]),
                **values,
                "brief_reason": reason,
                "label_cue_reason_flag": label_cue_reason_flag,
                "high_evidence_shortcut_flag": (
                    high_evidence_shortcut_flag
                ),
                "substantive_but_weak_target_flag": (
                    substantive_but_weak_target_flag
                ),
            }
        )

    overall = summarize(joined_rows)
    per_target = {
        label: summarize(
            [row for row in joined_rows if row["target_label"] == label]
        )
        for label in ("cs.AI", "cs.DS", "cs.PL")
    }
    per_rank_band = {
        band: summarize(
            [row for row in joined_rows if row["rank_band"] == band]
        )
        for band in ("high", "medium", "low")
    }

    metrics = {
        "important_caveat": (
            "This is a prompt-development audit on validation chunks, not "
            "a held-out estimate and not expert annotation. Target evidence "
            "scores are teacher judgments, not gold labels."
        ),
        "prompt_version": prompt_version,
        "overall": overall,
        "per_target_class": per_target,
        "per_original_rank_band": per_rank_band,
    }

    scored = pl.DataFrame(joined_rows).sort("audit_id")
    scored.write_parquet(scored_path)
    metrics_path.write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    flagged = [
        row
        for row in joined_rows
        if row["label_cue_reason_flag"]
        or row["high_evidence_shortcut_flag"]
        or row["substantive_but_weak_target_flag"]
    ]
    high = sorted(
        (row for row in joined_rows if row["target_evidence"] >= 3),
        key=lambda row: (-row["target_evidence"], row["audit_id"]),
    )
    low = sorted(
        (row for row in joined_rows if row["target_evidence"] <= 1),
        key=lambda row: (row["target_evidence"], row["audit_id"]),
    )

    report = [
        f"# Target-conditioned audit: {prompt_version}",
        "",
        metrics["important_caveat"],
        "",
        "## Summary",
        "",
        f"- Records: {overall['records']}",
        (
            f"- Mean target evidence: "
            f"{overall['mean_target_evidence']:.3f}"
        ),
        (
            f"- Clear/strong evidence (3-4): "
            f"{overall['strong_or_clear_evidence']}"
        ),
        (
            f"- Weak/absent evidence (0-1): "
            f"{overall['weak_or_absent_evidence']}"
        ),
        (
            f"- Label-cue wording flags: "
            f"{overall['label_cue_reason_flags']}"
        ),
        (
            f"- High-evidence shortcut flags: "
            f"{overall['high_evidence_shortcut_flags']}"
        ),
        "",
        "## Diagnostic flags",
        "",
    ]
    if flagged:
        report.extend(format_case(row) for row in flagged)
    else:
        report.extend(["No diagnostic flags.", ""])

    report.extend(["## Clear or strong target evidence", ""])
    if high:
        report.extend(format_case(row) for row in high)
    else:
        report.extend(["No chunks scored 3 or 4.", ""])

    report.extend(["## Weak or absent target evidence", ""])
    if low:
        report.extend(format_case(row) for row in low)
    else:
        report.extend(["No chunks scored 0 or 1.", ""])

    review_path.write_text("\n".join(report), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"Scored audit: {scored_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Review: {review_path}")


if __name__ == "__main__":
    main()

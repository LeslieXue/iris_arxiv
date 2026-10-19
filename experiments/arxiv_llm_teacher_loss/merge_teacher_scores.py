"""Validate all Qwen shards and build the IRIS teacher-score table."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import tomllib

SCORE_COLUMNS = [
    "document_id",
    "chunk_id",
    "target_label",
    "target_evidence",
    "substantive_relevance",
    "specificity",
    "context_completeness",
    "direct_label_disclosure",
    "keyword_only_risk",
]
SCORE_FIELDS = SCORE_COLUMNS[3:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=Path(__file__).with_name("config.toml"),
        type=Path,
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def resolve_path(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def shard_name(prefix: str, shard_id: int, num_shards: int, suffix: str) -> str:
    width = max(3, len(str(num_shards - 1)))
    return (
        f"{prefix}.shard-{shard_id:0{width}d}-of-"
        f"{num_shards:0{width}d}{suffix}"
    )


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    config_path = resolve_path(repo_root, args.config)
    cfg = load_toml(config_path)
    output_dir = args.output_dir or resolve_path(
        repo_root, cfg["paths"]["output_dir"]
    )
    num_shards = int(cfg["export"]["num_shards"])
    prompt_version = str(cfg["teacher"]["prompt_version"])
    tie_weight = float(cfg["score"]["tie_breaker_weight"])
    if not 0 <= tie_weight < 0.125:
        raise ValueError(
            "tie_breaker_weight must be in [0, 0.125) so secondary fields "
            "cannot reverse a one-point target-evidence difference"
        )

    manifest_path = output_dir / "all_chunks_manifest.parquet"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = pl.read_parquet(manifest_path)
    expected_keys = manifest.select(
        pl.col("id").alias("document_id"), "chunk_id", "target_label"
    )

    shard_frames = []
    shard_paths = []
    for shard_id in range(num_shards):
        path = output_dir / shard_name(
            f"qwen_scores.{prompt_version}",
            shard_id,
            num_shards,
            ".jsonl",
        )
        if not path.exists():
            raise FileNotFoundError(f"Missing score shard: {path}")
        shard_paths.append(path)
        shard_frames.append(pl.scan_ndjson(path).select(SCORE_COLUMNS).collect())

    scores = pl.concat(shard_frames, how="vertical")
    unique_keys = scores.select(
        pl.struct("document_id", "chunk_id").n_unique()
    ).item()
    if unique_keys != scores.height:
        raise ValueError(
            "Teacher outputs contain duplicate (document_id, chunk_id) keys"
        )

    for field in SCORE_FIELDS:
        invalid = scores.filter(~pl.col(field).is_between(0, 4, closed="both"))
        if invalid.height:
            raise ValueError(f"{field} contains values outside [0, 4]")

    comparison = expected_keys.join(
        scores.select("document_id", "chunk_id", "target_label"),
        on=["document_id", "chunk_id"],
        how="full",
        suffix="_score",
    )
    missing = comparison.filter(pl.col("target_label_score").is_null())
    unexpected = comparison.filter(pl.col("target_label").is_null())
    mismatched = comparison.filter(
        pl.col("target_label").is_not_null()
        & pl.col("target_label_score").is_not_null()
        & (pl.col("target_label") != pl.col("target_label_score"))
    )
    if missing.height or unexpected.height or mismatched.height:
        raise ValueError(
            "Teacher-score coverage failed: "
            f"missing={missing.height}, unexpected={unexpected.height}, "
            f"label_mismatch={mismatched.height}"
        )

    quality = pl.mean_horizontal(
        "substantive_relevance", "specificity", "context_completeness"
    )
    shortcut_risk = pl.mean_horizontal(
        "direct_label_disclosure", "keyword_only_risk"
    )
    scores = scores.with_columns(
        (
            pl.col("target_evidence").cast(pl.Float32)
            + tie_weight * (quality - shortcut_risk)
        ).alias("teacher_utility")
    ).sort("document_id", "chunk_id")

    final_columns = [
        "document_id",
        "chunk_id",
        "target_label",
        "teacher_utility",
        *SCORE_FIELDS,
    ]
    output_path = output_dir / "teacher_scores.parquet"
    scores.select(final_columns).write_parquet(output_path)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": cfg["version"],
        "prompt_version": prompt_version,
        "records": scores.height,
        "documents": scores["document_id"].n_unique(),
        "num_shards": num_shards,
        "tie_breaker_weight": tie_weight,
        "teacher_utility_formula": (
            "target_evidence + tie_breaker_weight * "
            "(mean(substantive_relevance,specificity,context_completeness) - "
            "mean(direct_label_disclosure,keyword_only_risk))"
        ),
        "target_evidence_distribution": {
            str(row["target_evidence"]): int(row["len"])
            for row in (
                scores.group_by("target_evidence")
                .len()
                .sort("target_evidence")
                .iter_rows(named=True)
            )
        },
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "score_shards": [str(path) for path in shard_paths],
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
    }
    metadata_path = output_dir / "teacher_scores.metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(metadata, indent=2))
    print("Full teacher-score merge completed successfully.")


if __name__ == "__main__":
    main()

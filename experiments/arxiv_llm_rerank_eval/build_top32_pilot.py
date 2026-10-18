import argparse
import hashlib
import json
import tomllib
from pathlib import Path

import numpy as np
import polars as pl

from iris_llm import TargetCandidateForScoring


def parse_args():
    default_config = Path(__file__).with_name("config.target.toml")
    parser = argparse.ArgumentParser(
        description="Build one-document IRIS top-32 reranking pilot"
    )
    parser.add_argument(
        "--config",
        default=default_config,
        help="Target-conditioned experiment configuration",
    )
    parser.add_argument(
        "--document-id",
        type=int,
        default=None,
        help="Use a specific eligible document instead of seeded selection",
    )
    return parser.parse_args()


def load_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def eligible_document_ids(
    candidates: pl.DataFrame,
    num_queries: int,
    candidates_per_query: int,
) -> list[int]:
    expected_queries = set(range(num_queries))
    eligible = []

    for document_id in sorted(
        int(value) for value in candidates["document_id"].unique()
    ):
        document = candidates.filter(
            pl.col("document_id") == document_id
        )
        observed_queries = set(document["query_id"].to_list())
        if observed_queries != expected_queries:
            continue

        counts = document.group_by("query_id").len()
        if counts["len"].min() < candidates_per_query:
            continue

        eligible.append(document_id)

    return eligible


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    config_path = resolve_repo_path(repo_root, str(args.config))
    cfg = load_toml(config_path)
    pilot_cfg = cfg["pilot"]

    source_dir = resolve_repo_path(
        repo_root,
        cfg["source"]["output_dir"],
    )
    target_root = resolve_repo_path(
        repo_root,
        cfg["paths"]["output_dir"],
    )
    output_dir = target_root / str(pilot_cfg["output_subdir"])

    candidates_path = source_dir / "candidates.parquet"
    development_audit_path = source_dir / "audit_manifest.parquet"
    manifest_path = output_dir / "pilot_manifest.parquet"
    input_path = output_dir / "target_teacher_inputs.pilot.jsonl"
    metadata_path = output_dir / "pilot_metadata.json"

    candidates = pl.read_parquet(candidates_path)
    development_audit = pl.read_parquet(development_audit_path)

    required = {
        "split",
        "document_id",
        "query_id",
        "chunk_id",
        "candidate_rank",
        "vector_distance",
        "chunk_text",
        "label",
        "label_name",
        "source_path",
    }
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(
            f"Candidate file is missing columns: {sorted(missing)}"
        )

    split = str(pilot_cfg["split"])
    target_label = str(pilot_cfg["target_label"])
    num_queries = int(pilot_cfg["num_queries"])
    candidates_per_query = int(pilot_cfg["candidates_per_query"])
    seed = int(pilot_cfg["selection_seed"])

    if candidates_per_query < 2:
        raise ValueError("pilot.candidates_per_query must be at least 2")
    if num_queries < 1:
        raise ValueError("pilot.num_queries must be at least 1")
    if target_label not in cfg["target_descriptions"]:
        raise ValueError(f"Unknown pilot target label: {target_label}")

    excluded_documents = set(
        int(value) for value in development_audit["document_id"].to_list()
    )
    pool = candidates.filter(
        (pl.col("split") == split)
        & (pl.col("label_name") == target_label)
        & (~pl.col("document_id").is_in(sorted(excluded_documents)))
    )
    if pool.height == 0:
        raise ValueError(
            "No candidates remain after split, target, and audit exclusions"
        )

    eligible = eligible_document_ids(
        pool,
        num_queries=num_queries,
        candidates_per_query=candidates_per_query,
    )
    if not eligible:
        raise ValueError(
            "No document has enough candidates for every configured query"
        )

    if args.document_id is None:
        rng = np.random.default_rng(seed)
        document_id = int(rng.choice(np.asarray(eligible)))
    else:
        document_id = int(args.document_id)
        if document_id not in eligible:
            raise ValueError(
                f"Document {document_id} is not eligible; "
                f"eligible documents: {eligible}"
            )

    document = pool.filter(
        pl.col("document_id") == document_id
    )
    selected_parts = []
    for query_id in range(num_queries):
        query_rows = (
            document.filter(pl.col("query_id") == query_id)
            .sort("candidate_rank", "vector_distance", "chunk_id")
            .head(candidates_per_query)
        )
        if query_rows.height != candidates_per_query:
            raise RuntimeError(
                f"Query {query_id} yielded {query_rows.height} rows"
            )
        selected_parts.append(query_rows)

    selected = (
        pl.concat(selected_parts)
        .sort("query_id", "candidate_rank")
        .with_columns(
            (pl.col("candidate_rank") == 1).alias(
                "original_iris_selection"
            )
        )
    )

    expected_rows = num_queries * candidates_per_query
    if selected.height != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} candidates, got {selected.height}"
        )
    if selected.select(
        pl.struct("document_id", "query_id", "chunk_id").n_unique()
    ).item() != expected_rows:
        raise RuntimeError("Pilot candidate keys are not unique")
    if selected["chunk_id"].n_unique() != expected_rows:
        raise RuntimeError("Pilot contains duplicate chunks across queries")
    if selected.filter(pl.col("original_iris_selection")).height != num_queries:
        raise RuntimeError("Expected one original selection per query")

    target_description = str(
        cfg["target_descriptions"][target_label]
    )
    teacher_records = [
        TargetCandidateForScoring(
            document_id=int(row["document_id"]),
            query_id=int(row["query_id"]),
            chunk_id=int(row["chunk_id"]),
            target_label=target_label,
            target_description=target_description,
            chunk_text=str(row["chunk_text"]),
        )
        for row in selected.iter_rows(named=True)
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    selected.write_parquet(manifest_path)
    with input_path.open("w", encoding="utf-8") as handle:
        for record in teacher_records:
            handle.write(record.model_dump_json() + "\n")

    labels = set(selected["label_name"].to_list())
    source_paths = set(selected["source_path"].to_list())
    metadata = {
        "purpose": "descriptive_top32_reranking_pilot",
        "selection_seed": seed,
        "source_candidates": str(candidates_path),
        "source_candidates_sha256": sha256_file(candidates_path),
        "excluded_prompt_development_documents": len(
            excluded_documents
        ),
        "eligible_documents": len(eligible),
        "split": split,
        "document_id": document_id,
        "target_label": target_label,
        "source_path": next(iter(source_paths)),
        "num_queries": num_queries,
        "candidates_per_query": candidates_per_query,
        "candidate_records": expected_rows,
        "prompt_version": cfg["teacher"]["prompt_version"],
        "manifest": str(manifest_path),
        "teacher_inputs": str(input_path),
    }
    if labels != {target_label}:
        raise RuntimeError(f"Unexpected labels in pilot: {labels}")
    if len(source_paths) != 1:
        raise RuntimeError(
            f"Expected one source path, got {sorted(source_paths)}"
        )
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(metadata, indent=2))
    print(f"Pilot manifest: {manifest_path}")
    print(f"Teacher inputs: {input_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()

import argparse
import json
import tomllib
from pathlib import Path

import duckdb
import lancedb
import numpy as np
import polars as pl
import torch

from iris.config import get_config
from iris.dataset import LanceTableDataset, train_val_test_split
from iris_llm import CandidateForScoring, CandidateRecord


LABEL_NAMES = {
    0: "cs.AI",
    1: "cs.DS",
    2: "cs.PL",
}


def parse_args():
    default_config = Path(__file__).with_name("config.toml")

    parser = argparse.ArgumentParser(
        description="Export fixed IRIS candidate chunks for LLM reranking"
    )
    parser.add_argument(
        "--config",
        default=default_config,
        help="Path to the arXiv LLM reranking experiment configuration",
    )
    return parser.parse_args()


def load_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def resolve_repo_path(repo_root: Path, path: str) -> Path:
    resolved = Path(path)

    if resolved.is_absolute():
        return resolved

    return repo_root / resolved


def get_document_labels(table) -> tuple[np.ndarray, np.ndarray]:
    lance_table = table.to_lance()
    duckdb.register("candidate_chunks", lance_table)

    rows = duckdb.sql(
        """
        SELECT DISTINCT id, label
        FROM candidate_chunks
        ORDER BY id
        """
    ).fetchall()

    ids = np.asarray([int(row[0]) for row in rows], dtype=np.int64)
    labels = np.asarray([int(row[1]) for row in rows], dtype=np.int64)

    return ids, labels


def sample_split(
    split_name: str,
    split_ids: np.ndarray,
    label_by_id: dict[int, int],
    per_class: int,
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []

    for label in sorted(LABEL_NAMES):
        available = np.asarray(
            sorted(
                int(document_id)
                for document_id in split_ids
                if label_by_id[int(document_id)] == label
            ),
            dtype=np.int64,
        )

        if available.size < per_class:
            raise ValueError(
                f"{split_name} has only {available.size} documents for "
                f"{LABEL_NAMES[label]}, but {per_class} were requested."
            )

        selected = np.sort(
            rng.choice(
                available,
                size=per_class,
                replace=False,
            )
        )

        rows.extend(
            {
                "split": split_name,
                "document_id": int(document_id),
                "label": label,
                "label_name": LABEL_NAMES[label],
            }
            for document_id in selected
        )

    return rows


def load_trained_queries(checkpoint_path: Path) -> np.ndarray:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if "queries" not in checkpoint:
        raise KeyError(
            f"Checkpoint does not contain trained queries: {checkpoint_path}"
        )

    queries = checkpoint["queries"]

    if isinstance(queries, torch.Tensor):
        queries = queries.detach().cpu().numpy()

    queries = np.asarray(queries, dtype=np.float32)

    if queries.ndim != 2:
        raise ValueError(
            f"Expected a two-dimensional query matrix, got {queries.shape}"
        )

    return queries


def export_candidates(
    table,
    manifest_rows: list[dict],
    trained_queries: np.ndarray,
    candidates_per_query: int,
    ignore_index: int,
) -> tuple[pl.DataFrame, pl.DataFrame, list[CandidateForScoring]]:
    document_ids = np.asarray(
        [row["document_id"] for row in manifest_rows],
        dtype=np.int64,
    )

    dataset = LanceTableDataset(
        ids=document_ids,
        tbl=table,
        vector_dim=trained_queries.shape[1],
        k_neighbors=candidates_per_query,
        ignore_index=ignore_index,
        ini_queries=trained_queries,
    )

    candidate_rows = []
    scoring_inputs = []

    for index, manifest_row in enumerate(manifest_rows, start=1):
        document_id = manifest_row["document_id"]

        chunks = dataset.search_chunks(
            document_id=document_id,
            include_text=True,
        )

        if chunks.height == 0:
            raise RuntimeError(
                f"No chunks were retrieved for document {document_id}"
            )

        observed_labels = set(chunks["label"].to_list())

        if observed_labels != {manifest_row["label"]}:
            raise ValueError(
                f"Document {document_id} has unexpected labels: "
                f"{sorted(observed_labels)}"
            )

        manifest_row["candidate_count"] = chunks.height
        manifest_row["source_path"] = str(chunks["source_path"][0])

        for chunk in chunks.iter_rows(named=True):
            record = CandidateRecord(
                document_id=int(chunk["id"]),
                query_id=int(chunk["query_id"]),
                chunk_id=int(chunk["chunk_id"]),
                candidate_rank=int(chunk["rank"]),
                vector_distance=float(chunk["_distance"]),
                chunk_text=str(chunk["chunk_text"]),
                label=int(chunk["label"]),
                label_name=manifest_row["label_name"],
                source_path=str(chunk["source_path"]),
            )

            candidate_row = {
                "split": manifest_row["split"],
                **record.model_dump(),
            }
            candidate_rows.append(candidate_row)

            scoring_inputs.append(
                CandidateForScoring(
                    document_id=record.document_id,
                    query_id=record.query_id,
                    chunk_id=record.chunk_id,
                    chunk_text=record.chunk_text,
                )
            )

        if index % 20 == 0 or index == len(manifest_rows):
            print(
                f"Exported candidates for {index}/"
                f"{len(manifest_rows)} documents"
            )

    manifest = pl.DataFrame(manifest_rows).sort(
        "split",
        "label",
        "document_id",
    )
    candidates = pl.DataFrame(candidate_rows).sort(
        "split",
        "document_id",
        "query_id",
        "candidate_rank",
    )

    return manifest, candidates, scoring_inputs


def validate_candidates(
    manifest: pl.DataFrame,
    candidates: pl.DataFrame,
    scoring_inputs: list[CandidateForScoring],
    expected_num_queries: int,
    candidates_per_query: int,
) -> None:
    duplicate_count = (
        candidates.group_by("document_id", "chunk_id")
        .len()
        .filter(pl.col("len") > 1)
        .height
    )

    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate document/chunk pairs"
        )

    maximum_candidates = expected_num_queries * candidates_per_query

    if manifest["candidate_count"].max() > maximum_candidates:
        raise ValueError(
            "A document contains more candidates than the configured maximum"
        )

    safe_columns = set(scoring_inputs[0].model_dump())

    forbidden_columns = {
        "label",
        "label_name",
        "source_path",
        "vector_distance",
        "candidate_rank",
        "split",
    }

    leaked_columns = safe_columns & forbidden_columns

    if leaked_columns:
        leaked = ", ".join(sorted(leaked_columns))
        raise ValueError(f"Teacher inputs leak protected fields: {leaked}")

    if len(scoring_inputs) != candidates.height:
        raise ValueError(
            "Teacher input count does not match candidate row count"
        )


def write_jsonl(path: Path, records: list[CandidateForScoring]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json())
            handle.write("\n")


def main():
    args = parse_args()
    experiment_config_path = Path(args.config).resolve()
    experiment_config = load_toml(experiment_config_path)

    repo_root = Path(__file__).resolve().parents[2]

    iris_config_path = resolve_repo_path(
        repo_root,
        experiment_config["iris"]["config_path"],
    )
    checkpoint_path = resolve_repo_path(
        repo_root,
        experiment_config["iris"]["checkpoint"],
    )
    output_dir = resolve_repo_path(
        repo_root,
        experiment_config["paths"]["output_dir"],
    )

    iris_config = get_config(iris_config_path)

    database = lancedb.connect(iris_config.database.path)
    table = database.open_table(
        experiment_config["iris"]["table_name"]
    )

    ids, labels = get_document_labels(table)
    label_by_id = {
        int(document_id): int(label)
        for document_id, label in zip(ids, labels, strict=True)
    }

    _, validation_ids, test_ids = train_val_test_split(
        ids,
        labels=labels,
        seed=iris_config.seed,
    )

    experiment_seed = int(experiment_config["seed"])

    manifest_rows = sample_split(
        split_name="validation",
        split_ids=validation_ids,
        label_by_id=label_by_id,
        per_class=int(
            experiment_config["sampling"]["validation_per_class"]
        ),
        seed=experiment_seed,
    )
    manifest_rows.extend(
        sample_split(
            split_name="test",
            split_ids=test_ids,
            label_by_id=label_by_id,
            per_class=int(
                experiment_config["sampling"]["test_per_class"]
            ),
            seed=experiment_seed + 1,
        )
    )

    trained_queries = load_trained_queries(checkpoint_path)

    configured_queries = int(
        experiment_config["retrieval"]["num_queries"]
    )

    if trained_queries.shape[0] != configured_queries:
        raise ValueError(
            f"Checkpoint contains {trained_queries.shape[0]} queries, "
            f"but the experiment expects {configured_queries}"
        )

    manifest, candidates, scoring_inputs = export_candidates(
        table=table,
        manifest_rows=manifest_rows,
        trained_queries=trained_queries,
        candidates_per_query=int(
            experiment_config["retrieval"]["candidates_per_query"]
        ),
        ignore_index=iris_config.trainer.ignore_index,
    )

    validate_candidates(
        manifest=manifest,
        candidates=candidates,
        scoring_inputs=scoring_inputs,
        expected_num_queries=configured_queries,
        candidates_per_query=int(
            experiment_config["retrieval"]["candidates_per_query"]
        ),
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "sample_manifest.parquet"
    candidates_path = output_dir / "candidates.parquet"
    scoring_path = output_dir / "teacher_inputs.jsonl"
    metadata_path = output_dir / "export_metadata.json"

    manifest.write_parquet(manifest_path)
    candidates.write_parquet(candidates_path)
    write_jsonl(scoring_path, scoring_inputs)

    metadata = {
        "experiment_version": experiment_config["version"],
        "experiment_config": str(experiment_config_path),
        "iris_config": str(iris_config_path),
        "checkpoint": str(checkpoint_path),
        "table_name": experiment_config["iris"]["table_name"],
        "query_shape": list(trained_queries.shape),
        "documents": manifest.height,
        "candidate_rows": candidates.height,
        "maximum_candidates_per_document": (
            configured_queries
            * int(
                experiment_config["retrieval"][
                    "candidates_per_query"
                ]
            )
        ),
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print()
    print("Sample distribution:")
    print(
        manifest.group_by("split", "label_name")
        .len()
        .sort("split", "label_name")
    )
    print()
    print("Candidate counts per document:")
    print(
        manifest.select(
            pl.col("candidate_count").min().alias("min"),
            pl.col("candidate_count").median().alias("median"),
            pl.col("candidate_count").max().alias("max"),
        )
    )
    print()
    print(f"Manifest: {manifest_path}")
    print(f"Candidates: {candidates_path}")
    print(f"Teacher inputs: {scoring_path}")
    print(f"Metadata: {metadata_path}")
    print("Candidate export completed successfully.")


if __name__ == "__main__":
    main()
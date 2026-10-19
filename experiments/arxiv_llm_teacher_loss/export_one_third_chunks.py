"""Export a stratified training subset and reuse completed Qwen scores."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import lancedb
import numpy as np
import polars as pl
import tomllib

from iris.sampling import subsample_train_ids, train_val_test_split
from iris_llm import TargetCandidateForScoring

LABEL_NAMES = {0: "cs.AI", 1: "cs.DS", 2: "cs.PL"}
REQUIRED_COLUMNS = {
    "id",
    "chunk_id",
    "offset",
    "label",
    "source_path",
    "chunk_text",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def load_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def resolve_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo_root / path).resolve()


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


def load_reusable_scores(
    pattern: str,
    expected_labels: dict[tuple[int, int], str],
) -> tuple[dict[tuple[int, int], dict], dict]:
    paths = [Path(path) for path in sorted(glob.glob(pattern))]
    reused: dict[tuple[int, int], dict] = {}
    malformed_lines = 0
    duplicate_lines = 0
    irrelevant_records = 0

    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    key = (
                        int(payload["document_id"]),
                        int(payload["chunk_id"]),
                    )
                except Exception:
                    # A cancelled process can leave one incomplete final line.
                    malformed_lines += 1
                    continue

                expected_label = expected_labels.get(key)
                if expected_label is None:
                    irrelevant_records += 1
                    continue
                if int(payload.get("query_id", -1)) != -1:
                    raise ValueError(
                        f"Reusable score has unexpected query_id at "
                        f"{path}:{line_number}"
                    )
                if str(payload.get("target_label")) != expected_label:
                    raise ValueError(
                        f"Reusable score label mismatch at {path}:{line_number}"
                    )
                if key in reused:
                    duplicate_lines += 1
                    continue
                reused[key] = payload

    metadata = {
        "pattern": pattern,
        "files": [str(path) for path in paths],
        "reused_records": len(reused),
        "malformed_lines_skipped": malformed_lines,
        "duplicate_lines_skipped": duplicate_lines,
        "irrelevant_records_skipped": irrelevant_records,
    }
    return reused, metadata


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    config_path = resolve_path(repo_root, args.config)
    cfg = load_toml(config_path)

    output_dir = resolve_path(repo_root, cfg["paths"]["output_dir"])
    database_path = resolve_path(repo_root, cfg["database"]["path"])
    num_shards = int(cfg["export"]["num_shards"])
    fraction = float(cfg["subset"]["train_fraction"])
    seed = int(cfg["subset"]["seed"])
    reuse_pattern = str(cfg["subset"]["reuse_score_glob"])
    prompt_version = str(cfg["teacher"]["prompt_version"])

    if num_shards < 1:
        raise ValueError("num_shards must be at least 1")

    target_descriptions = {
        str(label): str(description)
        for label, description in cfg["target_descriptions"].items()
    }
    if set(target_descriptions) != set(LABEL_NAMES.values()):
        raise ValueError("Target descriptions must define cs.AI, cs.DS, and cs.PL")

    table = lancedb.connect(database_path).open_table(cfg["database"]["table_name"])
    missing_columns = REQUIRED_COLUMNS - set(table.schema.names)
    if missing_columns:
        raise ValueError(
            f"Vector table is missing columns: {sorted(missing_columns)}"
        )

    arrow_table = (
        table.to_lance()
        .scanner(columns=sorted(REQUIRED_COLUMNS))
        .to_table()
    )
    chunks = pl.from_arrow(arrow_table)
    if not isinstance(chunks, pl.DataFrame):
        raise TypeError("Expected the Lance scan to produce a table")

    documents = chunks.select("id", "label").unique().sort("id")
    document_ids = documents["id"].to_numpy()
    document_labels = documents["label"].to_numpy()
    train_ids, validation_ids, test_ids = train_val_test_split(
        document_ids,
        labels=document_labels,
        seed=seed,
    )
    label_by_id = {
        int(document_id): int(label)
        for document_id, label in zip(
            document_ids,
            document_labels,
            strict=True,
        )
    }
    train_labels = np.array(
        [label_by_id[int(document_id)] for document_id in train_ids]
    )
    selected_train_ids = subsample_train_ids(
        train_ids=train_ids,
        train_labels=train_labels,
        fraction=fraction,
        seed=seed,
    )

    selected = (
        chunks.filter(
            pl.col("id").is_in(
                pl.Series(selected_train_ids, dtype=pl.Int64)
            )
        )
        .sort("id", "chunk_id")
        .with_row_index("row_number")
        .with_columns(
            pl.col("label").replace_strict(LABEL_NAMES).alias("target_label"),
            (pl.col("row_number") % num_shards).alias("shard_id"),
            pl.col("chunk_text").str.len_chars().alias("chunk_text_chars"),
        )
    )

    unique_chunk_keys = selected.select(
        pl.struct("id", "chunk_id").n_unique()
    ).item()
    if unique_chunk_keys != selected.height:
        raise ValueError("Selected chunks contain duplicate (id, chunk_id) keys")
    if selected.filter(pl.col("chunk_text_chars") == 0).height:
        raise ValueError("Selected chunks contain empty chunk_text values")

    expected_labels = {
        (int(row["id"]), int(row["chunk_id"])): str(row["target_label"])
        for row in selected.select(
            "id", "chunk_id", "target_label"
        ).iter_rows(named=True)
    }
    reusable_scores, reuse_metadata = load_reusable_scores(
        reuse_pattern,
        expected_labels,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "all_chunks_manifest.parquet"
    selected.select(
        "id",
        "chunk_id",
        "offset",
        "label",
        "target_label",
        "source_path",
        "chunk_text_chars",
        "shard_id",
    ).write_parquet(manifest_path)

    selected_documents_path = output_dir / "selected_train_documents.parquet"
    documents.filter(
        pl.col("id").is_in(
            pl.Series(selected_train_ids, dtype=pl.Int64)
        )
    ).sort("id").write_parquet(selected_documents_path)

    shard_counts: dict[str, int] = {}
    reused_counts: dict[str, int] = {}
    missing_counts: dict[str, int] = {}
    shard_inputs: list[str] = []
    shard_outputs: list[str] = []

    for shard_id in range(num_shards):
        shard = selected.filter(pl.col("shard_id") == shard_id)
        input_path = output_dir / shard_name(
            "teacher_inputs", shard_id, num_shards, ".jsonl"
        )
        score_path = output_dir / shard_name(
            f"qwen_scores.{prompt_version}",
            shard_id,
            num_shards,
            ".jsonl",
        )
        shard_inputs.append(str(input_path))
        shard_outputs.append(str(score_path))
        shard_counts[str(shard_id)] = shard.height

        shard_reused = 0
        with (
            input_path.open("w", encoding="utf-8") as input_handle,
            score_path.open("w", encoding="utf-8") as score_handle,
        ):
            for row in shard.iter_rows(named=True):
                record = TargetCandidateForScoring(
                    document_id=int(row["id"]),
                    query_id=-1,
                    chunk_id=int(row["chunk_id"]),
                    target_label=str(row["target_label"]),
                    target_description=target_descriptions[str(row["target_label"])],
                    chunk_text=str(row["chunk_text"]),
                )
                input_handle.write(record.model_dump_json() + "\n")

                key = (record.document_id, record.chunk_id)
                existing = reusable_scores.get(key)
                if existing is not None:
                    score_handle.write(
                        json.dumps(existing, ensure_ascii=False) + "\n"
                    )
                    shard_reused += 1

        reused_counts[str(shard_id)] = shard_reused
        missing_counts[str(shard_id)] = shard.height - shard_reused

    selected_label_counts = {
        LABEL_NAMES[int(row["label"])]: int(row["len"])
        for row in (
            documents.filter(
                pl.col("id").is_in(
                    pl.Series(selected_train_ids, dtype=pl.Int64)
                )
            )
            .group_by("label")
            .len()
            .sort("label")
            .iter_rows(named=True)
        )
    }
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": cfg["version"],
        "database_path": str(database_path),
        "table_name": cfg["database"]["table_name"],
        "table_version": table.to_lance().version,
        "seed": seed,
        "train_fraction": fraction,
        "all_documents": len(document_ids),
        "full_train_documents": len(train_ids),
        "selected_train_documents": len(selected_train_ids),
        "validation_documents": len(validation_ids),
        "test_documents": len(test_ids),
        "selected_document_label_counts": selected_label_counts,
        "records": selected.height,
        "reused_records": len(reusable_scores),
        "records_to_score": selected.height - len(reusable_scores),
        "num_shards": num_shards,
        "shard_counts": shard_counts,
        "reused_counts": reused_counts,
        "missing_counts": missing_counts,
        "reuse": reuse_metadata,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "selected_documents": str(selected_documents_path),
        "selected_documents_sha256": sha256_file(selected_documents_path),
        "shard_inputs": shard_inputs,
        "shard_outputs": shard_outputs,
    }
    metadata_path = output_dir / "export_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(metadata, indent=2))
    print("One-third training chunk export completed successfully.")


if __name__ == "__main__":
    main()

"""Export every arXiv vector chunk for sharded Qwen teacher scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import lancedb
import polars as pl
import tomllib

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
    parser.add_argument(
        "--config",
        default=Path(__file__).with_name("config.toml"),
        type=Path,
    )
    parser.add_argument("--database-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
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

    database_path = args.database_path or resolve_path(
        repo_root, cfg["database"]["path"]
    )
    output_dir = args.output_dir or resolve_path(
        repo_root, cfg["paths"]["output_dir"]
    )
    num_shards = args.num_shards or int(cfg["export"]["num_shards"])
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1")

    target_descriptions = {
        str(label): str(description)
        for label, description in cfg["target_descriptions"].items()
    }
    if set(target_descriptions) != set(LABEL_NAMES.values()):
        raise ValueError("Target descriptions must define cs.AI, cs.DS, and cs.PL")

    table = lancedb.connect(database_path).open_table(cfg["database"]["table_name"])
    available_columns = set(table.schema.names)
    missing = REQUIRED_COLUMNS - available_columns
    if missing:
        raise ValueError(f"Vector table is missing columns: {sorted(missing)}")

    arrow_table = (
        table.to_lance()
        .scanner(columns=sorted(REQUIRED_COLUMNS))
        .to_table()
    )
    chunks = pl.from_arrow(arrow_table)
    if not isinstance(chunks, pl.DataFrame):
        raise TypeError("Expected the Lance scan to produce a table")

    chunks = (
        chunks.sort("id", "chunk_id")
        .with_row_index("row_number")
        .with_columns(
            pl.col("label").replace_strict(LABEL_NAMES).alias("target_label"),
            (pl.col("row_number") % num_shards).alias("shard_id"),
            pl.col("chunk_text").str.len_chars().alias("chunk_text_chars"),
        )
    )

    if chunks.select(pl.struct("id", "chunk_id").n_unique()).item() != chunks.height:
        raise ValueError("The vector table contains duplicate (id, chunk_id) keys")
    if chunks.filter(pl.col("chunk_text_chars") == 0).height:
        raise ValueError("The vector table contains empty chunk_text values")
    observed_labels = set(chunks["target_label"].unique().to_list())
    if observed_labels != set(LABEL_NAMES.values()):
        raise ValueError(f"Unexpected labels: {sorted(observed_labels)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "all_chunks_manifest.parquet"
    manifest_columns = [
        "id",
        "chunk_id",
        "offset",
        "label",
        "target_label",
        "source_path",
        "chunk_text_chars",
        "shard_id",
    ]
    chunks.select(manifest_columns).write_parquet(manifest_path)

    shard_counts: dict[str, int] = {}
    shard_inputs: list[str] = []
    for shard_id in range(num_shards):
        shard = chunks.filter(pl.col("shard_id") == shard_id)
        input_path = output_dir / shard_name(
            "teacher_inputs", shard_id, num_shards, ".jsonl"
        )
        shard_inputs.append(str(input_path))
        shard_counts[str(shard_id)] = shard.height

        with input_path.open("w", encoding="utf-8") as handle:
            for row in shard.iter_rows(named=True):
                record = TargetCandidateForScoring(
                    document_id=int(row["id"]),
                    query_id=-1,
                    chunk_id=int(row["chunk_id"]),
                    target_label=str(row["target_label"]),
                    target_description=target_descriptions[str(row["target_label"])],
                    chunk_text=str(row["chunk_text"]),
                )
                handle.write(record.model_dump_json() + "\n")

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": cfg["version"],
        "database_path": str(database_path),
        "table_name": cfg["database"]["table_name"],
        "table_version": table.to_lance().version,
        "records": chunks.height,
        "documents": chunks["id"].n_unique(),
        "num_shards": num_shards,
        "shard_counts": shard_counts,
        "label_counts": {
            row["target_label"]: int(row["len"])
            for row in chunks.group_by("target_label").len().iter_rows(named=True)
        },
        "target_descriptions": target_descriptions,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "shard_inputs": shard_inputs,
    }
    metadata_path = output_dir / "export_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(metadata, indent=2))
    print("Full chunk export completed successfully.")


if __name__ == "__main__":
    main()

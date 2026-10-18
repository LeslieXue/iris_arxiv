import argparse
import hashlib
import json
import tomllib
from pathlib import Path

import polars as pl

from iris_llm import TargetCandidateForScoring


EXPECTED_LABELS = {"cs.AI", "cs.DS", "cs.PL"}
KEY_FIELDS = ("document_id", "query_id", "chunk_id")


def parse_args():
    default_config = Path(__file__).with_name("config.target.toml")

    parser = argparse.ArgumentParser(
        description=(
            "Build target-conditioned teacher inputs from the fixed "
            "validation audit"
        )
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


def resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_key(record: dict) -> tuple[int, int, int]:
    return tuple(int(record[field]) for field in KEY_FIELDS)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [
            json.loads(line)
            for line in handle
            if line.strip()
        ]


def main():
    args = parse_args()

    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    config_path = resolve_repo_path(repo_root, str(args.config))
    cfg = load_toml(config_path)

    source_dir = resolve_repo_path(
        repo_root,
        cfg["source"]["output_dir"],
    )
    output_dir = resolve_repo_path(
        repo_root,
        cfg["paths"]["output_dir"],
    )

    source_manifest_path = source_dir / "audit_manifest.parquet"
    source_input_path = source_dir / "teacher_inputs.audit.jsonl"
    output_manifest_path = output_dir / "audit_manifest.parquet"
    output_input_path = output_dir / "target_teacher_inputs.audit.jsonl"
    output_metadata_path = output_dir / "target_audit_metadata.json"

    for path in (source_manifest_path, source_input_path):
        if not path.exists():
            raise FileNotFoundError(f"Source audit artifact not found: {path}")

    manifest = pl.read_parquet(source_manifest_path).sort("audit_id")
    source_inputs = read_jsonl(source_input_path)
    source_by_key = {record_key(row): row for row in source_inputs}

    if len(source_by_key) != len(source_inputs):
        raise ValueError("Duplicate keys in source teacher inputs")

    required_manifest_columns = {
        "audit_id",
        "split",
        "document_id",
        "query_id",
        "chunk_id",
        "label_name",
    }
    missing = required_manifest_columns - set(manifest.columns)
    if missing:
        raise ValueError(
            f"Audit manifest is missing columns: {sorted(missing)}"
        )

    if set(manifest["split"].unique().to_list()) != {"validation"}:
        raise ValueError("Target prompt development must use validation only")

    observed_labels = set(manifest["label_name"].unique().to_list())
    if observed_labels != EXPECTED_LABELS:
        raise ValueError(f"Unexpected target labels: {sorted(observed_labels)}")

    target_descriptions = {
        str(label): str(description)
        for label, description in cfg["target_descriptions"].items()
    }
    if set(target_descriptions) != EXPECTED_LABELS:
        raise ValueError(
            "Target descriptions must define exactly cs.AI, cs.DS, and cs.PL"
        )
    if any(not value.strip() for value in target_descriptions.values()):
        raise ValueError("Target descriptions must not be empty")

    manifest_keys = {
        record_key(row)
        for row in manifest.select(KEY_FIELDS).to_dicts()
    }
    if manifest_keys != set(source_by_key):
        raise ValueError(
            "Source teacher-input keys do not match the audit manifest"
        )

    target_records = []
    for row in manifest.iter_rows(named=True):
        key = record_key(row)
        source = source_by_key[key]
        label = str(row["label_name"])
        target_records.append(
            TargetCandidateForScoring(
                document_id=key[0],
                query_id=key[1],
                chunk_id=key[2],
                target_label=label,
                target_description=target_descriptions[label],
                chunk_text=str(source["chunk_text"]),
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.write_parquet(output_manifest_path)

    safe_keys = {
        "document_id",
        "query_id",
        "chunk_id",
        "target_label",
        "target_description",
        "chunk_text",
    }
    with output_input_path.open("w", encoding="utf-8") as handle:
        for record in target_records:
            payload = record.model_dump()
            if set(payload) != safe_keys:
                raise RuntimeError(
                    "Unexpected fields in target-conditioned teacher input"
                )
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    metadata = {
        "version": cfg["version"],
        "split": "validation",
        "records": len(target_records),
        "distinct_documents": manifest["document_id"].n_unique(),
        "target_distribution": {
            label: int(
                manifest.filter(pl.col("label_name") == label).height
            )
            for label in sorted(EXPECTED_LABELS)
        },
        "target_descriptions": target_descriptions,
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_teacher_inputs": str(source_input_path),
        "source_teacher_inputs_sha256": sha256_file(source_input_path),
        "output_manifest": str(output_manifest_path),
        "output_teacher_inputs": str(output_input_path),
        "teacher_prompt_version": cfg["teacher"]["prompt_version"],
    }
    output_metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Target distribution:")
    print(
        manifest.group_by("label_name")
        .len()
        .sort("label_name")
    )
    print("Manifest:", output_manifest_path)
    print("Teacher inputs:", output_input_path)
    print("Metadata:", output_metadata_path)
    print("Target-conditioned audit inputs completed.")


if __name__ == "__main__":
    main()

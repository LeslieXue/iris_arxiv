import argparse
import hashlib
import json
import tomllib
from pathlib import Path

import numpy as np
import polars as pl

from iris_llm import CandidateForScoring


LABEL_NAMES = {
    0: "cs.AI",
    1: "cs.DS",
    2: "cs.PL",
}

RANK_BANDS = {
    "high": (1, 2),
    "medium": (3, 5),
    "low": (6, 8),
}


def parse_args():
    default_config = Path(__file__).with_name("config.toml")

    parser = argparse.ArgumentParser(
        description="Build a balanced validation audit set for prompt development"
    )
    parser.add_argument(
        "--config",
        default=default_config,
        help="Experiment configuration",
    )
    parser.add_argument(
        "--per-class-per-band",
        type=int,
        default=4,
        help="Number of chunks for each class and rank band",
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


def main():
    args = parse_args()

    if args.per_class_per_band < 1:
        raise ValueError("--per-class-per-band must be at least 1")

    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    config_path = resolve_repo_path(repo_root, str(args.config))
    cfg = load_toml(config_path)

    output_dir = resolve_repo_path(
        repo_root,
        cfg["paths"]["output_dir"],
    )
    candidates_path = output_dir / "candidates.parquet"
    audit_manifest_path = output_dir / "audit_manifest.parquet"
    audit_input_path = output_dir / "teacher_inputs.audit.jsonl"
    audit_metadata_path = output_dir / "audit_metadata.json"

    if not candidates_path.exists():
        raise FileNotFoundError(
            f"Candidate file not found: {candidates_path}"
        )

    candidates = pl.read_parquet(candidates_path)

    required_columns = {
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
    missing = required_columns - set(candidates.columns)

    if missing:
        raise ValueError(
            f"Candidate file is missing columns: {sorted(missing)}"
        )

    validation = candidates.filter(
        pl.col("split") == "validation"
    )

    if validation.height == 0:
        raise ValueError("No validation candidates were found")

    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed)

    selected_rows = []
    used_documents = set()

    for label, expected_name in LABEL_NAMES.items():
        class_rows = validation.filter(
            pl.col("label") == label
        )

        observed_names = set(
            class_rows["label_name"].unique().to_list()
        )

        if observed_names != {expected_name}:
            raise ValueError(
                f"Unexpected label names for label {label}: "
                f"{sorted(observed_names)}"
            )

        for band_name, (minimum_rank, maximum_rank) in RANK_BANDS.items():
            band_rows = class_rows.filter(
                pl.col("candidate_rank").is_between(
                    minimum_rank,
                    maximum_rank,
                    closed="both",
                )
            )

            available_documents = sorted(
                document_id
                for document_id in band_rows[
                    "document_id"
                ].unique().to_list()
                if document_id not in used_documents
            )

            if len(available_documents) < args.per_class_per_band:
                raise ValueError(
                    f"Only {len(available_documents)} unused documents "
                    f"are available for {expected_name}/{band_name}; "
                    f"{args.per_class_per_band} are required."
                )

            selected_documents = rng.choice(
                np.asarray(available_documents, dtype=np.int64),
                size=args.per_class_per_band,
                replace=False,
            )

            for document_id in selected_documents.tolist():
                document_rows = band_rows.filter(
                    pl.col("document_id") == int(document_id)
                ).sort(
                    [
                        "query_id",
                        "candidate_rank",
                        "chunk_id",
                    ]
                )

                row_index = int(
                    rng.integers(0, document_rows.height)
                )
                row = document_rows.row(
                    row_index,
                    named=True,
                )

                selected_rows.append(
                    {
                        "rank_band": band_name,
                        **row,
                    }
                )
                used_documents.add(int(document_id))

    selected_rows.sort(
        key=lambda row: (
            int(row["label"]),
            row["rank_band"],
            int(row["document_id"]),
        )
    )

    manifest_rows = []
    safe_records = []

    for audit_id, row in enumerate(selected_rows):
        manifest_rows.append(
            {
                "audit_id": audit_id,
                "split": row["split"],
                "rank_band": row["rank_band"],
                "document_id": int(row["document_id"]),
                "query_id": int(row["query_id"]),
                "chunk_id": int(row["chunk_id"]),
                "candidate_rank": int(row["candidate_rank"]),
                "vector_distance": float(row["vector_distance"]),
                "label": int(row["label"]),
                "label_name": str(row["label_name"]),
                "source_path": str(row["source_path"]),
                "chunk_text_chars": len(str(row["chunk_text"])),
            }
        )

        safe_records.append(
            CandidateForScoring(
                document_id=int(row["document_id"]),
                query_id=int(row["query_id"]),
                chunk_id=int(row["chunk_id"]),
                chunk_text=str(row["chunk_text"]),
            )
        )

    expected_total = (
        len(LABEL_NAMES)
        * len(RANK_BANDS)
        * args.per_class_per_band
    )

    if len(manifest_rows) != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} audit records, "
            f"created {len(manifest_rows)}"
        )

    manifest = pl.DataFrame(manifest_rows)

    if manifest["document_id"].n_unique() != expected_total:
        raise RuntimeError(
            "Audit records must come from distinct documents"
        )

    if set(manifest["split"].unique().to_list()) != {"validation"}:
        raise RuntimeError(
            "The prompt-development audit set must use validation only"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.write_parquet(audit_manifest_path)

    with audit_input_path.open("w", encoding="utf-8") as handle:
        for record in safe_records:
            payload = record.model_dump()

            forbidden = {
                "label",
                "label_name",
                "source_path",
                "candidate_rank",
                "vector_distance",
                "rank_band",
                "split",
            }

            leaked = forbidden & set(payload)

            if leaked:
                raise RuntimeError(
                    f"Protected fields leaked into teacher input: "
                    f"{sorted(leaked)}"
                )

            handle.write(
                json.dumps(payload, ensure_ascii=False) + "\n"
            )

    metadata = {
        "seed": seed,
        "source_candidates": str(candidates_path),
        "source_candidates_sha256": sha256_file(
            candidates_path
        ),
        "split": "validation",
        "prompt_version_at_creation": cfg["teacher"][
            "prompt_version"
        ],
        "per_class_per_rank_band": args.per_class_per_band,
        "rank_bands": {
            name: {
                "minimum_candidate_rank": limits[0],
                "maximum_candidate_rank": limits[1],
            }
            for name, limits in RANK_BANDS.items()
        },
        "audit_records": expected_total,
        "distinct_documents": manifest[
            "document_id"
        ].n_unique(),
        "manifest": str(audit_manifest_path),
        "teacher_inputs": str(audit_input_path),
    }

    audit_metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Audit distribution:")
    print(
        manifest.group_by(
            ["label_name", "rank_band"]
        )
        .len()
        .sort(["label_name", "rank_band"])
    )

    print(
        "Distinct documents:",
        manifest["document_id"].n_unique(),
    )
    print("Audit manifest:", audit_manifest_path)
    print("Safe teacher inputs:", audit_input_path)
    print("Audit metadata:", audit_metadata_path)
    print("Validation audit set completed successfully.")


if __name__ == "__main__":
    main()
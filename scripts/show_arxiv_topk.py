"""Display top-k arXiv chunks and their original source text."""

from __future__ import annotations

import argparse
from pathlib import Path

import lancedb
import numpy as np
import torch

from iris.config import get_config
from iris.dataset import make_loaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="test",
        help="Dataset split containing the document",
    )
    parser.add_argument(
        "--document-id",
        type=int,
        default=None,
        help="Internal IRIS document ID; defaults to the first ID in the split",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional trained IRIS checkpoint containing learned queries",
    )

    return parser.parse_args()


def load_checkpoint_queries(checkpoint: Path) -> np.ndarray:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    state_dict = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    if "queries" not in state_dict:
        raise ValueError(
            f"Checkpoint {checkpoint} does not contain IRIS queries"
        )

    return state_dict["queries"].detach().cpu().numpy()


def main() -> None:
    args = parse_args()
    cfg = get_config()

    np.random.seed(cfg.seed)

    db = lancedb.connect(cfg.database.path)
    table = db.open_table("arxiv11")

    vector_type = table.schema.field("vector").type
    cfg.model.d_model = vector_type.list_size

    loaders = make_loaders(cfg=cfg, tbl=table)

    datasets = {
        "train": loaders.train.dataset,
        "validation": loaders.validation.dataset,
        "test": loaders.test.dataset,
    }
    dataset = datasets[args.split]

    if args.checkpoint is None:
        query_source = "initial_vector_index_queries"
    else:
        queries = load_checkpoint_queries(args.checkpoint)

        if queries.shape[1] != cfg.model.d_model:
            raise ValueError(
                "Checkpoint query dimension does not match the vector database: "
                f"{queries.shape[1]} != {cfg.model.d_model}"
            )

        dataset.queries = queries
        query_source = f"trained_checkpoint:{args.checkpoint}"

    if args.document_id is None:
        document_id = int(dataset.ids[0])
    else:
        document_id = args.document_id

    if document_id not in dataset.ids:
        raise ValueError(
            f"Document ID {document_id} is not present in the "
            f"{args.split} split"
        )

    result = dataset.search_chunks(
        document_id=document_id,
        include_text=True,
    )

    if result.is_empty():
        print(f"No chunks found for document ID {document_id}")
        return

    metadata_columns = [
        "id",
        "chunk_id",
        "offset",
        "query_id",
        "rank",
        "_distance",
        "label",
        "source_path",
    ]

    print(f"query_source={query_source}")
    print(f"split={args.split}")
    print(f"document_id={document_id}")
    print(result.select(metadata_columns))

    text_rows = result.select(
        "query_id",
        "rank",
        "id",
        "chunk_id",
        "source_path",
        "chunk_text",
    ).iter_rows(named=True)

    for row in text_rows:
        print()
        print(
            f"query_id={row['query_id']} "
            f"rank={row['rank']} "
            f"document_id={row['id']} "
            f"chunk_id={row['chunk_id']}"
        )
        print(f"source_path={row['source_path']}")
        print(row["chunk_text"])


if __name__ == "__main__":
    main()
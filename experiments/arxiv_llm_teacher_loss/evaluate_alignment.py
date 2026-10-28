"""Measure how a trained IRIS checkpoint aligns with Qwen teacher scores."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import lancedb
import numpy as np
import polars as pl
import tomllib
import torch
from run_training import find_variant
from transformers import AutoConfig

from iris.config import get_config
from iris.dataset import make_loaders

SCORE_COLUMNS = [
    "document_id",
    "chunk_id",
    "teacher_utility",
    "target_evidence",
    "substantive_relevance",
    "specificity",
    "context_completeness",
    "direct_label_disclosure",
    "keyword_only_risk",
]

LABEL_NAMES = {
    0: "cs.AI",
    1: "cs.DS",
    2: "cs.PL",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--iris-config", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint override for repeated-seed experiments",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="test",
        help="Dataset split on which to measure Qwen-score alignment",
    )
    return parser.parse_args()


def load_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def summarize(frame: pl.DataFrame) -> dict:
    """Summarize retrieval quality with equal weight for each document."""
    distribution = Counter(frame["target_evidence"].to_list())

    document_level = frame.group_by("document_id").agg(
        pl.len().alias("selected_chunks"),
        pl.col("teacher_utility")
        .mean()
        .alias("mean_teacher_utility"),
        pl.col("target_evidence")
        .mean()
        .alias("mean_target_evidence"),
        (pl.col("target_evidence") >= 3)
        .cast(pl.Float64)
        .mean()
        .alias("high_evidence_rate"),
        (pl.col("target_evidence") == 0)
        .cast(pl.Float64)
        .mean()
        .alias("zero_evidence_rate"),
        pl.col("substantive_relevance")
        .mean()
        .alias("mean_substantive_relevance"),
        pl.col("specificity")
        .mean()
        .alias("mean_specificity"),
        pl.col("context_completeness")
        .mean()
        .alias("mean_context_completeness"),
        pl.col("direct_label_disclosure")
        .mean()
        .alias("mean_direct_label_disclosure"),
        pl.col("keyword_only_risk")
        .mean()
        .alias("mean_keyword_only_risk"),
    )

    return {
        "documents": document_level.height,
        "selected_chunks": frame.height,
        "mean_selected_chunks_per_document": (
            document_level["selected_chunks"].mean()
        ),
        "mean_teacher_utility": (
            document_level["mean_teacher_utility"].mean()
        ),
        "mean_target_evidence": (
            document_level["mean_target_evidence"].mean()
        ),
        "high_evidence_rate": (
            document_level["high_evidence_rate"].mean()
        ),
        "zero_evidence_rate": (
            document_level["zero_evidence_rate"].mean()
        ),
        "mean_substantive_relevance": (
            document_level["mean_substantive_relevance"].mean()
        ),
        "mean_specificity": (
            document_level["mean_specificity"].mean()
        ),
        "mean_context_completeness": (
            document_level["mean_context_completeness"].mean()
        ),
        "mean_direct_label_disclosure": (
            document_level["mean_direct_label_disclosure"].mean()
        ),
        "mean_keyword_only_risk": (
            document_level["mean_keyword_only_risk"].mean()
        ),
        "target_evidence_distribution": {
            str(score): int(distribution.get(score, 0))
            for score in range(5)
        },
    }


def main() -> None:
    args = parse_args()
    experiment = load_toml(args.experiment_config)
    find_variant(experiment, args.variant)
    training = experiment["training"]
    cfg = get_config(args.iris_config)
    subset = experiment.get("subset", {})
    if subset:
        subset_seed = int(subset["seed"])
        if cfg.seed != subset_seed:
            raise ValueError(
                f"IRIS seed {cfg.seed} does not match subset seed {subset_seed}"
            )
        cfg.trainer.train_fraction = float(subset["train_fraction"])
    checkpoint = args.checkpoint or (
        Path(training["checkpoint_dir"]) / f"{args.variant}.pt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    np.random.seed(cfg.seed)
    table = lancedb.connect(cfg.database.path).open_table("arxiv11")
    cfg.model.d_model = AutoConfig.from_pretrained(cfg.model_id).hidden_size
    loaders = make_loaders(
        cfg=cfg,
        tbl=table,
        include_teacher_scores=False,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    queries = state["queries"].numpy()
    dataset = getattr(loaders, args.split).dataset
    dataset.queries = queries  # type: ignore

    selected_frames = []
    document_ids = dataset.ids  # type: ignore
    for index, document_id in enumerate(document_ids, start=1):
        chunks = dataset.search_chunks(int(document_id)).select(
            pl.col("id").alias("document_id"),
            "chunk_id",
            "label",
            "query_id",
            "rank",
            "_distance",
        )
        selected_frames.append(chunks)
        if index % 100 == 0 or index == len(document_ids):
            print(
                "Retrieved teacher-alignment chunks for "
                f"{index}/{len(document_ids)} documents"
            )

    selected = pl.concat(selected_frames)
    teacher_scores = pl.read_parquet(
        training["teacher_scores_path"],
        columns=SCORE_COLUMNS,
    )
    scored = selected.join(
        teacher_scores,
        on=["document_id", "chunk_id"],
        how="left",
        validate="m:1",
    )
    missing = scored.filter(pl.col("teacher_utility").is_null())
    if missing.height:
        raise ValueError(
            f"Selected chunks are missing {missing.height} teacher scores"
        )

    metrics = {
        "important_caveat": (
            "Teacher alignment measures compliance with the Qwen scoring "
            "preference. It is not independent evidence of explanation quality."
        ),
        "experiment_version": experiment["version"],
        "variant": args.variant,
        "checkpoint": str(checkpoint),
        "alignment_split": args.split,
        "alignment_documents": len(document_ids),
        "alignment_aggregation": (
            "Mean within each document, then mean across documents"
        ),
        "overall": summarize(scored),
        "per_class": {
            label_name: summarize(
                scored.filter(pl.col("label") == label_id)
            )
            for label_id, label_name in LABEL_NAMES.items()
        },
        "per_query": {
            str(query_id): summarize(
                scored.filter(pl.col("query_id") == query_id)
            )
            for query_id in sorted(scored["query_id"].unique().to_list())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))
    print(f"Teacher-alignment metrics written to: {args.output}")


if __name__ == "__main__":
    main()

"""Train one full-data IRIS teacher-loss experiment variant."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import lancedb
import numpy as np
import tomllib
import torch
from transformers import AutoConfig

from iris.config import get_config
from iris.dataset import get_num_classes
from iris.train import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--iris-config", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    return parser.parse_args()


def load_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def find_variant(experiment: dict, name: str) -> dict:
    variants = experiment["training"]["variants"]
    matches = [variant for variant in variants if variant["name"] == name]
    if len(matches) != 1:
        available = [variant["name"] for variant in variants]
        raise ValueError(f"Unknown variant {name!r}; available variants: {available}")
    return matches[0]


def main() -> None:
    args = parse_args()
    experiment = load_toml(args.experiment_config)
    variant = find_variant(experiment, args.variant)
    training = experiment["training"]
    cfg = get_config(args.iris_config)

    checkpoint_dir = Path(training["checkpoint_dir"])
    metadata_dir = Path(training["metadata_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    cfg.trainer.checkpoint = checkpoint_dir / f"{args.variant}.pt"
    cfg.teacher_loss.enabled = bool(variant["enabled"])
    cfg.teacher_loss.scores_path = Path(training["teacher_scores_path"])
    cfg.teacher_loss.weight = float(variant["weight"])
    cfg.teacher_loss.pair_temperature = float(
        training["teacher_pair_temperature"]
    )
    cfg.teacher_loss.min_score_gap = float(training["teacher_min_score_gap"])
    cfg.teacher_loss.score_mode = str(variant["score_mode"])
    cfg.teacher_loss.shuffle_seed = cfg.seed

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.set_float32_matmul_precision("high")

    table = lancedb.connect(cfg.database.path).open_table("arxiv11")
    cfg.model.d_model = AutoConfig.from_pretrained(cfg.model_id).hidden_size
    cfg.model.output_dim = get_num_classes(table)
    cfg.task = "binary" if cfg.model.output_dim == 2 else "multiclass"

    run_metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_version": experiment["version"],
        "variant": args.variant,
        "iris_config": str(args.iris_config),
        "checkpoint": str(cfg.trainer.checkpoint),
        "seed": cfg.seed,
        "epochs": cfg.trainer.max_epochs,
        "teacher_loss": cfg.teacher_loss.model_dump(mode="json"),
        "status": "started",
    }
    metadata_path = metadata_dir / f"{args.variant}.training.json"
    metadata_path.write_text(
        json.dumps(run_metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(run_metadata, indent=2))

    train_model(cfg=cfg, tbl=table, use_best=False)

    run_metadata["status"] = "completed"
    run_metadata["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    run_metadata["checkpoint_size_bytes"] = cfg.trainer.checkpoint.stat().st_size
    metadata_path.write_text(
        json.dumps(run_metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Completed variant: {args.variant}")


if __name__ == "__main__":
    main()

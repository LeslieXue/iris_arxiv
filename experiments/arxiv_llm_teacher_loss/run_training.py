"""Train one full-data IRIS teacher-loss experiment variant."""

from __future__ import annotations

import argparse
import json
import random
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
    parser.add_argument(
        "--training-seed",
        type=int,
        default=None,
        help=(
            "Seed for query initialization, batching, and dropout. "
            "The IRIS config seed remains the fixed document-split seed."
        ),
    )
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
    training_seed = (
        cfg.seed if args.training_seed is None else args.training_seed
    )
    expected_split_seed = training.get("split_seed")
    if (
        expected_split_seed is not None
        and cfg.seed != int(expected_split_seed)
    ):
        raise ValueError(
            f"IRIS split seed {cfg.seed} does not match experiment "
            f"split seed {expected_split_seed}"
        )

    subset = experiment.get("subset", {})
    if subset:
        subset_seed = int(subset["seed"])
        if cfg.seed != subset_seed:
            raise ValueError(
                f"IRIS seed {cfg.seed} does not match subset seed {subset_seed}"
            )
        cfg.trainer.train_fraction = float(subset["train_fraction"])

    checkpoint_dir = Path(training["checkpoint_dir"])
    metadata_dir = Path(training["metadata_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.variant
    if args.training_seed is not None:
        run_name = f"{args.variant}.seed-{training_seed}"
    cfg.trainer.checkpoint = checkpoint_dir / f"{run_name}.pt"
    if bool(training.get("resume", False)):
        cfg.trainer.resume_checkpoint = (
            checkpoint_dir / f"{run_name}.resume.pt"
        )
    cfg.teacher_loss.enabled = bool(variant["enabled"])
    cfg.teacher_loss.scores_path = Path(training["teacher_scores_path"])
    cfg.teacher_loss.weight = float(variant["weight"])
    cfg.teacher_loss.pair_temperature = float(
        training["teacher_pair_temperature"]
    )
    cfg.teacher_loss.min_score_gap = float(training["teacher_min_score_gap"])
    cfg.teacher_loss.score_mode = str(variant["score_mode"])
    cfg.teacher_loss.shuffle_seed = training_seed

    random.seed(training_seed)
    torch.manual_seed(training_seed)
    torch.cuda.manual_seed_all(training_seed)
    np.random.seed(training_seed)
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
        "resume_checkpoint": (
            str(cfg.trainer.resume_checkpoint)
            if cfg.trainer.resume_checkpoint is not None
            else None
        ),
        "resuming": (
            cfg.trainer.resume_checkpoint is not None
            and cfg.trainer.resume_checkpoint.is_file()
        ),
        "split_seed": cfg.seed,
        "training_seed": training_seed,
        "max_epochs": cfg.trainer.max_epochs,
        "early_stopping_patience": cfg.trainer.early_stopping_patience,
        "early_stopping_min_delta": cfg.trainer.early_stopping_min_delta,
        "minimum_epochs": cfg.trainer.minimum_epochs,
        "early_stopping_metric": cfg.trainer.early_stopping_metric,
        "train_fraction": cfg.trainer.train_fraction,
        "teacher_loss": cfg.teacher_loss.model_dump(mode="json"),
        "status": "started",
    }
    metadata_path = metadata_dir / f"{run_name}.training.json"
    metadata_path.write_text(
        json.dumps(run_metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(run_metadata, indent=2))

    trainer = train_model(cfg=cfg, tbl=table, use_best=False)

    run_metadata["status"] = "completed"
    run_metadata["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    run_metadata["checkpoint_size_bytes"] = cfg.trainer.checkpoint.stat().st_size
    run_metadata["epochs_trained"] = trainer.epochs_trained
    run_metadata["best_epoch"] = trainer.best_epoch
    run_metadata["best_monitor_value"] = trainer.best_monitor_value
    metadata_path.write_text(
        json.dumps(run_metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Completed variant: {args.variant}")


if __name__ == "__main__":
    main()

from pathlib import Path
from typing import Any, Literal

import tomllib
import torch
from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    in_path: Path
    inter_path: Path
    out_path: Path
    train_size: float
    chunk_length: int
    overlap: int


class DataloaderConfig(BaseModel):
    batch_size: int
    pin_memory: bool
    num_workers: int


class DatabaseConfig(BaseModel):
    path: Path
    num_partitions: int
    num_sub_vectors: int
    k_neighbors: int


class ModelConfig(BaseModel):
    num_queries: int
    d_model: int
    nhead: int
    hidden_dim: int
    dropout: float
    output_dim: int
    query_ini_random: bool
    auto_device_map: bool = True
    device: str = "cpu"


class OptimizerConfig(BaseModel):
    lr: float
    weight_decay: float


class TrainerConfig(BaseModel):
    max_epochs: int
    checkpoint: Path
    ignore_index: int
    gradient_clip: float
    temperature: float
    train_fraction: float = Field(default=1.0, gt=0.0, le=1.0)
    early_stopping_patience: int | None = Field(default=None, ge=1)
    early_stopping_min_delta: float = Field(default=0.0, ge=0.0)
    minimum_epochs: int = Field(default=1, ge=1)
    early_stopping_metric: Literal[
        "validation_loss",
        "classification_loss",
    ] = "validation_loss"
    device: str = "cpu"


class TeacherLossConfig(BaseModel):
    enabled: bool = False
    scores_path: Path | None = None
    weight: float = 0.0
    pair_temperature: float = 1.0
    min_score_gap: float = 0.05
    score_mode: Literal["real", "shuffled"] = "real"
    shuffle_seed: int = 42


class TunerConfig(BaseModel):
    path: Path
    n_trials: int
    prune: bool


class HyperparameterConfig(BaseModel):
    max_epochs: dict
    lr: dict
    weight_decay: dict
    dropout: dict
    hidden_dim: dict
    temperature: dict


class EvaluatorConfig(BaseModel):
    n_bootstraps: int
    path: Path


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():
        return "mps"
    else:
        return "cpu"


class Config(BaseModel):
    seed: int
    fast_dev_run: bool
    distributed: bool
    quantize: bool
    compile: bool
    make_dataset: bool
    make_database: bool
    tune: bool
    train: bool
    evaluate: bool
    model_id: str
    verbose: bool
    dataset: DatasetConfig
    dataloader: DataloaderConfig
    database: DatabaseConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    trainer: TrainerConfig
    tuner: TunerConfig
    hparams: HyperparameterConfig
    evaluator: EvaluatorConfig
    teacher_loss: TeacherLossConfig = Field(default_factory=TeacherLossConfig)
    task: Literal["binary", "multiclass"] = "binary"

    def model_post_init(self, __context: Any) -> None:
        device = get_device()
        self.trainer.device = device
        self.model.device = device


def get_config(path: str | Path = "config.toml"):
    with open(path, "rb") as f:
        config_data = tomllib.load(f)
    return Config(**config_data)

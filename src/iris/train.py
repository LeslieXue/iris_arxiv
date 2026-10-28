import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from lancedb.table import LanceTable
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import ExponentialLR, LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, Metric
from tqdm import tqdm

from iris.config import Config, TrainerConfig
from iris.dataset import DataLoaders, make_loaders
from iris.loss import QueryLoss
from iris.model import IRIS, make_model


class Trainer:
    def __init__(
        self,
        loaders: DataLoaders,
        model: IRIS,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        criterion: nn.Module,
        metric: Metric,
        cfg: TrainerConfig,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.loaders = loaders
        self.max_epochs = cfg.max_epochs
        num_steps = len(loaders.train) * cfg.max_epochs
        self.progress_bar = tqdm(total=num_steps, desc="Epoch")
        self.cfg = cfg
        self.train_loss = float("inf")
        self.val_loss = float("inf")
        self.val_classification_loss = float("inf")
        self.best_val_loss = float("inf")
        self.best_monitor_value = float("inf")
        self.best_epoch = 0
        self.epochs_trained = 0
        self.epochs_without_improvement = 0
        self.value = 0.0
        self.metric = metric

    def save_training_state(
        self,
        next_epoch: int,
        early_stopping_triggered: bool,
    ) -> None:
        path = self.cfg.resume_checkpoint
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        state = {
            "next_epoch": next_epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_monitor_value": self.best_monitor_value,
            "best_epoch": self.best_epoch,
            "epochs_trained": self.epochs_trained,
            "epochs_without_improvement": self.epochs_without_improvement,
            "early_stopping_triggered": early_stopping_triggered,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
        }
        torch.save(state, temporary_path)
        os.replace(temporary_path, path)
        self.progress_bar.write(
            f"Saved resumable training state after epoch {next_epoch}: {path}"
        )

    def load_training_state(self) -> int:
        path = self.cfg.resume_checkpoint
        if path is None or not path.is_file():
            return 0
        state = torch.load(
            path,
            map_location=self.cfg.device,
            weights_only=False,
        )
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.best_val_loss = float(state["best_val_loss"])
        self.best_monitor_value = float(state["best_monitor_value"])
        self.best_epoch = int(state["best_epoch"])
        self.epochs_trained = int(state["epochs_trained"])
        self.epochs_without_improvement = int(
            state["epochs_without_improvement"]
        )
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        torch.set_rng_state(state["torch_rng_state"].cpu())
        cuda_rng_state_all = state["cuda_rng_state_all"]
        if torch.cuda.is_available() and cuda_rng_state_all is not None:
            torch.cuda.set_rng_state_all(
                [rng_state.cpu() for rng_state in cuda_rng_state_all]
            )
        queries = self.model.queries.detach().cpu().numpy()
        self.loaders.train.dataset.queries = queries  # type: ignore
        next_epoch = int(state["next_epoch"])
        if bool(state.get("early_stopping_triggered", False)):
            self.progress_bar.write(
                f"Training had already early-stopped at epoch {next_epoch}; "
                "restoring the best checkpoint without further epochs"
            )
            return self.max_epochs
        self.progress_bar.write(
            f"Resuming training at epoch {next_epoch + 1}/{self.max_epochs} "
            f"from {path}"
        )
        return next_epoch

    def update_progress_bar(self) -> None:
        postfix = f"Train loss: {self.train_loss:.3f}"
        self.progress_bar.set_postfix_str(postfix)
        self.progress_bar.update()

    def train_step(self, batch: tuple[Tensor, ...]) -> dict[str, float]:
        inputs, labels, padding_mask, query_ids, teacher_scores = batch
        self.optimizer.zero_grad()
        inputs = inputs.to(self.cfg.device)
        labels = labels.to(self.cfg.device)
        padding_mask = padding_mask.to(self.cfg.device)
        query_ids = query_ids.to(self.cfg.device)
        teacher_scores = teacher_scores.to(self.cfg.device)
        outputs = self.model(inputs, padding_mask)
        losses = self.criterion.components(
            outputs,
            labels,
            self.model.queries,
            vectors=inputs,
            query_ids=query_ids,
            teacher_scores=teacher_scores,
            attention_mask=padding_mask,
        )
        loss = losses["total"]
        loss.backward()
        clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.gradient_clip)
        self.optimizer.step()
        queries = self.model.queries.detach().cpu().numpy()
        self.loaders.train.dataset.queries = queries  # type: ignore
        return {name: value.detach().item() for name, value in losses.items()}

    def train_epoch(self, epoch: int) -> bool:
        self.model.train()
        totals: dict[str, float] = {}
        for step, batch in enumerate(self.loaders.train, start=1):
            step_losses = self.train_step(batch)
            for name, value in step_losses.items():
                totals[name] = totals.get(name, 0.0) + value
            self.train_loss = totals["total"] / step
            self.update_progress_bar()
        self.validate()
        self.scheduler.step()
        self.epochs_trained = epoch + 1
        monitor_value = (
            self.val_classification_loss
            if self.cfg.early_stopping_metric == "classification_loss"
            else self.val_loss
        )
        improved = (
            monitor_value
            < self.best_monitor_value - self.cfg.early_stopping_min_delta
        )
        if improved:
            self.best_monitor_value = monitor_value
            self.best_val_loss = self.val_loss
            self.best_epoch = epoch + 1
            self.epochs_without_improvement = 0
            self.cfg.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), self.cfg.checkpoint)
            self.progress_bar.write(
                f"Saved best checkpoint at epoch {self.best_epoch}: "
                f"{self.cfg.checkpoint}"
            )
        else:
            self.epochs_without_improvement += 1
        self.progress_bar.write(
            f"Epoch {epoch + 1}/{self.max_epochs}: "
            f"train_loss={self.train_loss:.6f}, "
            f"train_ce={totals['classification'] / step:.6f}, "
            f"train_query_penalty={totals['query_penalty'] / step:.6f}, "
            f"train_teacher={totals['teacher_ranking'] / step:.6f}, "
            f"train_weighted_teacher={totals['weighted_teacher'] / step:.6f}, "
            f"val_loss={self.val_loss:.6f}, "
            f"val_ce={self.val_classification_loss:.6f}, "
            f"val_acc={self.value:.6f}"
        )
        patience = self.cfg.early_stopping_patience
        should_stop = (
            patience is not None
            and epoch + 1 >= self.cfg.minimum_epochs
            and self.epochs_without_improvement >= patience
        )
        if should_stop:
            self.progress_bar.write(
                "Early stopping at epoch "
                f"{epoch + 1}: no improvement in "
                f"{self.cfg.early_stopping_metric} for "
                f"{self.epochs_without_improvement} epochs"
            )
        return should_stop

    def train(self) -> None:
        self.model.to(self.cfg.device)
        start_epoch = self.load_training_state()
        for epoch in range(start_epoch, self.max_epochs):
            should_stop = self.train_epoch(epoch)
            self.save_training_state(
                next_epoch=epoch + 1,
                early_stopping_triggered=should_stop,
            )
            if should_stop:
                break
        self.progress_bar.close()
        best_weights = torch.load(
            self.cfg.checkpoint,
            map_location=self.cfg.device,
            weights_only=True,
        )
        self.model.load_state_dict(best_weights)
        queries = self.model.queries.detach().cpu().numpy()
        self.loaders.test.dataset.queries = queries  # type: ignore

    @torch.no_grad()
    def validate(self) -> None:
        self.model.eval()
        queries = self.model.queries.detach().cpu().numpy()
        self.loaders.validation.dataset.queries = queries  # type: ignore
        total_loss = 0.0
        classification_loss = 0.0
        for inputs, labels, padding_mask, _, _ in self.loaders.validation:
            inputs = inputs.to(self.cfg.device)
            labels = labels.to(self.cfg.device)
            padding_mask = padding_mask.to(self.cfg.device)
            outputs = self.model(inputs, padding_mask)
            components = self.criterion.components(
                outputs,
                labels,
                self.model.queries,
            )
            total_loss += components["total"].item()
            classification_loss += components["classification"].item()
            preds = outputs.argmax(dim=1)
            self.metric.update(preds, labels)
        num_batches = len(self.loaders.validation)
        self.val_loss = total_loss / num_batches
        self.val_classification_loss = classification_loss / num_batches
        self.value = self.metric.compute().item()
        self.metric.reset()


@torch.inference_mode()
def predict(
    model: nn.Module, dataloader: DataLoader, device: str | torch.device
) -> list[Tensor]:
    model.eval()
    predictions = []
    for inputs, labels, padding_mask, _, _ in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        padding_mask = padding_mask.to(device)
        outputs = model(inputs, padding_mask)
        predictions.append(outputs)
    return predictions


def make_trainer(cfg: Config, tbl: LanceTable) -> Trainer:
    loaders = make_loaders(cfg=cfg, tbl=tbl)
    ini_queries = loaders.train.dataset.queries  # type: ignore
    model = make_model(ini_queries=ini_queries, cfg=cfg.model)
    model.to(cfg.trainer.device)
    optimizer = AdamW(model.parameters(), **cfg.optimizer.model_dump(), fused=True)
    criterion = QueryLoss(
        temperature=cfg.trainer.temperature,
        ignore_index=cfg.trainer.ignore_index,
        teacher_weight=(
            cfg.teacher_loss.weight if cfg.teacher_loss.enabled else 0.0
        ),
        teacher_pair_temperature=cfg.teacher_loss.pair_temperature,
        teacher_min_score_gap=cfg.teacher_loss.min_score_gap,
    )
    scheduler = ExponentialLR(optimizer, gamma=0.9, last_epoch=-1)
    metric = Accuracy(
        task=cfg.task,
        num_classes=cfg.model.output_dim,
        ignore_index=cfg.trainer.ignore_index,
    )
    metric = metric.to(cfg.trainer.device)
    return Trainer(
        loaders=loaders,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        metric=metric,
        cfg=cfg.trainer,
    )


def set_hyperparams(
    cfg: Config,
    max_epochs: int,
    lr: float,
    weight_decay: float,
    temperature: float,
    hidden_dim: int,
    dropout: float,
) -> Config:
    cfg.trainer.max_epochs = max_epochs
    cfg.optimizer.lr = lr
    cfg.optimizer.weight_decay = weight_decay
    cfg.model.hidden_dim = hidden_dim
    cfg.model.dropout = dropout
    cfg.trainer.temperature = temperature
    return cfg


def train_model(cfg: Config, tbl: LanceTable, use_best: bool = False):
    if use_best:
        hyperparameters = json.load(open(cfg.tuner.path, "r"))
        cfg = set_hyperparams(cfg=cfg, **hyperparameters)
    trainer = make_trainer(cfg=cfg, tbl=tbl)
    trainer.train()
    return trainer

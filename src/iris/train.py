import json

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
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.value = 0.0
        self.metric = metric

    def update_progress_bar(self) -> None:
        postfix = f"Train loss: {self.train_loss:.3f}"
        self.progress_bar.set_postfix_str(postfix)
        self.progress_bar.update()

    def train_step(self, batch: tuple[Tensor, Tensor, Tensor]) -> float:
        inputs, labels, padding_mask = batch
        self.optimizer.zero_grad()
        inputs = inputs.to(self.cfg.device)
        labels = labels.to(self.cfg.device)
        padding_mask = padding_mask.to(self.cfg.device)
        outputs = self.model(inputs, padding_mask)
        loss = self.criterion(outputs, labels, self.model.queries)
        loss.backward()
        clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.gradient_clip)
        self.optimizer.step()
        queries = self.model.queries.detach().cpu().numpy()
        self.loaders.train.dataset.queries = queries  # type: ignore
        return loss.detach().item()

    def train_epoch(self, epoch: int) -> None:
        self.model.train()
        loss = 0.0
        for step, batch in enumerate(self.loaders.train, start=1):
            loss += self.train_step(batch)
            self.train_loss = loss / step
            self.update_progress_bar()
        self.validate()
        self.scheduler.step()
        if self.val_loss < self.best_val_loss:
            self.best_val_loss = self.val_loss
            self.best_epoch = epoch + 1
            self.cfg.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), self.cfg.checkpoint)
            self.progress_bar.write(
                f"Saved best checkpoint at epoch {self.best_epoch}: "
                f"{self.cfg.checkpoint}"
            )
        self.progress_bar.write(
            f"Epoch {epoch + 1}/{self.max_epochs}: "
            f"train_loss={self.train_loss:.6f}, "
            f"val_loss={self.val_loss:.6f}, "
            f"val_acc={self.value:.6f}"
        )

    def train(self) -> None:
        self.model.to(self.cfg.device)
        for epoch in range(self.max_epochs):
            self.train_epoch(epoch)
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
        loss = 0.0
        for inputs, labels, padding_mask in self.loaders.validation:
            inputs = inputs.to(self.cfg.device)
            labels = labels.to(self.cfg.device)
            padding_mask = padding_mask.to(self.cfg.device)
            outputs = self.model(inputs, padding_mask)
            loss += self.criterion(outputs, labels, self.model.queries).item()
            preds = outputs.argmax(dim=1)
            self.metric.update(preds, labels)
        loss /= len(self.loaders.validation)
        self.val_loss = loss
        self.value = self.metric.compute().item()
        self.metric.reset()


@torch.inference_mode()
def predict(
    model: nn.Module, dataloader: DataLoader, device: str | torch.device
) -> list[Tensor]:
    model.eval()
    predictions = []
    for inputs, labels, padding_mask in dataloader:
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
        temperature=cfg.trainer.temperature, ignore_index=cfg.trainer.ignore_index
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

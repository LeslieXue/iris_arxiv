import torch
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, Metric
from torchmetrics.wrappers import BootStrapper

from iris.config import Config
from iris.model import IRIS
from iris.tune import load_best_checkpoint


def bootstrap_metric(
    metric, outputs: torch.Tensor, labels: torch.Tensor, n_bootstraps: int
):
    bootstrap = BootStrapper(
        metric, num_bootstraps=n_bootstraps, mean=True, std=True, raw=True
    )
    bootstrap.update(outputs, labels)
    return bootstrap.compute()


def bootstrap_metrics(cfg: Config, metrics: dict[str, Metric], n_bootstraps: int):
    bootstrapped = {}
    for name, metric in metrics.items():
        bootstrapper = BootStrapper(
            metric, num_bootstraps=n_bootstraps, mean=True, std=True, raw=True
        )
        bootstrapper.to(cfg.trainer.device)
        bootstrapped[name] = bootstrapper
    return bootstrapped


@torch.no_grad()
def evaluate_model(cfg: Config, dataloader: DataLoader) -> dict:
    metrics = {
        "accuracy": Accuracy(
            task=cfg.task,
            num_classes=cfg.model.output_dim,
            ignore_index=cfg.trainer.ignore_index,
        )
    }
    model = load_best_checkpoint(cfg=cfg, model_class=IRIS)
    results = {}
    metrics = bootstrap_metrics(cfg, metrics, cfg.evaluator.n_bootstraps)
    for name, metric in metrics.items():
        for inputs, labels, padding_mask, _, _ in dataloader:
            inputs = inputs.to(cfg.trainer.device)
            labels = labels.to(cfg.trainer.device)
            padding_mask = padding_mask.to(cfg.trainer.device)
            outputs = model(inputs, padding_mask)
            pred = outputs.argmax(dim=1)
            metric.update(pred, labels)
        results[name] = metric.compute()
        metric.reset()
    return {
        "mean": results["accuracy"]["mean"].item(),
        "std": results["accuracy"]["std"].item(),
    }

import os
from typing import cast
import argparse
import lancedb
import polars as pl
import torch
from transformers import AutoConfig

from iris.config import get_config
from iris.data import make_dataset
from iris.database import make_database
from iris.dataset import get_num_classes, make_loaders
from iris.evaluate import evaluate_model
from iris.train import train_model
from iris.tune import tune

# TODO pip install --upgrade transformers accelerate bitsandbytes
# TODO torchrun --nproc-per-node 4 -m iris

def parse_args():
    parser = argparse.ArgumentParser(description="Run IRIS")
    parser.add_argument(
        "--config",
        default="config.toml",
        help="Path to the TOML configuration file",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = get_config(path=args.config)
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")

    if cfg.distributed:
        rank = int(os.environ["RANK"])
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        torch.distributed.init_process_group("nccl", device_id=device)

    cfg.model.d_model = AutoConfig.from_pretrained(cfg.model_id).hidden_size

    in_file = "arxiv11/data"
    tbl_name = "arxiv11"

    if cfg.make_dataset:
        make_dataset(in_file=in_file, cfg=cfg)

    if not cfg.database.path.exists() or cfg.make_database:
        make_database(tbl_name=tbl_name, cfg=cfg)

    db = lancedb.connect(cfg.database.path)
    tbl = db.open_table(tbl_name)
    tbl = cast(lancedb.table.LanceTable, tbl)

    cfg.model.output_dim = get_num_classes(tbl=tbl)
    cfg.task = "binary" if cfg.model.output_dim == 2 else "multiclass"

    if cfg.tune:
        tune(cfg=cfg, tbl=tbl)
    if cfg.train:
        train_model(cfg=cfg, tbl=tbl, use_best=False)
    if cfg.evaluate:
        loaders = make_loaders(cfg=cfg, tbl=tbl)
        data = evaluate_model(cfg=cfg, dataloader=loaders.test)
        df = pl.DataFrame(data)
        df.write_parquet(cfg.evaluator.path)


if __name__ == "__main__":
    main()

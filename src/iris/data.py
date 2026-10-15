from functools import partial
from pathlib import Path
from typing import Callable, cast

import polars as pl
import polars.selectors as cs
from datasets import Dataset
from transformers import AutoTokenizer

from iris.config import Config, DatasetConfig


def make_arxiv11_dataset(path: Path) -> pl.DataFrame:
    files = sorted(path.rglob("*.txt"))

    rows = [
        {
            "text": file_path.read_text(encoding="utf-8"),
            "label": file_path.parent.name,
            "source_path": file_path.relative_to(path).as_posix(),
        }
        for file_path in files
    ]

    df = pl.LazyFrame(rows)
    include = ["cs.AI", "cs.DS", "cs.PL"]

    df = (
        df.filter(pl.col("label").is_in(include))
        .with_columns((pl.col("label").rank("dense") - 1).cast(pl.Int32))
        .with_row_index(name="id")
        .with_columns(cs.integer().cast(pl.Int32))
        .collect()
    )

    return df


def make_hyperpartisan_dataset(path: Path):
    df = pl.read_csv(path)
    df = df.with_columns(label=pl.col("Hyperpartisan").cast(pl.Int32))
    df = df.rename({"Article ID": "id", "Content": "text"})
    return df


def make_imdb_dataset(path: Path):
    df = (
        pl.read_csv(path)
        .with_row_index("id")
        .rename({"review": "text", "sentiment": "label"})
        .with_columns(pl.col("label").replace({"positive": 1, "negative": 0}))
    )
    return df


def get_df_fn(name: str) -> Callable[[Path], pl.DataFrame]:
    match name:
        case "arxiv11/data":
            return make_arxiv11_dataset
        case "hyperpartisan.csv":
            return make_hyperpartisan_dataset
        case "IMDB_Dataset.csv":
            return make_imdb_dataset
        case _:
            raise ValueError(f"Unknown dataset: {name}")


def make_chunks(
    input_ids: list[int],
    attention_mask: list[int],
    chunk_length: int,
    overlap: int,
    pad_id: int,
):
    chunks = []
    masks = []
    offsets = []
    num_sequences = len(input_ids)
    step_size = chunk_length - overlap
    for i in range(0, num_sequences, step_size):
        chunk = input_ids[i : i + chunk_length]
        mask = attention_mask[i : i + chunk_length]
        if i + chunk_length > num_sequences:
            padding_length = chunk_length - len(chunk)
            chunk = chunk + [pad_id] * padding_length
            mask = mask + [0] * padding_length
        chunks.append(chunk)
        masks.append(mask)
        offsets.append(i)
    return chunks, masks, offsets


def chunk_batch(batch, pad_id: int, cfg: DatasetConfig) -> dict:
    result = {
        "input_ids": [],
        "attention_mask": [],
        "id": [],
        "label": [],
        "chunk_id": [],
        "offset": [],
    }
    for idx, (input_ids, attention_mask) in enumerate(
        zip(batch["input_ids"], batch["attention_mask"])
    ):
        chunks, masks, offsets = make_chunks(
            input_ids,
            attention_mask,
            chunk_length=cfg.chunk_length,
            overlap=cfg.overlap,
            pad_id=pad_id,
        )
        for chunk_idx, (chunk, mask, offset) in enumerate(zip(chunks, masks, offsets)):
            result["input_ids"].append(chunk)
            result["attention_mask"].append(mask)
            result["id"].append(batch["id"][idx])
            result["label"].append(batch["label"][idx])
            result["chunk_id"].append(chunk_idx)
            result["offset"].append(offset)
    return result


def tokenize(batch, tokenizer) -> AutoTokenizer:
    return tokenizer(batch["text"])


def make_dataset(in_file: str, cfg: Config) -> Dataset:
    make_df = get_df_fn(name=in_file)
    df = make_df(cfg.dataset.in_path / in_file).select("id", "label", "text")
    if cfg.fast_dev_run:
        df = df.head(10)
    dataset = Dataset.from_polars(df)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    tokenize_fn = partial(tokenize, tokenizer=tokenizer)
    dataset = dataset.map(tokenize_fn, batched=True, desc="Tokenizing")
    pad_id = cast(int, tokenizer.pad_token_id)
    chunk_fn = partial(chunk_batch, pad_id=pad_id, cfg=cfg.dataset)
    dataset = dataset.map(
        chunk_fn, batched=True, desc="Chunking", remove_columns=dataset.column_names
    )
    dataset.save_to_disk(cfg.dataset.out_path)
    return dataset

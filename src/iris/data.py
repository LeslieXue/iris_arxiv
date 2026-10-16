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
        "chunk_text": [],
    }

    has_source_path = "source_path" in batch
    if has_source_path:
        result["source_path"] = []

    rows = zip(
        batch["text"],
        batch["input_ids"],
        batch["attention_mask"],
        batch["offset_mapping"],
    )

    for idx, (text, input_ids, attention_mask, token_offsets) in enumerate(rows):
        chunks, masks, offsets = make_chunks(
            input_ids,
            attention_mask,
            chunk_length=cfg.chunk_length,
            overlap=cfg.overlap,
            pad_id=pad_id,
        )

        for chunk_idx, (chunk, mask, offset) in enumerate(
            zip(chunks, masks, offsets)
        ):
            chunk_offsets = token_offsets[offset : offset + cfg.chunk_length]

            valid_offsets = [
                (char_start, char_end)
                for char_start, char_end in chunk_offsets
                if char_end > char_start
            ]

            # Special tokens such as BOS/EOS have offsets (0, 0).
            # Do not create a vector chunk if it contains no source text.
            if not valid_offsets:
                continue

            char_start = valid_offsets[0][0]
            char_end = valid_offsets[-1][1]
            chunk_text = text[char_start:char_end]

            result["input_ids"].append(chunk)
            result["attention_mask"].append(mask)
            result["id"].append(batch["id"][idx])
            result["label"].append(batch["label"][idx])
            result["chunk_id"].append(chunk_idx)
            result["offset"].append(offset)
            result["chunk_text"].append(chunk_text)

            if has_source_path:
                result["source_path"].append(batch["source_path"][idx])

    return result


def tokenize(batch, tokenizer):
    return tokenizer(
        batch["text"],
        return_offsets_mapping=True,
    )


def make_dataset(in_file: str, cfg: Config) -> Dataset:
    make_df = get_df_fn(name=in_file)
    df = make_df(cfg.dataset.in_path / in_file)

    columns = ["id", "label", "text"]
    if "source_path" in df.columns:
        columns.append("source_path")

    df = df.select(columns)
    if cfg.fast_dev_run:
        # Keep the smoke dataset small while retaining examples from every
        # class. Taking the first rows only can select a single arXiv category
        # because source files are sorted by directory name.
        df = df.group_by("label", maintain_order=True).head(10).sort("id")
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

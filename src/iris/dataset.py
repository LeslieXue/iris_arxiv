from dataclasses import dataclass
from functools import partial

import duckdb
import numpy as np
import polars as pl
import torch
from lancedb.table import LanceTable
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from iris.config import Config
from iris.sampling import subsample_train_ids, train_val_test_split


def load_teacher_scores(
    cfg: Config,
    tbl: LanceTable,
    required_document_ids: np.ndarray,
) -> dict[tuple[int, int], float] | None:
    teacher_cfg = cfg.teacher_loss

    if not teacher_cfg.enabled:
        return None
    if teacher_cfg.scores_path is None:
        raise ValueError(
            "teacher_loss.scores_path is required when teacher loss is enabled"
        )
    if not teacher_cfg.scores_path.is_file():
        raise FileNotFoundError(
            f"Teacher score table not found: {teacher_cfg.scores_path}"
        )

    required_ids = pl.Series(
        "document_id",
        required_document_ids,
        dtype=pl.Int64,
    )
    scores = (
        pl.read_parquet(teacher_cfg.scores_path)
        .select("document_id", "chunk_id", "teacher_utility")
        .filter(pl.col("document_id").is_in(required_ids))
        .sort("document_id", "chunk_id")
    )
    duplicate_count = scores.select(
        pl.struct(["document_id", "chunk_id"]).is_duplicated().sum()
    ).item()
    if duplicate_count:
        raise ValueError(
            f"Teacher score table has {duplicate_count} duplicate chunk keys"
        )
    if scores["teacher_utility"].is_null().any():
        raise ValueError("Teacher score table contains null teacher utilities")

    database_keys = pl.from_arrow(
        tbl.to_lance()
        .scanner(columns=["id", "chunk_id"])
        .to_table()
    ).rename({"id": "document_id"}).filter(
        pl.col("document_id").is_in(required_ids)
    )
    missing = database_keys.join(
        scores.select("document_id", "chunk_id"),
        on=["document_id", "chunk_id"],
        how="anti",
    )
    extra = scores.join(
        database_keys,
        on=["document_id", "chunk_id"],
        how="anti",
    )
    if missing.height or extra.height:
        raise ValueError(
            "Teacher score coverage does not match the vector database: "
            f"missing={missing.height}, extra={extra.height}"
        )

    utilities = scores["teacher_utility"].to_numpy().astype(np.float32)
    if teacher_cfg.score_mode == "shuffled":
        rng = np.random.default_rng(teacher_cfg.shuffle_seed)
        document_ids = scores["document_id"].to_numpy()
        boundaries = np.flatnonzero(document_ids[1:] != document_ids[:-1]) + 1
        starts = np.concatenate(([0], boundaries))
        stops = np.concatenate((boundaries, [len(document_ids)]))
        for start, stop in zip(starts, stops, strict=True):
            utilities[start:stop] = rng.permutation(utilities[start:stop])

    keys = zip(
        scores["document_id"].to_list(),
        scores["chunk_id"].to_list(),
        strict=True,
    )
    mapping = {
        (int(document_id), int(chunk_id)): float(utility)
        for (document_id, chunk_id), utility in zip(
            keys,
            utilities,
            strict=True,
        )
    }
    print(
        "Loaded teacher scores: "
        f"records={len(mapping)}, mode={teacher_cfg.score_mode}, "
        f"path={teacher_cfg.scores_path}"
    )
    return mapping


def get_centroids(tbl: LanceTable) -> np.ndarray:
    lance_tbl = tbl.to_lance()
    vector_index = lance_tbl.index_statistics("vector_idx")
    centroids = vector_index["indices"][0]["centroids"]
    return np.array(centroids)


def get_ini_queries(tbl: LanceTable, cfg: Config) -> np.ndarray:
    if cfg.model.query_ini_random:
        return np.random.randn(cfg.model.num_queries, cfg.model.d_model)
    else:
        centroids = get_centroids(tbl=tbl)
        # down sample centroids to number of queries
        indices = np.random.choice(
            np.arange(cfg.database.num_partitions),
            replace=False,
            size=cfg.model.num_queries,
        )
        return centroids[indices]


def get_num_classes(tbl: LanceTable):
    lance_tbl = tbl.to_lance()
    duckdb.register("lance_tbl", lance_tbl)
    num_labels = duckdb.sql("SELECT COUNT(DISTINCT label) FROM lance_tbl").fetchone()
    return num_labels[0] if num_labels else 0


class LanceTableDataset(Dataset):
    def __init__(
        self,
        ids: np.ndarray,
        tbl: LanceTable,
        vector_dim: int,
        k_neighbors: int,
        ignore_index: int,
        ini_queries: np.ndarray,
        teacher_scores: dict[tuple[int, int], float] | None = None,
    ):
        self.ids = ids
        self.tbl = tbl
        self.vector_dim = vector_dim
        self.k_neighbors = k_neighbors
        self.ignore_index = ignore_index
        self.queries = ini_queries
        self.teacher_scores = teacher_scores

    def __len__(self):
        return self.ids.shape[0]

    def search_chunks(
        self,
        document_id: int,
        include_text: bool = False,
    ) -> pl.DataFrame:
        dfs: list[pl.DataFrame] = []
        chunk_ids = []

        available_columns = set(self.tbl.schema.names)

        columns = [
            "id",
            "chunk_id",
            "offset",
            "label",
            "vector",
            "_distance",
        ]

        if include_text:
            required_columns = {"source_path", "chunk_text"}
            missing_columns = required_columns - available_columns

            if missing_columns:
                missing = ", ".join(sorted(missing_columns))
                raise ValueError(
                    f"The vector table is missing: {missing}. "
                    "Rebuild the processed dataset and database."
                )

            columns.extend(["source_path", "chunk_text"])

        for query_id, query in enumerate(self.queries):
            condition = f"(id = {document_id})"

            if chunk_ids:
                excluded_chunks = ", ".join(chunk_ids)
                condition += f" AND (chunk_id NOT IN ({excluded_chunks}))"

            search = (
                self.tbl.search(query=query)
                .where(condition, prefilter=True)
                .select(columns)
                .limit(self.k_neighbors)
            )
            df = search.to_polars()

            # A global IVF search can occasionally probe no partition that
            # contains chunks for the requested document. Fall back to an
            # exact search over the prefiltered document in that case.
            if df.height == 0:
                df = (
                    self.tbl.search(query=query)
                    .where(condition, prefilter=True)
                    .bypass_vector_index()
                    .select(columns)
                    .limit(self.k_neighbors)
                    .to_polars()
                )

            df = df.with_row_index("rank", offset=1).with_columns(
                pl.lit(query_id).alias("query_id")
            )

            chunk_ids.extend(df["chunk_id"].cast(pl.String).to_list())
            dfs.append(df)

        return pl.concat(dfs)

    def __getitem__(self, idx):
        document_id = self.ids[idx]
        df = self.search_chunks(document_id)

        if df.height == 0:
            vectors = torch.randn(1, self.vector_dim)
            label = self.ignore_index
            query_ids = torch.full((1,), -1, dtype=torch.long)
            teacher_scores = torch.full((1,), float("nan"))
        else:
            vectors = df["vector"].to_torch()
            label = df["label"][0]
            query_ids = df["query_id"].to_torch().long()
            if self.teacher_scores is None:
                teacher_scores = torch.full((df.height,), float("nan"))
            else:
                teacher_scores = torch.tensor(
                    [
                        self.teacher_scores[(int(document_id), int(chunk_id))]
                        for chunk_id in df["chunk_id"].to_list()
                    ],
                    dtype=torch.float32,
                )

        return vectors, torch.tensor(label), query_ids, teacher_scores


def collate_fn(batch):
    vectors, labels, query_ids, teacher_scores = zip(*batch)
    vectors = pad_sequence(vectors, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)
    attention_mask = (vectors != 0).all(dim=-1)
    query_ids = pad_sequence(query_ids, batch_first=True, padding_value=-1)
    teacher_scores = pad_sequence(
        teacher_scores,
        batch_first=True,
        padding_value=float("nan"),
    )
    return vectors, labels, attention_mask, query_ids, teacher_scores


@dataclass
class DataLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader


def make_loaders(
    cfg: Config,
    tbl: LanceTable,
    include_teacher_scores: bool = True,
):
    lance_tbl = tbl.to_lance()
    duckdb.register("lance_tbl", lance_tbl)
    df = duckdb.sql(
        "SELECT DISTINCT id, label FROM lance_tbl ORDER BY id"
    ).to_df()
    ids = df["id"].to_numpy()
    labels = df["label"].to_numpy()
    train, val, test = train_val_test_split(ids, labels=labels, seed=cfg.seed)
    label_by_id = {
        int(document_id): int(label)
        for document_id, label in zip(ids, labels, strict=True)
    }
    train_labels = np.array(
        [label_by_id[int(document_id)] for document_id in train]
    )
    train = subsample_train_ids(
        train_ids=train,
        train_labels=train_labels,
        fraction=cfg.trainer.train_fraction,
        seed=cfg.seed,
    )
    ini_queries = get_ini_queries(tbl=tbl, cfg=cfg)
    teacher_scores = (
        load_teacher_scores(
            cfg=cfg,
            tbl=tbl,
            required_document_ids=train,
        )
        if include_teacher_scores
        else None
    )
    dataset = partial(
        LanceTableDataset,
        tbl=tbl,
        vector_dim=cfg.model.d_model,
        k_neighbors=cfg.database.k_neighbors,
        ignore_index=cfg.trainer.ignore_index,
        ini_queries=ini_queries,
    )
    train = dataset(ids=train, teacher_scores=teacher_scores)
    val = dataset(ids=val)
    test = dataset(ids=test)
    loader = partial(DataLoader, **cfg.dataloader.model_dump(), collate_fn=collate_fn)
    train_loader = loader(train, shuffle=True, drop_last=True)
    val_loader = loader(val)
    test_loader = loader(test)
    return DataLoaders(train=train_loader, validation=val_loader, test=test_loader)

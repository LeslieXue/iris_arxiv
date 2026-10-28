import argparse
import json
from collections import Counter
from pathlib import Path

import duckdb
import lancedb
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from transformers import AutoConfig

from iris.config import get_config
from iris.dataset import get_num_classes, make_loaders
from iris.loss import query_similarity_penalty
from iris.model import make_model

LABEL_NAMES = {
    0: "cs.AI",
    1: "cs.DS",
    2: "cs.PL",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained IRIS model")
    parser.add_argument("--config", required=True, help="Training TOML configuration")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint override",
    )
    parser.add_argument("--output", required=True, help="Metrics JSON output path")
    return parser.parse_args()


def split_distribution(ids, label_by_id):
    counts = Counter(label_by_id[int(document_id)] for document_id in ids)
    return {
        LABEL_NAMES.get(int(label), str(label)): int(count)
        for label, count in sorted(counts.items())
    }


@torch.inference_mode()
def main():
    args = parse_args()
    cfg = get_config(args.config)
    if args.checkpoint is not None:
        cfg.trainer.checkpoint = args.checkpoint

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    db = lancedb.connect(cfg.database.path)
    table = db.open_table("arxiv11")
    cfg.model.d_model = AutoConfig.from_pretrained(cfg.model_id).hidden_size
    cfg.model.output_dim = get_num_classes(table)
    cfg.task = "binary" if cfg.model.output_dim == 2 else "multiclass"

    loaders = make_loaders(
        cfg=cfg,
        tbl=table,
        include_teacher_scores=False,
    )
    checkpoint = torch.load(
        cfg.trainer.checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    trained_queries = checkpoint["queries"].numpy()
    loaders.test.dataset.queries = trained_queries  # type: ignore

    model = make_model(ini_queries=trained_queries, cfg=cfg.model)
    model.load_state_dict(checkpoint)
    model.to(cfg.trainer.device)
    model.eval()

    predictions = []
    probabilities = []
    targets = []
    ignored_document_ids = []
    cross_entropy_sum = 0.0
    example_count = 0
    test_document_ids = loaders.test.dataset.ids  # type: ignore
    document_offset = 0

    for inputs, labels, padding_mask, _, _ in loaders.test:
        batch_size = labels.numel()
        batch_document_ids = test_document_ids[
            document_offset : document_offset + batch_size
        ]
        document_offset += batch_size

        inputs = inputs.to(cfg.trainer.device)
        labels = labels.to(cfg.trainer.device)
        padding_mask = padding_mask.to(cfg.trainer.device)
        outputs = model(inputs, padding_mask)

        valid = labels != cfg.trainer.ignore_index
        ignored_document_ids.extend(
            int(document_id)
            for document_id in batch_document_ids[~valid.cpu().numpy()]
        )
        outputs = outputs[valid]
        labels = labels[valid]

        if labels.numel() == 0:
            continue

        batch_size = labels.numel()
        cross_entropy_sum += F.cross_entropy(
            outputs,
            labels,
            reduction="sum",
        ).item()
        example_count += batch_size
        predictions.extend(outputs.argmax(dim=1).cpu().tolist())
        probabilities.extend(F.softmax(outputs, dim=1).cpu().tolist())
        targets.extend(labels.cpu().tolist())

    cross_entropy = cross_entropy_sum / example_count
    query_penalty = query_similarity_penalty(
        model.queries,
        cfg.trainer.temperature,
    ).item()

    labels = list(range(cfg.model.output_dim))
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=labels,
        zero_division=0,
    )
    probability_array = np.asarray(probabilities, dtype=np.float64)
    target_array = np.asarray(targets)
    macro_ovr_roc_auc = roc_auc_score(
        target_array,
        probability_array,
        labels=labels,
        multi_class="ovr",
        average="macro",
    )
    per_class_roc_auc = {
        label: roc_auc_score(
            (target_array == label).astype(np.int64),
            probability_array[:, label],
        )
        for label in labels
    }

    lance_table = table.to_lance()
    duckdb.register("chunks", lance_table)
    label_rows = duckdb.sql(
        "SELECT DISTINCT id, label FROM chunks ORDER BY id"
    ).fetchall()
    label_by_id = {int(document_id): int(label) for document_id, label in label_rows}

    metrics = {
        "checkpoint": str(cfg.trainer.checkpoint),
        "test_documents": int(len(test_document_ids)),
        "test_examples": example_count,
        "ignored_documents": ignored_document_ids,
        "split_documents": {
            "train": split_distribution(loaders.train.dataset.ids, label_by_id),  # type: ignore
            "validation": split_distribution(
                loaders.validation.dataset.ids, label_by_id  # type: ignore
            ),
            "test": split_distribution(loaders.test.dataset.ids, label_by_id),  # type: ignore
        },
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(targets, predictions)
        ),
        "macro_f1": float(f1_score(targets, predictions, average="macro")),
        "macro_ovr_roc_auc": float(macro_ovr_roc_auc),
        "cross_entropy": cross_entropy,
        "query_penalty": query_penalty,
        "total_loss": cross_entropy + query_penalty,
        "confusion_matrix": confusion_matrix(
            targets,
            predictions,
            labels=labels,
        ).tolist(),
        "per_class": {
            LABEL_NAMES.get(label, str(label)): {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "roc_auc_ovr": float(per_class_roc_auc[label]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"Metrics written to: {output_path}")


if __name__ == "__main__":
    main()

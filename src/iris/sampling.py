"""Deterministic document splitting and training-set subsampling."""

import numpy as np
from sklearn.model_selection import train_test_split


def train_val_test_split(data: np.ndarray, labels: np.ndarray, seed: int):
    train, temp, _, temp_labels = train_test_split(
        data,
        labels,
        test_size=0.2,
        shuffle=True,
        random_state=seed,
        stratify=labels,
    )
    val, test = train_test_split(
        temp,
        test_size=0.5,
        random_state=seed,
        stratify=temp_labels,
    )
    return train, val, test


def subsample_train_ids(
    train_ids: np.ndarray,
    train_labels: np.ndarray,
    fraction: float,
    seed: int,
) -> np.ndarray:
    """Select a deterministic, label-stratified subset of training documents."""
    if not 0 < fraction <= 1:
        raise ValueError("Training fraction must be in (0, 1]")
    if fraction == 1:
        return train_ids

    subset, _ = train_test_split(
        train_ids,
        train_size=fraction,
        shuffle=True,
        random_state=seed,
        stratify=train_labels,
    )
    return np.sort(subset)

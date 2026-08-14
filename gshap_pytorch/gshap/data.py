from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import DataConfig
from .io_utils import atomic_json, atomic_numpy, ensure_dir, sha256_array


@dataclass(frozen=True)
class DatasetBundle:
    features: np.ndarray
    labels: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray
    lr_val_indices: np.ndarray


def _stratified_take(indices: np.ndarray, labels: np.ndarray, size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if size <= 0 or size >= len(indices):
        raise ValueError(f"Invalid split size {size} for {len(indices)} samples")
    local_labels = labels[indices]
    classes, counts = np.unique(local_labels, return_counts=True)
    if len(classes) != 2:
        raise ValueError("Binary stratified split requires exactly two classes")
    raw = counts / counts.sum() * size
    allocations = np.floor(raw).astype(int)
    for position in np.argsort(-(raw - allocations))[: size - allocations.sum()]:
        allocations[position] += 1
    selected_parts: list[np.ndarray] = []
    remaining_parts: list[np.ndarray] = []
    for label, amount in zip(classes, allocations, strict=True):
        members = indices[local_labels == label].copy()
        rng.shuffle(members)
        selected_parts.append(members[:amount])
        remaining_parts.append(members[amount:])
    selected = np.concatenate(selected_parts).astype(np.int64, copy=False)
    remaining = np.concatenate(remaining_parts).astype(np.int64, copy=False)
    rng.shuffle(selected)
    rng.shuffle(remaining)
    return selected, remaining


def create_split(labels: np.ndarray, test_size: int, lr_val_size: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels).reshape(-1)
    rng = np.random.default_rng(seed)
    available = np.arange(len(labels), dtype=np.int64)
    test, available = _stratified_take(available, labels, test_size, rng)
    lr_val, train = _stratified_take(available, labels, lr_val_size, rng)
    return np.sort(train), np.sort(test), np.sort(lr_val)


def load_or_create_bundle(data_dir: Path, split_dir: Path, config: DataConfig) -> DatasetBundle:
    features = np.load(data_dir / config.feature_file, mmap_mode="r")
    labels = np.load(data_dir / config.label_file, mmap_mode="r").reshape(-1)
    if features.ndim != 2 or features.shape[1] != 512 or len(features) != len(labels):
        raise ValueError(f"Unexpected data shapes: X={features.shape}, y={labels.shape}")
    if features.dtype != np.float32:
        raise ValueError(f"Features must be float32, got {features.dtype}")
    if not np.all(np.isin(np.unique(labels), (0, 1))):
        raise ValueError("Labels must be binary")

    train_path = split_dir / "train_indices.npy"
    test_path = split_dir / "test_indices.npy"
    lr_val_path = split_dir / "lr_val_indices.npy"
    if train_path.exists() and test_path.exists() and lr_val_path.exists():
        train = np.load(train_path)
        test = np.load(test_path)
        lr_val = np.load(lr_val_path)
        if len(test) != config.test_size or len(lr_val) != config.lr_val_size:
            raise ValueError(
                "Saved split sizes disagree with configuration; use a new results directory "
                "or remove only the explicit split files after verifying their paths"
            )
    else:
        train, test, lr_val = create_split(labels, config.test_size, config.lr_val_size, config.split_seed)
        ensure_dir(split_dir)
        atomic_numpy(train_path, train)
        atomic_numpy(test_path, test)
        atomic_numpy(lr_val_path, lr_val)
        atomic_json(split_dir / "split_meta.json", {
            "split_seed": config.split_seed,
            "n_total": len(labels), "n_train": len(train),
            "n_test": len(test), "n_lr_val": len(lr_val),
            "train_class_counts": np.bincount(labels[train].astype(int), minlength=2).tolist(),
            "test_class_counts": np.bincount(labels[test].astype(int), minlength=2).tolist(),
            "lr_val_class_counts": np.bincount(labels[lr_val].astype(int), minlength=2).tolist(),
            "train_sha256": sha256_array(train),
            "test_sha256": sha256_array(test),
            "lr_val_sha256": sha256_array(lr_val),
        })
    combined = np.concatenate((train, test, lr_val))
    if len(np.unique(combined)) != len(labels) or len(combined) != len(labels):
        raise ValueError("Saved split is not a disjoint partition of the dataset")
    if combined.min() < 0 or combined.max() >= len(labels):
        raise ValueError("Saved split contains out-of-range indices")
    return DatasetBundle(features, labels, train, test, lr_val)


def synthetic_dataset(path: Path, n: int = 5_000, dim: int = 512, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, dim)).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True).clip(1e-12)
    weights = rng.normal(size=dim)
    logits = features @ weights + 0.25 * rng.normal(size=n)
    labels = (logits > np.quantile(logits, 0.84)).astype(np.float32).reshape(-1, 1)
    ensure_dir(path)
    np.save(path / "X.npy", features)
    np.save(path / "y.npy", labels)

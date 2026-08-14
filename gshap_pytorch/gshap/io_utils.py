from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_array(array: np.ndarray) -> str:
    view = np.ascontiguousarray(array).view(np.uint8)
    return hashlib.sha256(view).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_numpy(path: Path, array: np.ndarray) -> None:
    ensure_dir(path.parent)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npy.tmp")
    os.close(fd)
    try:
        with open(temporary, "wb") as handle:
            np.save(handle, array)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_torch(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".pt.tmp")
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".csv.tmp")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


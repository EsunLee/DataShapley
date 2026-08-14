from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch


def generate_master_seeds(master_seed: int = 41, count: int = 100) -> list[int]:
    rng = np.random.RandomState(master_seed)
    return sorted(rng.randint(1_000, 999_999, size=count).tolist())


def derive_iteration_seeds(master_seed: int, count: int = 50) -> np.ndarray:
    return np.random.SeedSequence(master_seed).generate_state(count, dtype=np.uint32)


def calibration_seeds(seed: int = 17, count: int = 3) -> np.ndarray:
    return np.random.SeedSequence(seed).generate_state(count, dtype=np.uint32)


def set_seed(seed: int, deterministic: bool = True) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def load_used_seeds(path: Path) -> list[int]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    seeds = [int(value) for value in payload["seeds"]]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Group seeds must be unique")
    return seeds


from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DataConfig:
    feature_file: str = "ucf_train_feat_704459x512.npy"
    label_file: str = "ucf_train_label_704459x1.npy"
    test_size: int = 5_000
    lr_val_size: int = 5_000
    split_seed: int = 0
    lr_train_size: int = 100_000


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int = 512
    hidden_dim: int = 128


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 128
    iterations: int = 50
    learning_rate: float | None = None
    lr_candidates: tuple[float, ...] = (
        1e-1, 10**-1.5, 1e-2, 10**-2.5,
        1e-3, 10**-3.5, 1e-4, 10**-4.5,
    )
    lr_repeats: int = 3
    lr_tie_tolerance: float = 1e-4
    baseline_auc: float = 0.5
    efficiency_tolerance: float = 2e-6


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "auto"
    keep_permutations: bool = False
    keep_marginals: bool = False
    deterministic: bool = True
    checkpoint_every: int = 5


@dataclass(frozen=True)
class ExperimentConfig:
    data_dir: Path
    results_dir: Path
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["data_dir"] = str(self.data_dir)
        result["results_dir"] = str(self.results_dir)
        return result


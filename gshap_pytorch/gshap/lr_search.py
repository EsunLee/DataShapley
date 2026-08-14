from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import ExperimentConfig
from .data import DatasetBundle, _stratified_take
from .device import resolve_device
from .io_utils import atomic_json, write_csv
from .metrics import evaluate_auc
from .model import PlovadDecoder
from .seeds import calibration_seeds, set_seed


@dataclass(frozen=True)
class LearningRateResult:
    learning_rate: float
    confirmation_auc: float


def _to_device(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(array), device=device)


def _one_pass(
    features: torch.Tensor,
    labels: torch.Tensor,
    learning_rate: float,
    seed: int,
    config: ExperimentConfig,
) -> tuple[PlovadDecoder, float]:
    set_seed(seed, config.runtime.deterministic)
    model = PlovadDecoder(config.model.input_dim, config.model.hidden_dim).to(features.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.0, weight_decay=0.0)
    loss_function = nn.BCEWithLogitsLoss()
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutation = torch.randperm(len(features), generator=generator)
    last_loss = math.nan
    model.train()
    for start in range(0, len(permutation), config.train.batch_size):
        compact = permutation[start : start + config.train.batch_size].to(features.device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features[compact])
        loss = loss_function(logits, labels[compact])
        if not torch.isfinite(loss):
            return model, math.nan
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().item())
    return model, last_loss


def calibrate_learning_rate(
    bundle: DatasetBundle,
    config: ExperimentConfig,
    output_dir: Path,
    calibration_seed: int = 17,
    confirm: bool = True,
) -> LearningRateResult:
    device = resolve_device(config.runtime.device)
    rng = np.random.default_rng(calibration_seed)
    amount = min(config.data.lr_train_size, len(bundle.train_indices) - 1)
    lr_train_indices, _ = _stratified_take(bundle.train_indices, bundle.labels, amount, rng)
    train_x = _to_device(bundle.features[lr_train_indices], device).to(torch.float32)
    train_y = _to_device(bundle.labels[lr_train_indices], device).to(torch.float32)
    val_x = _to_device(bundle.features[bundle.lr_val_indices], device).to(torch.float32)
    val_y = _to_device(bundle.labels[bundle.lr_val_indices], device).to(torch.float32)
    seeds = calibration_seeds(calibration_seed, config.train.lr_repeats)

    rows: list[dict[str, float | int | bool]] = []
    summaries: list[dict[str, float | bool]] = []
    for learning_rate in config.train.lr_candidates:
        aucs: list[float] = []
        valid = True
        for repeat, seed in enumerate(seeds, start=1):
            model, final_loss = _one_pass(train_x, train_y, learning_rate, int(seed), config)
            auc = evaluate_auc(model, val_x, val_y) if math.isfinite(final_loss) else math.nan
            valid = valid and math.isfinite(auc) and math.isfinite(final_loss)
            aucs.append(auc)
            rows.append({"learning_rate": learning_rate, "repeat": repeat, "seed": int(seed),
                         "final_loss": final_loss, "final_auc": auc, "valid": valid})
            del model
        mean = float(np.mean(aucs)) if valid else math.nan
        std = float(np.std(aucs, ddof=0)) if valid else math.nan
        summaries.append({"learning_rate": learning_rate, "mean_auc": mean, "std_auc": std,
                          "score": mean - std if valid else -math.inf, "valid": valid})

    valid_summaries = [item for item in summaries if item["valid"]]
    if not valid_summaries:
        raise RuntimeError("All learning-rate candidates diverged")
    best_score = max(float(item["score"]) for item in valid_summaries)
    tied = [item for item in valid_summaries
            if best_score - float(item["score"]) < config.train.lr_tie_tolerance]
    selected = min(float(item["learning_rate"]) for item in tied)

    confirmation_auc = math.nan
    confirmation_loss = math.nan
    if confirm:
        full_x = _to_device(bundle.features[bundle.train_indices], device).to(torch.float32)
        full_y = _to_device(bundle.labels[bundle.train_indices], device).to(torch.float32)
        model, confirmation_loss = _one_pass(full_x, full_y, selected, int(seeds[0]), config)
        confirmation_auc = evaluate_auc(model, val_x, val_y) if math.isfinite(confirmation_loss) else math.nan
        if not math.isfinite(confirmation_auc) or confirmation_auc <= 0.5:
            raise RuntimeError(f"Selected LR failed full-data confirmation: AUC={confirmation_auc}")

    fieldnames = ["learning_rate", "repeat", "seed", "final_loss", "final_auc", "valid"]
    write_csv(output_dir / "lr_search.csv", rows, fieldnames)
    atomic_json(output_dir / "lr_search.json", {
        "candidates": summaries, "calibration_seeds": seeds.tolist(),
        "lr_train_size": amount, "selected_learning_rate": selected,
        "confirmation_loss": confirmation_loss, "confirmation_auc": confirmation_auc,
    })
    return LearningRateResult(selected, confirmation_auc)


def read_selected_learning_rate(path: Path) -> float:
    import json
    with path.open("r", encoding="utf-8") as handle:
        return float(json.load(handle)["selected_learning_rate"])


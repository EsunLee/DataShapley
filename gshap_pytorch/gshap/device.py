from __future__ import annotations

import torch


def resolve_device(specification: str) -> torch.device:
    if specification == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(specification)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {specification}")
    return device


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


from __future__ import annotations

import torch
from torch import nn


class PlovadDecoder(nn.Module):
    """PLOVAD classifier from official model.py lines 109-113."""

    def __init__(self, input_dim: int = 512, hidden_dim: int = 128) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv1d(hidden_dim, 1, kernel_size=1, padding=0),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 2:
            features = features.unsqueeze(-1)
        if features.ndim != 3:
            raise ValueError(f"Expected [N, D] or [N, D, T], got {features.shape}")
        return self.classifier(features).squeeze(1).squeeze(-1)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


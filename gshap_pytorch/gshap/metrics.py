from __future__ import annotations

import torch


@torch.inference_mode()
def binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Exact ROC AUC on the input device, including average ranks for ties."""
    scores = scores.reshape(-1).to(torch.float64)
    labels = labels.reshape(-1).to(torch.bool)
    if scores.numel() != labels.numel():
        raise ValueError("scores and labels must have the same length")
    positives = labels.sum()
    negatives = labels.numel() - positives
    if int(positives) == 0 or int(negatives) == 0:
        raise ValueError("ROC AUC requires both classes")

    sorted_scores, order = torch.sort(scores)
    sorted_labels = labels[order]
    _, inverse, counts = torch.unique_consecutive(
        sorted_scores, return_inverse=True, return_counts=True
    )
    ends = counts.cumsum(0).to(torch.float64)
    starts = ends - counts.to(torch.float64) + 1.0
    average_ranks = (starts + ends) / 2.0
    ranks = average_ranks[inverse]
    positive_rank_sum = ranks[sorted_labels].sum()
    return (positive_rank_sum - positives * (positives + 1) / 2) / (
        positives * negatives
    )


@torch.inference_mode()
def evaluate_auc(model: torch.nn.Module, features: torch.Tensor, labels: torch.Tensor) -> float:
    model.eval()
    logits = model(features)
    return float(binary_auc(logits, labels).item())


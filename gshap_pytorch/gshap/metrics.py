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
def binary_auc_batch(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Batch of `binary_auc` over the leading axis of `scores`.

    Every arithmetic step mirrors `binary_auc` per row (same float64 integer-exact
    rank arithmetic), so each stream's result is bit-identical to the per-row call.
    """
    scores = scores.to(torch.float64)
    labels = labels.reshape(-1).to(torch.bool)
    streams, count = scores.shape
    positives = labels.sum()
    negatives = count - positives
    if int(positives) == 0 or int(negatives) == 0:
        raise ValueError("ROC AUC requires both classes")

    sorted_scores, order = torch.sort(scores, dim=1)
    sorted_labels = labels[order]
    is_start = torch.ones((streams, 1), dtype=torch.bool, device=scores.device)
    if count > 1:
        is_start = torch.cat([is_start, sorted_scores[:, 1:] != sorted_scores[:, :-1]], dim=1)
    group_id = is_start.to(torch.int64).cumsum(dim=1) - 1
    ones = torch.ones_like(sorted_scores, dtype=torch.float64)
    counts = torch.zeros_like(sorted_scores, dtype=torch.float64).scatter_add_(1, group_id, ones)
    ends = counts.cumsum(dim=1)
    starts = ends - counts + 1.0
    average_ranks = (starts + ends) / 2.0
    ranks = average_ranks.gather(1, group_id)
    positive_rank_sum = torch.empty(streams, dtype=torch.float64, device=scores.device)
    for stream in range(streams):
        positive_rank_sum[stream] = ranks[stream][sorted_labels[stream]].sum()
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


@torch.inference_mode()
def evaluate_auc(model: torch.nn.Module, features: torch.Tensor, labels: torch.Tensor) -> float:
    model.eval()
    logits = model(features)
    return float(binary_auc(logits, labels).item())


import torch

from gshap.metrics import binary_auc


def test_auc_perfect_and_reversed():
    labels = torch.tensor([0, 0, 1, 1])
    assert binary_auc(torch.tensor([0.1, 0.2, 0.8, 0.9]), labels).item() == 1.0
    assert binary_auc(torch.tensor([0.9, 0.8, 0.2, 0.1]), labels).item() == 0.0


def test_auc_averages_tied_ranks():
    labels = torch.tensor([0, 1, 0, 1])
    scores = torch.tensor([0.0, 0.5, 0.5, 1.0])
    assert binary_auc(scores, labels).item() == 0.875


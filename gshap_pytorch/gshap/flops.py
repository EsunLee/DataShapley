from __future__ import annotations

import math


def forward_flops(input_dim: int = 512, hidden_dim: int = 128) -> int:
    return 2 * input_dim * hidden_dim + 2 * hidden_dim


def training_flops_per_sample(input_dim: int = 512, hidden_dim: int = 128) -> int:
    return 3 * forward_flops(input_dim, hidden_dim)


def permutation_flops(
    n_train: int,
    n_test: int,
    batch_size: int,
    input_dim: int = 512,
    hidden_dim: int = 128,
) -> tuple[int, int, int]:
    train = n_train * training_flops_per_sample(input_dim, hidden_dim)
    evaluations = math.ceil(n_train / batch_size)
    evaluation = evaluations * n_test * forward_flops(input_dim, hidden_dim)
    return train, evaluation, train + evaluation


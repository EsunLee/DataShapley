import numpy as np


def test_batch_delta_is_shared_without_duplication():
    marginal = np.zeros(7)
    batches = [np.array([2, 0, 5]), np.array([1, 4, 3]), np.array([6])]
    scores = [0.55, 0.61, 0.60]
    previous = 0.5
    for batch, score in zip(batches, scores, strict=True):
        marginal[batch] = (score - previous) / len(batch)
        previous = score
    assert np.isclose(marginal.sum(), scores[-1] - 0.5)
    assert np.allclose(marginal[batches[0]], (0.55 - 0.5) / 3)
    assert marginal[6] == 0.60 - 0.61


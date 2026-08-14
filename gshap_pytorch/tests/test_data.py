import numpy as np

from gshap.data import create_split


def test_stratified_split_is_disjoint_complete_and_reproducible():
    labels = np.array([0] * 840 + [1] * 160)
    first = create_split(labels, test_size=100, lr_val_size=100, seed=0)
    second = create_split(labels, test_size=100, lr_val_size=100, seed=0)
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)
    train, test, lr_val = first
    combined = np.concatenate((train, test, lr_val))
    assert len(np.unique(combined)) == len(labels)
    assert np.bincount(labels[test], minlength=2).tolist() == [84, 16]
    assert np.bincount(labels[lr_val], minlength=2).tolist() == [84, 16]


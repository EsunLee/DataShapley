from gshap.flops import forward_flops, permutation_flops, training_flops_per_sample


def test_flop_formulas():
    assert forward_flops() == 131_328
    assert training_flops_per_sample() == 393_984
    train, evaluation, total = permutation_flops(694_459, 5_000, 128)
    assert train == 694_459 * 393_984
    assert evaluation == 5_426 * 5_000 * 131_328
    assert total == train + evaluation


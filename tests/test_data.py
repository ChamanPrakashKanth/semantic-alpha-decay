import torch

from data.selective_recall import BatchFactory, VALUES


def test_values_are_distinct_and_positions_vary():
    batch = BatchFactory(0).batch(256)
    positions = set(batch.relevant_positions.tolist())
    assert len(positions) > 2
    for row in batch.tokens:
        present = [int(x) for x in row if int(x) in VALUES]
        assert len(present) == 2 and present[0] != present[1]


def test_batches_are_reproducible_and_ood_is_longer():
    a, b = BatchFactory(7), BatchFactory(7)
    assert torch.equal(a.batch(32).tokens, b.batch(32).tokens)
    assert BatchFactory(1).batch(64, "ood").tokens.shape[1] > BatchFactory(1).batch(64, "train").tokens.shape[1]

import torch

from data.selective_recall import BatchFactory, KEY_A, KEY_B, SEP, VALUES


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


def test_shifts_are_isolated():
    factory = BatchFactory(11)
    ordinary = factory.batch(128, "unseen_combinations")
    layout = BatchFactory(11).batch(128, "unseen_layout")
    reversed_batch = BatchFactory(11).batch(128, "reversed_order")
    # Only unseen-layout moves SEP before the query.
    for row, q in zip(ordinary.tokens, ordinary.query_positions):
        assert int(row[q + 1]) == SEP
    for row, q in zip(layout.tokens, layout.query_positions):
        assert SEP in row[:q]
    # Only reversed-order puts a value immediately before each key.
    for row in reversed_batch.tokens:
        for key in (KEY_A, KEY_B):
            pos = (row == key).nonzero()[0].item()
            assert int(row[pos - 1]) in VALUES

import torch

from gru_linear_small_model.data import RelationChainGenerator


def test_generator_is_deterministic_and_target_is_true_endpoint():
    left = RelationChainGenerator(24, 7)
    right = RelationChainGenerator(24, 7)
    for _ in range(20):
        a = left.sample_example(5, 4)
        b = right.sample_example(5, 4)
        assert a == b
        edge_map = dict(a.edges)
        node = a.query
        for _ in range(a.chain_length):
            node = edge_map[node]
        assert node == a.target


def test_padding_and_shapes():
    batch = RelationChainGenerator(24, 2).sample_batch(8, [2, 3], [0, 1])
    assert batch["tokens"].shape == batch["mask"].shape == batch["roles"].shape
    assert batch["targets"].shape == (8,)
    assert batch["tokens"].dtype == torch.long
    assert batch["mask"].dtype == torch.bool


def test_distractors_do_not_create_query_path():
    generator = RelationChainGenerator(24, 17)
    for _ in range(100):
        example = generator.sample_example(4, 6)
        outgoing = {}
        for src, dst in example.edges:
            outgoing.setdefault(src, []).append(dst)
        frontier, seen = [example.query], {example.query}
        while frontier:
            src = frontier.pop()
            for dst in outgoing.get(src, []):
                if dst not in seen:
                    seen.add(dst)
                    frontier.append(dst)
        # Only the true chain is reachable, so it contains exactly L+1 nodes.
        assert len(seen) == example.chain_length + 1


def test_destination_balancing_removes_candidate_position_shortcut():
    generator = RelationChainGenerator(24, 31)
    first_correct = last_correct = 0
    trials = 2400
    for _ in range(trials):
        example = generator.sample_example(2, 0)
        destinations = example.tokens[2:-2:4]
        assert sorted(destinations) == list(range(24))
        first_correct += destinations[0] == example.target
        last_correct += destinations[-1] == example.target
    # Sampling noise around 1/24; this bound is intentionally generous.
    assert first_correct / trials < 0.065
    assert last_correct / trials < 0.065

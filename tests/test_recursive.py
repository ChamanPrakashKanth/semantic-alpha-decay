import torch

from data import BatchFactory
from recursive_learning import add_self_feedback


def test_feedback_uses_own_answer_and_repeats_query_without_new_tokens():
    batch = BatchFactory(0).batch(8)
    predicted = torch.arange(8) + 7
    feedback = add_self_feedback(batch, predicted)
    rows = torch.arange(8)
    assert torch.equal(feedback.tokens[rows, feedback.query_positions - 1], predicted)
    assert torch.equal(feedback.tokens[rows, feedback.query_positions],
                       batch.tokens[rows, batch.query_positions])
    assert torch.equal(feedback.relevant_positions, batch.relevant_positions)

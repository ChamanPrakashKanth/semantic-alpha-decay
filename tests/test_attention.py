import torch

from sadt.attention import SemanticDecayAttention


def test_post_softmax_is_not_renormalized():
    torch.manual_seed(0)
    layer = SemanticDecayAttention(mode="fixed")
    _, info = layer(torch.randn(2, 6, 32), exposure=1.0)
    sums = info["attention"].sum(-1)
    assert torch.allclose(sums, torch.full_like(sums, torch.exp(torch.tensor(-1.0))), atol=1e-5)


def test_renormalized_rows_sum_to_one():
    layer = SemanticDecayAttention(mode="renorm")
    _, info = layer(torch.randn(2, 6, 32))
    assert torch.allclose(info["attention"].sum(-1), torch.ones(2, 4, 6), atol=1e-5)


def test_zero_relevant_is_causal_intervention():
    layer = SemanticDecayAttention(mode="learned")
    q = torch.tensor([5, 4]); r = torch.tensor([2, 1])
    _, info = layer(torch.randn(2, 6, 32), intervention="zero_relevant",
                    query_positions=q, relevant_positions=r)
    assert torch.equal(info["survival"][torch.arange(2), :, q, r], torch.zeros(2, 4))

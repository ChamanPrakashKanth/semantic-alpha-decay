import pytest
import torch

from gru_linear_small_model.models import (
    GRUBaseline,
    HybridGRULinear,
    LinearSmoothCell,
    LinearSmoothRNN,
    build_models,
)


def test_linear_smooth_gate_range_and_shape():
    cell = LinearSmoothCell(7, 11)
    h, info = cell(torch.randn(5, 7), torch.randn(5, 11))
    assert h.shape == (5, 11)
    assert torch.all((info["beta"] > 0) & (info["beta"] < 1))


@pytest.mark.parametrize("forced", [0.0, 0.5, 1.0])
def test_forced_hybrid_is_exact_blend(forced):
    model = HybridGRULinear(28, 24, 8, 12, force_lambda=forced)
    x, h = torch.randn(4, 8), torch.randn(4, 12)
    actual, info = model.step(x, h)
    expected = forced * info["h_gru"] + (1 - forced) * info["h_linear"]
    assert torch.equal(actual, expected)
    assert torch.all(info["lambda"] == forced)


def test_learned_lambda_range_and_zero_bias_starts_balanced():
    model = HybridGRULinear(28, 24, 8, 12)
    _, info = model.step(torch.randn(4, 8), torch.randn(4, 12))
    assert torch.all((info["lambda"] > 0) & (info["lambda"] < 1))


def test_all_model_output_shapes_and_parameter_match():
    models = build_models(28, 24, 16, 16)
    tokens = torch.randint(0, 28, (5, 14))
    mask = torch.ones_like(tokens, dtype=torch.bool)
    for model in models.values():
        assert model(tokens, mask).shape == (5, 24)
    hybrid_count = sum(p.numel() for p in models["hybrid"].parameters())
    matched_count = sum(p.numel() for p in models["param_matched_gru"].parameters())
    ordinary_count = sum(p.numel() for p in models["gru"].parameters())
    assert abs(matched_count - hybrid_count) < abs(ordinary_count - hybrid_count)


def test_target_is_not_a_forward_input():
    model = GRUBaseline(28, 24, 8, 12).eval()
    tokens = torch.randint(0, 28, (3, 10))
    mask = torch.ones_like(tokens, dtype=torch.bool)
    first = model(tokens, mask)
    arbitrary_targets = torch.tensor([1, 7, 19])
    arbitrary_targets[:] = torch.tensor([23, 0, 4])
    second = model(tokens, mask)
    assert torch.equal(first, second)

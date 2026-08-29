"""Unit tests for autonomous teacher-free recursive self-learning."""

import pytest
import torch

from autonomous_recursive_test import (
    evaluate_autonomous_recursion,
    recursive_teacher_free_forward,
)
from controlled_competence_rsl import (
    ControlledCompetenceModel,
    ThreeSectorAngularEnvironment,
)


def test_recursive_teacher_free_forward_isolation():
    """Verify that multi-pass recursion executes without targets, losses, or external rewards."""
    torch.manual_seed(42)
    model = ControlledCompetenceModel(in_dim=16, num_classes=4)
    env = ThreeSectorAngularEnvironment(seed=0)
    batch = env.generate_batch(8, split="test_sector")

    # Run 5 recursive passes
    with torch.no_grad():
        passes = recursive_teacher_free_forward(
            model,
            batch.tokens,
            batch.query_positions,
            k_passes=5,
            padding_mask=batch.padding_mask,
        )

    assert len(passes) == 5
    for p in passes:
        assert p["predictions"].shape == (8,)
        assert 0.0 <= p["mean_gate"] <= 1.0
        assert 0.0 <= p["mean_effective_D"] <= 1.0
        assert p["representation_drift"] >= 0.0


def test_forced_intervention_gate_override():
    """Verify that forced gate overrides gate behavior exactly."""
    torch.manual_seed(42)
    model = ControlledCompetenceModel(in_dim=16, num_classes=4)
    env = ThreeSectorAngularEnvironment(seed=0)
    batch = env.generate_batch(8, split="test_sector")

    with torch.no_grad():
        passes_0 = recursive_teacher_free_forward(
            model, batch.tokens, batch.query_positions, k_passes=3, forced_gate=0.0
        )
        for p in passes_0:
            assert abs(p["mean_effective_D"] - 1.0) < 1e-5

        passes_1 = recursive_teacher_free_forward(
            model, batch.tokens, batch.query_positions, k_passes=3, forced_gate=1.0
        )
        for p in passes_1:
            assert p["mean_gate"] == 1.0

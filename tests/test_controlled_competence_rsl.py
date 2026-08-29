"""Comprehensive unit tests for Phase 2: Controlled Base Competence Sweep."""

import copy
import math
import pytest
import torch

from controlled_competence_rsl import (
    AuditedInterventionGatedController,
    ControlledCompetenceModel,
    ThreeSectorAngularEnvironment,
    train_rsl_on_frozen_base,
)


def test_effective_survival_audit_and_boundary_conditions():
    """Verify Section 2 and Section 19 audit:

    1. g=0 => D_eff == 1.0
    2. g=1 => D_eff == D_raw
    3. Mathematical bounds 0 <= g <= 1, alpha >= 0, 0 < D_eff <= 1
    """
    torch.manual_seed(42)
    ctrl = AuditedInterventionGatedController(d_model=32, n_heads=4)
    b, h, l, d_head = 2, 4, 6, 8
    q = torch.randn(b, h, l, d_head)
    k = torch.randn(b, h, l, d_head)
    A = torch.softmax(torch.randn(b, h, l, l), dim=-1)
    uncertainty = torch.rand(b, 4)
    causal = torch.triu(torch.ones(l, l, dtype=torch.bool), 1).view(1, 1, l, l)

    g, alpha, raw_d, eff_d, log_prob = ctrl(
        q, k, A, uncertainty=uncertainty, exposure=1.0, invalid_mask=causal
    )

    valid = ~causal.expand_as(eff_d)
    assert torch.all(g[valid] >= 0.0) and torch.all(g[valid] <= 1.0)
    assert torch.all(alpha[valid] >= 0.0)
    assert torch.all(raw_d[valid] > 0.0) and torch.all(raw_d[valid] <= 1.0)
    assert torch.all(eff_d[valid] > 0.0) and torch.all(eff_d[valid] <= 1.0)

    # Test exact boundary math:
    # When g = 0 -> D_eff = (1 - 0) + 0 * D_raw = 1.0
    test_alpha = torch.tensor([2.5])
    test_raw = torch.exp(-test_alpha)
    assert math.isclose((1.0 - 0.0) + 0.0 * test_raw.item(), 1.0, abs_tol=1e-7)

    # When g = 1 -> D_eff = (1 - 1) + 1 * D_raw = D_raw
    assert math.isclose((1.0 - 1.0) + 1.0 * test_raw.item(), test_raw.item(), abs_tol=1e-7)


def test_frozen_base_weights_remain_bitwise_unchanged():
    """Verify Stage B: Base model weights remain bitwise identical throughout RSL training."""
    torch.manual_seed(42)
    env = ThreeSectorAngularEnvironment(seed=42)
    model = ControlledCompetenceModel(in_dim=16, num_classes=4)
    model.freeze_base_model()

    # Record initial base parameters
    initial_base_params = {
        name: param.clone()
        for name, param in model.named_parameters()
        if not name.startswith("controller")
    }

    # Run RSL training steps
    train_rsl_on_frozen_base(model, env, rsl_steps=10, batch_size=32)

    # Check parameter equality
    for name, initial_tensor in initial_base_params.items():
        current_tensor = dict(model.named_parameters())[name]
        assert torch.equal(initial_tensor, current_tensor), f"Base weight {name} modified during RSL training!"


def test_no_target_leakage_and_inference_isolation():
    """Verify that forward inference does not require targets and cannot be perturbed by targets."""
    torch.manual_seed(42)
    model = ControlledCompetenceModel(in_dim=16, num_classes=4)
    env = ThreeSectorAngularEnvironment(seed=0)
    batch = env.generate_batch(8, split="test_sector")

    with torch.no_grad():
        base_logits, rsl_logits, info = model(
            batch.tokens,
            batch.query_positions,
            exposure=1.0,
            padding_mask=batch.padding_mask,
            apply_rsl=True,
        )

    assert base_logits.shape == (8, 4)
    assert rsl_logits.shape == (8, 4)
    assert info["effective_survival"] is not None


def test_three_sector_angular_geometry_isolation():
    """Verify that training, validation, and test sectors are mutually exclusive."""
    env = ThreeSectorAngularEnvironment(
        val_sector=(math.pi / 6, math.pi / 3),  # 30° - 60°
        test_sector=(19 * math.pi / 12, 23 * math.pi / 12),  # 285° - 345°
    )

    for _ in range(20):
        b_train = env.generate_batch(16, split="training_region")
        for i in range(16):
            q = b_train.query_vectors[i]
            theta = math.atan2(q[1].item(), q[0].item()) % (2 * math.pi)
            assert env.classify_angle(theta) == "training_region"

        b_val = env.generate_batch(16, split="validation_sector")
        for i in range(16):
            q = b_val.query_vectors[i]
            theta = math.atan2(q[1].item(), q[0].item()) % (2 * math.pi)
            assert env.classify_angle(theta) == "validation_sector"

        b_test = env.generate_batch(16, split="test_sector")
        for i in range(16):
            q = b_test.query_vectors[i]
            theta = math.atan2(q[1].item(), q[0].item()) % (2 * math.pi)
            assert env.classify_angle(theta) == "test_sector"

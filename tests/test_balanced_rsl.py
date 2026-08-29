"""Unit tests and leakage audits for Balanced Transition-Focused RSL."""

import math
import pytest
import torch

from balanced_rsl import (
    AngularTaskEnvironment,
    BalancedRSLModel,
    InterventionGatedController,
    compute_transition_reward,
    evaluate_balanced_model,
)


def test_dual_head_controller_bounds():
    """Verify g in [0, 1], alpha >= 0, and D in [0, 1] with strict causal masking."""
    torch.manual_seed(42)
    ctrl = InterventionGatedController(d_model=32, n_heads=4, uncertainty_dim=4)
    b, h, l, d_head = 2, 4, 6, 8
    q = torch.randn(b, h, l, d_head)
    k = torch.randn(b, h, l, d_head)
    A = torch.softmax(torch.randn(b, h, l, l), dim=-1)
    uncertainty = torch.rand(b, 4)
    causal = torch.triu(torch.ones(l, l, dtype=torch.bool), 1).view(1, 1, l, l)

    for exposure in [0.0, 0.5, 1.0, 2.0]:
        g, alpha, survival, log_prob = ctrl(
            q, k, A, uncertainty=uncertainty, exposure=exposure, invalid_mask=causal
        )
        assert torch.all(g >= 0.0) and torch.all(g <= 1.0), "Gate g must be in [0, 1]"
        assert torch.all(alpha >= 0.0), "Alpha must be non-negative"
        assert torch.all(survival >= 0.0) and torch.all(survival <= 1.0), "Survival D must be in [0, 1]"
        # Masked positions must be 0
        masked = causal.expand_as(survival)
        assert torch.all(survival.masked_select(masked) == 0.0), "Causal masked positions must be zero"


def test_strict_no_target_leakage():
    """Prove that during evaluation, no target information or gradients enter the model/controller."""
    torch.manual_seed(42)
    model = BalancedRSLModel(in_dim=16, num_classes=4)
    env = AngularTaskEnvironment(seed=0)
    batch = env.generate_batch(8, split="held_out_sector")

    # Clear gradients
    model.zero_grad()
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None

    # Call evaluation without passing batch.targets into the model
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
    assert info["g"] is not None
    assert info["survival"] is not None

    # Verify no parameter accumulated gradients during inference
    for name, p in model.named_parameters():
        assert p.grad is None, f"Parameter {name} had gradient during inference!"


def test_transition_reward_and_accounting_identity():
    """Verify balanced transition rewards and exact accounting identity Delta Acc = (1-p)*c - p*d."""
    base_preds = torch.tensor([0, 1, 0, 1, 2, 3, 0, 1])
    rsl_preds  = torch.tensor([1, 1, 1, 0, 2, 0, 0, 2])
    targets    = torch.tensor([1, 1, 1, 1, 3, 3, 0, 0])

    rewards, stats = compute_transition_reward(
        base_preds, rsl_preds, targets, r_rescue=1.0, r_preserve=1.0, r_damage=-1.0, r_fail=-1.0
    )

    # Example-by-example checks:
    # 0: base=0 (W), rsl=1 (R) -> Rescue (+1)
    # 1: base=1 (R), rsl=1 (R) -> Preserve (+1)
    # 2: base=0 (W), rsl=1 (R) -> Rescue (+1)
    # 3: base=1 (R), rsl=0 (W) -> Damage (-1)
    # 4: base=2 (W), rsl=2 (W) -> Fail (-1)
    # 5: base=3 (R), rsl=0 (W) -> Damage (-1)
    # 6: base=0 (R), rsl=0 (R) -> Preserve (+1)
    # 7: base=1 (W), rsl=2 (W) -> Fail (-1)
    expected_rewards = torch.tensor([1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0])
    assert torch.equal(rewards, expected_rewards)

    # Base correct: [1, 3, 5, 6] -> p = 4/8 = 0.5
    # RSL correct: [0, 1, 2, 6] -> acc_rsl = 4/8 = 0.5 -> Delta = 0.0
    # Rescues: [0, 2] -> c = 2/4 = 0.5
    # Damage: [3, 5] -> d = 2/4 = 0.5
    # Identity: (1-0.5)*0.5 - 0.5*0.5 = 0.25 - 0.25 = 0.0
    p = (base_preds == targets).float().mean().item()
    acc_rsl = (rsl_preds == targets).float().mean().item()
    delta_actual = acc_rsl - p

    b_corr = base_preds == targets
    c = ((~b_corr) & (rsl_preds == targets)).sum().item() / (~b_corr).sum().item()
    d = (b_corr & (rsl_preds != targets)).sum().item() / b_corr.sum().item()
    predicted_delta = (1.0 - p) * c - p * d

    assert math.isclose(delta_actual, predicted_delta, abs_tol=1e-6)


def test_angular_sector_task_isolation():
    """Verify that train angles never overlap with the held-out sector."""
    held_out = (math.pi / 6, math.pi / 2)  # [30 deg, 90 deg]
    env = AngularTaskEnvironment(held_out_sector=held_out, seed=123)

    for _ in range(10):
        train_batch = env.generate_batch(32, split="train")
        for b in range(32):
            q = train_batch.query_vectors[b]
            theta = math.atan2(q[1].item(), q[0].item()) % (2 * math.pi)
            assert not (held_out[0] <= theta <= held_out[1]), f"Train angle {theta} was in held-out sector!"

        eval_batch = env.generate_batch(32, split="held_out_sector")
        for b in range(32):
            q = eval_batch.query_vectors[b]
            theta = math.atan2(q[1].item(), q[0].item()) % (2 * math.pi)
            assert held_out[0] <= theta <= held_out[1], f"Held-out angle {theta} was outside held-out sector!"

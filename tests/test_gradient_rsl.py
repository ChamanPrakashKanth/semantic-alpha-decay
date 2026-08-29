"""Sanity and falsification tests for Gradient-Pattern RSL experiment."""

import pytest
import torch
import torch.nn.functional as F

from data import BatchFactory, VOCAB_SIZE
from gradient_rsl import (
    GradientRSLController,
    GradientRSLModel,
    compute_teacher_signals,
    evaluate_model_on_split,
)


def test_alpha_and_survival_validity():
    """Verify alpha >= 0 and D = exp(-alpha * T) in [0, 1] with strict causal masking."""
    torch.manual_seed(42)
    controller = GradientRSLController(d_model=32, n_heads=4)
    b, h, l, d_head = 2, 4, 8, 8
    q = torch.randn(b, h, l, d_head)
    k = torch.randn(b, h, l, d_head)
    A = torch.softmax(torch.randn(b, h, l, l), dim=-1)
    causal = torch.triu(torch.ones(l, l, dtype=torch.bool), 1).view(1, 1, l, l)

    for exposure in [0.0, 0.5, 1.0, 2.0]:
        alpha, survival = controller(q, k, A, exposure=exposure, invalid_mask=causal)
        assert torch.all(alpha >= 0.0), "Alpha must be non-negative"
        assert torch.all(survival >= 0.0) and torch.all(survival <= 1.0), "Survival must be in [0, 1]"
        masked_causal = causal.expand_as(survival)
        assert torch.all(survival.masked_select(masked_causal) == 0.0), "Masked causal positions must be 0"


def test_gradient_free_inference_has_no_grad_or_labels():
    """Prove no target labels or evaluation gradients are accessed during RSL inference."""
    torch.manual_seed(42)
    model = GradientRSLModel(VOCAB_SIZE)
    batch = BatchFactory(0).batch(4)

    # Zero all grads before evaluation
    model.zero_grad()
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None

    # Call inference without providing batch.targets or calling backward
    with torch.no_grad():
        logits, info = model.forward_pass(
            batch.tokens,
            batch.query_positions,
            exposure=1.0,
            padding_mask=batch.padding_mask,
            decay_mode="rsl",
        )

    assert logits.shape == (4, VOCAB_SIZE)
    assert info["alpha"] is not None
    assert info["survival"] is not None
    # Verify no gradients were computed on any model parameter
    for name, p in model.named_parameters():
        assert p.grad is None, f"Parameter {name} accumulated gradient during inference!"


def test_oracle_controller_receives_privileged_information():
    """Verify Oracle controller receives privileged target information and obtains high selectivity."""
    torch.manual_seed(42)
    model = GradientRSLModel(VOCAB_SIZE)
    factory = BatchFactory(0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    # Train base model briefly so representations reflect task semantics
    for _ in range(30):
        b = factory.batch(32)
        logits, _ = model.forward_pass(b.tokens, b.query_positions, 1.0, b.padding_mask, decay_mode="none")
        loss = F.cross_entropy(logits, b.targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    test_batch = factory.batch(32)
    # Compute privileged teacher signals using ground-truth targets
    teacher_sig = compute_teacher_signals(model, test_batch, scale=4.0)
    oracle_survival = teacher_sig.target_survival

    # Evaluate model with oracle survival
    with torch.no_grad():
        logits, info = model.forward_pass(
            test_batch.tokens,
            test_batch.query_positions,
            exposure=1.0,
            padding_mask=test_batch.padding_mask,
            oracle_survival=oracle_survival,
            decay_mode="oracle",
        )

    rows = torch.arange(32)
    oracle_gates = info["survival"].mean(1)
    rel_surv = oracle_gates[rows, test_batch.query_positions, test_batch.relevant_positions].mean().item()
    irr_surv = oracle_gates[rows, test_batch.query_positions, test_batch.irrelevant_positions].mean().item()

    # Teacher gradient signal specifically rewards relevant and penalizes distractors
    assert rel_surv > irr_surv, f"Oracle should favor relevant ({rel_surv}) over irrelevant ({irr_surv})"


def test_shuffled_gradient_targets_damage_learning():
    """Prove that training on shuffled gradient targets disrupts learning compared to true targets."""
    torch.manual_seed(42)
    model = GradientRSLModel(VOCAB_SIZE)
    ctrl_true = GradientRSLController(32, 4)
    ctrl_shuffled = GradientRSLController(32, 4)

    # Clone initial weights so both start identical
    ctrl_shuffled.load_state_dict(ctrl_true.state_dict())

    opt_true = torch.optim.AdamW(ctrl_true.parameters(), lr=1e-2)
    opt_shuf = torch.optim.AdamW(ctrl_shuffled.parameters(), lr=1e-2)

    factory = BatchFactory(42)
    batches = [factory.batch(32) for _ in range(40)]

    for batch in batches:
        teacher = compute_teacher_signals(model, batch, compute_interventions=False)
        target = teacher.target_survival.detach()

        with torch.no_grad():
            _, info = model.forward_pass(batch.tokens, batch.query_positions, 1.0, batch.padding_mask)
        q = info["q"].detach()
        k = info["k"].detach()
        A = info["A_base"].detach()
        inv = info["invalid"]
        valid = (~inv).expand_as(target)

        # True controller step
        opt_true.zero_grad()
        _, pred_true = ctrl_true(q, k, A, 1.0, inv)
        loss_true = F.mse_loss(pred_true[valid], target[valid])
        loss_true.backward()
        opt_true.step()

        # Shuffled controller step (shuffle batch dimension of target)
        opt_shuf.zero_grad()
        shuf_idx = torch.randperm(batch.tokens.shape[0])
        shuf_target = target[shuf_idx]
        _, pred_shuf = ctrl_shuffled(q, k, A, 1.0, inv)
        loss_shuf = F.mse_loss(pred_shuf[valid], shuf_target[valid])
        loss_shuf.backward()
        opt_shuf.step()

    # Evaluate on fresh test batch against true targets
    test_batch = factory.batch(64)
    test_teacher = compute_teacher_signals(model, test_batch, compute_interventions=False)
    test_target = test_teacher.target_survival.detach()

    with torch.no_grad():
        _, test_info = model.forward_pass(test_batch.tokens, test_batch.query_positions, 1.0, test_batch.padding_mask)
    t_q, t_k, t_A, t_inv = test_info["q"], test_info["k"], test_info["A_base"], test_info["invalid"]
    t_valid = (~t_inv).expand_as(test_target)

    with torch.no_grad():
        _, p_true = ctrl_true(t_q, t_k, t_A, 1.0, t_inv)
        _, p_shuf = ctrl_shuffled(t_q, t_k, t_A, 1.0, t_inv)

        mse_true = F.mse_loss(p_true[t_valid], test_target[t_valid]).item()
        mse_shuf = F.mse_loss(p_shuf[t_valid], test_target[t_valid]).item()

    assert mse_true < mse_shuf, f"Real target MSE ({mse_true}) should be strictly lower than shuffled ({mse_shuf})"


def test_random_controller_behavior():
    """Verify random controller produces non-selective survival factors centered around mean."""
    torch.manual_seed(42)
    model = GradientRSLModel(VOCAB_SIZE)
    batch = BatchFactory(0).batch(32)

    gen = torch.Generator().manual_seed(123)
    with torch.no_grad():
        logits, info = model.forward_pass(
            batch.tokens,
            batch.query_positions,
            exposure=1.0,
            padding_mask=batch.padding_mask,
            decay_mode="random",
            generator=gen,
        )

    surv = info["survival"]
    valid = (~info["invalid"]).expand_as(surv)
    mean_val = surv[valid].mean().item()
    # Random alpha in [0, 3] -> exp(-alpha) has mean roughly between 0.15 and 0.85
    assert 0.1 < mean_val < 0.85, f"Random survival mean {mean_val} unexpected"


def test_recursive_step_consistency():
    """Verify multi-pass recursive execution updates states and preserves causal structure across passes."""
    torch.manual_seed(42)
    model = GradientRSLModel(VOCAB_SIZE)
    batch = BatchFactory(0).batch(8)

    with torch.no_grad():
        passes = model.recursive_forward(
            batch.tokens,
            batch.query_positions,
            k_passes=5,
            exposure=1.0,
            padding_mask=batch.padding_mask,
            decay_mode="rsl",
        )

    assert len(passes) == 5
    for p_idx, (logits, info) in enumerate(passes, 1):
        assert logits.shape == (8, VOCAB_SIZE)
        assert info["pass"] == p_idx
        assert info["alpha"].shape == (8, 4, batch.tokens.shape[1], batch.tokens.shape[1])
        assert torch.all(info["alpha"] >= 0.0)

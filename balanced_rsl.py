"""Balanced Transition-Focused Reinforcement Learning (Transition RL) for Semantic Decay.

Implements the research program from gradient_rsl_research_cookbook.md:
1. Dual-head controller: explicit intervention gate g in [0, 1] and decay rate alpha >= 0.
   D = (1 - g) + g * exp(-alpha * T)
2. Inference uncertainty features (max prob, margin, predictive entropy, attention entropy).
3. Balanced transition reward: R(W->R)=+1, R(R->R)=+1, R(R->W)=-1, R(W->W)=-1.
4. Unnecessary-decay preservation regularizer L_preserve = lambda_p * E[1 - D].
5. Angular sector task manifold with held-out never-rewarded sector.
6. Accounting identity verification: Delta Acc = (1 - p)*c - p*d.
7. Multi-seed statistical benchmark (10-20 seeds) with confidence intervals and CSV tables.
"""

import argparse
import copy
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from statistics import mean, median, stdev
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import BatchFactory, SHIFTS, VOCAB_SIZE
from data.selective_recall import Batch
from sadt.losses import retention_loss
from sadt.model import TinySADT

torch.set_num_threads(1)


# =============================================================================
# Angular Sector Task Environment
# =============================================================================

@dataclass
class AngularBatch:
    """Batch of inputs for multi-task angular linear reasoning."""
    tokens: torch.Tensor  # (B, L, dim)
    query_vectors: torch.Tensor  # (B, 2) task unit vector q = (cos theta, sin theta)
    key_vectors: torch.Tensor  # (B, N, 2) candidate slot vectors
    targets: torch.Tensor  # (B,) index of the correct key / class
    query_positions: torch.Tensor
    relevant_positions: torch.Tensor
    irrelevant_positions: torch.Tensor
    padding_mask: torch.Tensor
    is_held_out: torch.Tensor  # Boolean mask per example


class AngularTaskEnvironment:
    """Task manifold parameterized by direction angle theta in [0, 2*pi).

    Withholds a contiguous angular sector [theta_min, theta_max] from training reward.
    """

    def __init__(
        self,
        held_out_sector: Tuple[float, float] = (math.pi / 6, math.pi / 2),  # 30 deg to 90 deg
        num_slots: int = 4,
        dim: int = 16,
        seed: int = 0,
    ):
        self.held_out_min, self.held_out_max = held_out_sector
        self.num_slots = num_slots
        self.dim = dim
        self.rng = random.Random(seed)

    def is_angle_held_out(self, theta: float) -> bool:
        theta = theta % (2 * math.pi)
        return self.held_out_min <= theta <= self.held_out_max

    def sample_angle(self, held_out: bool) -> float:
        while True:
            theta = self.rng.uniform(0, 2 * math.pi)
            if self.is_angle_held_out(theta) == held_out:
                return theta

    def generate_batch(
        self, batch_size: int, split: str = "train", device: str = "cpu"
    ) -> AngularBatch:
        """Generate a batch of angular reasoning problems.

        split in {'train', 'seen_eval', 'held_out_sector'}.
        """
        is_eval_held_out = split == "held_out_sector"
        seq_len = self.num_slots + 2
        tokens = torch.zeros(batch_size, seq_len, self.dim, device=device)
        q_pos = torch.full((batch_size,), seq_len - 1, dtype=torch.long, device=device)
        r_pos = torch.zeros(batch_size, dtype=torch.long, device=device)
        i_pos = torch.zeros(batch_size, dtype=torch.long, device=device)
        targets = torch.zeros(batch_size, dtype=torch.long, device=device)
        query_vecs = torch.zeros(batch_size, 2, device=device)
        key_vecs = torch.zeros(batch_size, self.num_slots, 2, device=device)
        held_out_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for b in range(batch_size):
            theta = self.sample_angle(held_out=is_eval_held_out)
            held_out_mask[b] = is_eval_held_out
            q_2d = torch.tensor([math.cos(theta), math.sin(theta)], device=device)
            query_vecs[b] = q_2d

            # Generate candidate 2D directions for slots
            angles = [self.rng.uniform(0, 2 * math.pi) for _ in range(self.num_slots)]
            # Target is the slot with highest inner product q^T k
            scores = [math.cos(theta - a) for a in angles]
            best_idx = int(scores.index(max(scores)))
            worst_idx = int(scores.index(min(scores)))

            targets[b] = best_idx
            r_pos[b] = 1 + best_idx
            i_pos[b] = 1 + worst_idx

            # Embed into token vectors
            # BOS token at index 0
            tokens[b, 0, 0] = 1.0
            for s in range(self.num_slots):
                k_2d = torch.tensor([math.cos(angles[s]), math.sin(angles[s])], device=device)
                key_vecs[b, s] = k_2d
                tokens[b, 1 + s, :2] = k_2d
                tokens[b, 1 + s, 2 + s] = 1.0  # Slot indicator

            # Query token at last position
            tokens[b, seq_len - 1, :2] = q_2d
            tokens[b, seq_len - 1, 2 + self.num_slots] = 1.0

        padding = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        return AngularBatch(
            tokens=tokens,
            query_vectors=query_vecs,
            key_vectors=key_vecs,
            targets=targets,
            query_positions=q_pos,
            relevant_positions=r_pos,
            irrelevant_positions=i_pos,
            padding_mask=padding,
            is_held_out=held_out_mask,
        )


# =============================================================================
# Dual-Head Intervention Gate + Semantic Decay Controller
# =============================================================================

class InterventionGatedController(nn.Module):
    """Dual-head controller predicting both intervention gate g and decay rate alpha.

    Equation:
        g = sigmoid(f_g(q, k, A, uncertainty)) in [0, 1]
        alpha = softplus(f_alpha(q, k, A, uncertainty)) >= 0
        D = (1 - g) + g * exp(-alpha * exposure) in [0, 1]
    """

    def __init__(
        self,
        d_model: int = 32,
        n_heads: int = 4,
        uncertainty_dim: int = 4,  # [max_prob, margin, entropy, attn_entropy]
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.uncertainty_dim = uncertainty_dim

        # Pairwise features: [q_i, k_j, q_i * k_j, |q_i - k_j|] = 4 * d_head
        # Plus attention weight A_ij (1) + uncertainty broadcast (uncertainty_dim)
        feature_dim = 4 * self.d_head + 1 + uncertainty_dim

        self.shared = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        # Head 1: Intervention Gate logit
        self.gate_head = nn.Linear(hidden_dim, 1)
        # Head 2: Decay Magnitude logit
        self.alpha_head = nn.Linear(hidden_dim, 1)

    def extract_features(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        A: torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, h, l, d = q.shape
        qi = q.unsqueeze(3).expand(-1, -1, -1, l, -1)  # (B, H, L, L, d)
        kj = k.unsqueeze(2).expand(-1, -1, l, -1, -1)  # (B, H, L, L, d)
        prod = qi * kj
        diff = (qi - kj).abs()
        a_weight = A.unsqueeze(-1)  # (B, H, L, L, 1)

        feats = [qi, kj, prod, diff, a_weight]
        if uncertainty is not None:
            u_exp = uncertainty[:, None, None, None, :].expand(-1, h, l, l, -1)
            feats.append(u_exp)
        else:
            dummy_u = torch.zeros(b, h, l, l, self.uncertainty_dim, device=q.device)
            feats.append(dummy_u)

        return torch.cat(feats, dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        A: torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None,
        exposure: float = 1.0,
        invalid_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute intervention gate g, decay alpha, survival factor D, and gate log_probs.

        Returns:
            g: Intervention gate in [0, 1], shape (B, H, L, L)
            alpha: Decay rate >= 0, shape (B, H, L, L)
            survival: D = (1 - g) + g * exp(-alpha * exposure), shape (B, H, L, L)
            log_prob: Policy log-probability for RL updates
        """
        feat = self.extract_features(q, k, A, uncertainty)
        h = self.shared(feat)
        gate_logits = self.gate_head(h).squeeze(-1)  # (B, H, L, L)
        alpha_logits = self.alpha_head(h).squeeze(-1)  # (B, H, L, L)

        g = torch.sigmoid(gate_logits)
        alpha = F.softplus(alpha_logits)

        # Survival calculation: D = (1 - g) + g * exp(-alpha * exposure)
        decay_factor = torch.exp(-alpha * exposure)
        survival = (1.0 - g) + g * decay_factor

        if invalid_mask is not None:
            mask = invalid_mask.expand_as(survival)
            survival = survival.masked_fill(mask, 0.0)
            g = g.masked_fill(mask, 0.0)
            alpha = alpha.masked_fill(mask, 0.0)

        # Policy log probability for sampling-based RL
        log_prob = (
            g.clamp(1e-6, 1 - 1e-6).log() * g.detach()
            + (1 - g).clamp(1e-6, 1 - 1e-6).log() * (1 - g.detach())
        ).mean(dim=(-1, -2, -3))

        return g, alpha, survival, log_prob


# =============================================================================
# Transformer Architecture with Intervention-Gated RSL
# =============================================================================

class BalancedRSLModel(nn.Module):
    """Transformer equipped with Intervention-Gated Transition RL Controller."""

    def __init__(
        self,
        in_dim: int = 16,
        num_classes: int = 4,
        max_len: int = 16,
        d_model: int = 32,
        n_heads: int = 4,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.final_ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        self.controller = InterventionGatedController(d_model, n_heads, hidden_dim=hidden_dim)

    def compute_attention(
        self, h: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, l, _ = h.shape
        q, k, v = self.qkv(h).chunk(3, -1)
        shape = (b, l, self.n_heads, self.d_head)
        q, k, v = [z.view(shape).transpose(1, 2) for z in (q, k, v)]
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        causal = torch.triu(torch.ones(l, l, dtype=torch.bool, device=h.device), 1)
        invalid = causal.view(1, 1, l, l)
        if padding_mask is not None:
            invalid = invalid | padding_mask[:, None, None, :]
        A = F.softmax(scores.masked_fill(invalid, -torch.inf), dim=-1)
        A = A.masked_fill(invalid, 0.0)
        return q, k, v, A, invalid

    def extract_uncertainty(self, base_logits: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """Extract legitimate uncertainty metrics available at inference time."""
        probs = F.softmax(base_logits, dim=-1)  # (B, num_classes)
        sorted_probs, _ = torch.sort(probs, descending=True, dim=-1)
        max_prob = sorted_probs[:, 0:1]  # (B, 1)
        second_prob = sorted_probs[:, 1:2] if sorted_probs.shape[1] > 1 else torch.zeros_like(max_prob)
        margin = max_prob - second_prob  # (B, 1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1, keepdim=True)  # (B, 1)
        attn_entropy = -(A * torch.log(A.clamp_min(1e-8))).sum(dim=-1).mean(dim=(-1, -2), keepdim=False)[:, None]  # (B, 1)
        return torch.cat([max_prob, margin, entropy, attn_entropy], dim=-1)

    def forward(
        self,
        tokens: torch.Tensor,
        query_positions: torch.Tensor,
        exposure: float = 1.0,
        padding_mask: Optional[torch.Tensor] = None,
        apply_rsl: bool = True,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Forward execution yielding both base prediction and RSL-corrected prediction."""
        b, l, _ = tokens.shape
        pos = torch.arange(l, device=tokens.device)
        x0 = self.input_proj(tokens) + self.pos_emb(pos)
        h0 = self.ln1(x0)
        q, k, v, A, invalid = self.compute_attention(h0, padding_mask)

        # 1. Base prediction (no decay, D = 1)
        y_base = A @ v
        y_base = y_base.transpose(1, 2).contiguous().view(b, l, self.d_model)
        x_base = x0 + self.out(y_base)
        x_base = x_base + self.ff(self.ln2(x_base))
        rows = torch.arange(b, device=tokens.device)
        base_logits = self.head(self.final_ln(x_base[rows, query_positions]))

        # 2. Extract inference uncertainty features from base prediction
        uncertainty = self.extract_uncertainty(base_logits.detach(), A.detach())

        # 3. RSL Controller prediction
        g = alpha = survival = log_prob = None
        if apply_rsl:
            g, alpha, survival, log_prob = self.controller(
                q.detach(),
                k.detach(),
                A.detach(),
                uncertainty=uncertainty,
                exposure=exposure,
                invalid_mask=invalid,
                deterministic=deterministic,
            )
            decayed_A = A * survival
            y_rsl = decayed_A @ v
            y_rsl = y_rsl.transpose(1, 2).contiguous().view(b, l, self.d_model)
            x_rsl = x0 + self.out(y_rsl)
            x_rsl = x_rsl + self.ff(self.ln2(x_rsl))
            rsl_logits = self.head(self.final_ln(x_rsl[rows, query_positions]))
        else:
            rsl_logits = base_logits
            survival = torch.ones_like(A).masked_fill(invalid.expand_as(A), 0.0)

        info = {
            "q": q,
            "k": k,
            "v": v,
            "A_base": A,
            "g": g,
            "alpha": alpha,
            "survival": survival,
            "log_prob": log_prob,
            "uncertainty": uncertainty,
        }
        return base_logits, rsl_logits, info


# =============================================================================
# Transition Reward & Accounting Identity
# =============================================================================

def compute_transition_reward(
    base_pred: torch.Tensor,
    rsl_pred: torch.Tensor,
    targets: torch.Tensor,
    r_rescue: float = 1.0,  # W -> R
    r_preserve: float = 1.0,  # R -> R
    r_damage: float = -1.0,  # R -> W
    r_fail: float = -1.0,  # W -> W
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute balanced transition reward R and transition statistics."""
    b_corr = base_pred == targets
    r_corr = rsl_pred == targets

    w_to_r = (~b_corr) & r_corr
    r_to_r = b_corr & r_corr
    r_to_w = b_corr & (~r_corr)
    w_to_w = (~b_corr) & (~r_corr)

    reward = torch.zeros_like(base_pred, dtype=torch.float)
    reward[w_to_r] = r_rescue
    reward[r_to_r] = r_preserve
    reward[r_to_w] = r_damage
    reward[w_to_w] = r_fail

    total = float(base_pred.shape[0])
    stats = {
        "P_W_to_R": w_to_r.sum().item() / total,
        "P_R_to_R": r_to_r.sum().item() / total,
        "P_R_to_W": r_to_w.sum().item() / total,
        "P_W_to_W": w_to_w.sum().item() / total,
    }
    return reward, stats


# =============================================================================
# Multi-Seed Evaluation & Benchmark Runner
# =============================================================================

def evaluate_balanced_model(
    model: BalancedRSLModel,
    env: AngularTaskEnvironment,
    split: str,
    batches: int = 10,
    batch_size: int = 128,
) -> Dict[str, Any]:
    """Evaluate base model vs RSL on a split with full transition accounting."""
    model.eval()
    total = 0
    base_correct = 0
    rsl_correct = 0
    w_to_r_total = 0
    r_to_w_total = 0
    r_to_r_total = 0
    w_to_w_total = 0
    all_g = []
    all_alpha = []
    all_survival = []
    all_rel_d = []
    all_irr_d = []

    with torch.no_grad():
        for _ in range(batches):
            b = env.generate_batch(batch_size, split=split)
            b_size = b.tokens.shape[0]
            total += b_size

            base_logits, rsl_logits, info = model(
                b.tokens, b.query_positions, exposure=1.0, padding_mask=b.padding_mask, apply_rsl=True
            )
            base_pred = base_logits.argmax(-1)
            rsl_pred = rsl_logits.argmax(-1)

            b_corr = base_pred == b.targets
            r_corr = rsl_pred == b.targets

            base_correct += b_corr.sum().item()
            rsl_correct += r_corr.sum().item()
            w_to_r_total += ((~b_corr) & r_corr).sum().item()
            r_to_w_total += (b_corr & (~r_corr)).sum().item()
            r_to_r_total += (b_corr & r_corr).sum().item()
            w_to_w_total += ((~b_corr) & (~r_corr)).sum().item()

            if info["g"] is not None:
                all_g.append(info["g"].mean().item())
            if info["alpha"] is not None:
                all_alpha.append(info["alpha"].mean().item())
            if info["survival"] is not None:
                surv = info["survival"]
                all_survival.append(surv.mean().item())
                gates = surv.mean(1)
                rows = torch.arange(b_size)
                all_rel_d.append(gates[rows, b.query_positions, b.relevant_positions].mean().item())
                all_irr_d.append(gates[rows, b.query_positions, b.irrelevant_positions].mean().item())

    p = base_correct / total
    acc_rsl = rsl_correct / total
    c = (w_to_r_total / (total - base_correct)) if total > base_correct else 0.0
    d = (r_to_w_total / base_correct) if base_correct > 0 else 0.0
    delta_acc = acc_rsl - p
    predicted_delta = (1.0 - p) * c - p * d

    rel_d = mean(all_rel_d) if all_rel_d else 1.0
    irr_d = mean(all_irr_d) if all_irr_d else 1.0

    return {
        "base_accuracy": p,
        "rsl_accuracy": acc_rsl,
        "delta_transfer": delta_acc,
        "predicted_delta_identity": predicted_delta,
        "identity_gap": abs(delta_acc - predicted_delta),
        "P_W_to_R_marginal": w_to_r_total / total,
        "P_R_to_W_marginal": r_to_w_total / total,
        "P_W_to_R_conditional_c": c,
        "P_R_to_W_conditional_d": d,
        "P_R_to_R_marginal": r_to_r_total / total,
        "P_W_to_W_marginal": w_to_w_total / total,
        "mean_gate_g": mean(all_g) if all_g else 0.0,
        "mean_alpha": mean(all_alpha) if all_alpha else 0.0,
        "mean_survival_D": mean(all_survival) if all_survival else 1.0,
        "D_rel": rel_d,
        "D_irr": irr_d,
        "delta_D": rel_d - irr_d,
    }


def train_balanced_seed(
    seed: int,
    steps: int = 800,
    batch_size: int = 128,
    base_lr: float = 3e-3,
    rsl_lr: float = 3e-3,
    lambda_p: float = 0.02,
    num_slots: int = 4,
) -> Dict[str, Any]:
    """Train single seed with base Transformer pretraining and balanced transition RL."""
    random.seed(seed)
    torch.manual_seed(seed)

    env = AngularTaskEnvironment(seed=seed, num_slots=num_slots)
    model = BalancedRSLModel(in_dim=16, num_classes=num_slots)

    # 1. Base model supervised training
    opt_base = torch.optim.AdamW(
        [p for n, p in model.named_parameters() if not n.startswith("controller")],
        lr=base_lr,
    )
    # 2. Controller Policy Gradient Optimizer
    opt_ctrl = torch.optim.AdamW(model.controller.parameters(), lr=rsl_lr)

    model.train()
    running_baseline_reward = 0.0

    for step in range(1, steps + 1):
        exposure = min(1.0, step / max(1, steps // 2))

        # Sample training batch (excluding held-out sector)
        batch = env.generate_batch(batch_size, split="train")

        # Step A: Update Base Model with standard supervised task loss
        opt_base.zero_grad()
        base_logits, _, _ = model(
            batch.tokens, batch.query_positions, exposure=1.0, padding_mask=batch.padding_mask, apply_rsl=False
        )
        base_ce = F.cross_entropy(base_logits, batch.targets)
        base_ce.backward()
        opt_base.step()

        # Step B: Train Controller with Balanced Transition Policy Gradient
        opt_ctrl.zero_grad()
        with torch.no_grad():
            base_logits_det, _, _ = model(
                batch.tokens, batch.query_positions, exposure=1.0, padding_mask=batch.padding_mask, apply_rsl=False
            )
            base_preds = base_logits_det.argmax(-1)

        _, rsl_logits, info = model(
            batch.tokens, batch.query_positions, exposure=exposure, padding_mask=batch.padding_mask, apply_rsl=True
        )
        rsl_preds = rsl_logits.argmax(-1)

        rewards, _ = compute_transition_reward(
            base_preds, rsl_preds, batch.targets, r_rescue=1.0, r_preserve=1.0, r_damage=-1.0, r_fail=-1.0
        )
        reward_mean = rewards.mean().item()
        running_baseline_reward = 0.95 * running_baseline_reward + 0.05 * reward_mean
        advantage = rewards - running_baseline_reward

        # Policy Gradient Loss
        log_prob = info["log_prob"]  # (B,)
        policy_loss = -(advantage.detach() * log_prob).mean()

        # Preservation Penalty L_preserve = lambda_p * E[g * (1 - exp(-alpha))]
        g_val = info["g"]
        alpha_val = info["alpha"]
        preservation_penalty = (g_val * (1.0 - torch.exp(-alpha_val * exposure))).mean()

        total_loss = policy_loss + lambda_p * preservation_penalty
        total_loss.backward()
        opt_ctrl.step()

    # Evaluation on Seen Directions vs Never-Rewarded Held-Out Sector
    eval_env = AngularTaskEnvironment(seed=seed + 100_000, num_slots=num_slots)
    seen_metrics = evaluate_balanced_model(model, eval_env, split="seen_eval")
    held_out_metrics = evaluate_balanced_model(model, eval_env, split="held_out_sector")

    return {
        "seed": seed,
        "seen_directions": seen_metrics,
        "held_out_sector": held_out_metrics,
    }


# =============================================================================
# Summary Aggregation & Statistics
# =============================================================================

def stat_summary(values: List[float]) -> Dict[str, float]:
    """Calculate mean, std, median, min, max, and 95% confidence interval."""
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "ci95": 0.0, "min": 0.0, "max": 0.0}
    m = mean(values)
    s = stdev(values) if n > 1 else 0.0
    ci = 1.96 * s / math.sqrt(n) if n > 1 else 0.0
    return {
        "mean": m,
        "std": s,
        "median": median(values),
        "ci95_low": m - ci,
        "ci95_high": m + ci,
        "min": min(values),
        "max": max(values),
    }


def aggregate_benchmark(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    splits = ["seen_directions", "held_out_sector"]
    metrics = [
        "base_accuracy",
        "rsl_accuracy",
        "delta_transfer",
        "predicted_delta_identity",
        "P_W_to_R_marginal",
        "P_R_to_W_marginal",
        "P_W_to_R_conditional_c",
        "P_R_to_W_conditional_d",
        "P_R_to_R_marginal",
        "P_W_to_W_marginal",
        "mean_gate_g",
        "mean_alpha",
        "mean_survival_D",
        "delta_D",
    ]

    summary = {}
    for s in splits:
        summary[s] = {}
        for m in metrics:
            vals = [r[s][m] for r in runs]
            summary[s][m] = stat_summary(vals)

        # Positive transfer fraction: Delta_transfer > 0
        deltas = [r[s]["delta_transfer"] for r in runs]
        summary[s]["positive_transfer_rate"] = sum(1 for d in deltas if d > 0) / len(deltas)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Balanced Transition-Reward RSL Benchmark")
    parser.add_argument("--seeds", type=int, default=10, help="Number of deterministic seeds (>=10)")
    parser.add_argument("--steps", type=int, default=800, help="Training steps per seed")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lambda-p", type=float, default=0.02, help="Preservation penalty weight")
    parser.add_argument(
        "--output", default="results/balanced_rsl_report.json", help="JSON report path"
    )
    parser.add_argument(
        "--csv-output", default="results/balanced_rsl_seed_table.csv", help="CSV seed table path"
    )
    args = parser.parse_args()

    print(f"Running Balanced Transition RSL Experiment: {args.seeds} seeds, {args.steps} steps...")
    runs = []
    for seed in range(args.seeds):
        res = train_balanced_seed(
            seed=seed, steps=args.steps, batch_size=args.batch_size, lambda_p=args.lambda_p
        )
        runs.append(res)
        s_del = res["seen_directions"]["delta_transfer"]
        h_del = res["held_out_sector"]["delta_transfer"]
        print(
            f"Seed {seed:2d}: Seen Delta={s_del:+.4f} (Base={res['seen_directions']['base_accuracy']:.3f} -> RSL={res['seen_directions']['rsl_accuracy']:.3f}), "
            f"Held-Out Delta={h_del:+.4f} (Base={res['held_out_sector']['base_accuracy']:.3f} -> RSL={res['held_out_sector']['rsl_accuracy']:.3f})"
        )

    summary = aggregate_benchmark(runs)
    payload = {
        "title": "Balanced Transition-Focused Reinforcement Learning (Transition RL) Benchmark",
        "hypothesis": (
            "Balanced transition-based reinforcement learning can train an intervention-gated decay controller "
            "to rescue errors (W->R) while preserving already-correct states (R->R), transferring self-correction "
            "policy to never-rewarded held-out task sectors without target leakage."
        ),
        "config": vars(args),
        "runs": runs,
        "summary": summary,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nReport saved to {args.output}")

    # Write per-seed CSV table
    os.makedirs(os.path.dirname(args.csv_output) or ".", exist_ok=True)
    with open(args.csv_output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seed",
            "seen_base_acc",
            "seen_rsl_acc",
            "seen_delta",
            "seen_P_W_to_R",
            "seen_P_R_to_W",
            "held_out_base_acc",
            "held_out_rsl_acc",
            "held_out_delta",
            "held_out_P_W_to_R",
            "held_out_P_R_to_W",
            "mean_gate_g",
            "mean_survival_D",
        ])
        for r in runs:
            writer.writerow([
                r["seed"],
                f"{r['seen_directions']['base_accuracy']:.4f}",
                f"{r['seen_directions']['rsl_accuracy']:.4f}",
                f"{r['seen_directions']['delta_transfer']:+.4f}",
                f"{r['seen_directions']['P_W_to_R_marginal']:.4f}",
                f"{r['seen_directions']['P_R_to_W_marginal']:.4f}",
                f"{r['held_out_sector']['base_accuracy']:.4f}",
                f"{r['held_out_sector']['rsl_accuracy']:.4f}",
                f"{r['held_out_sector']['delta_transfer']:+.4f}",
                f"{r['held_out_sector']['P_W_to_R_marginal']:.4f}",
                f"{r['held_out_sector']['P_R_to_W_marginal']:.4f}",
                f"{r['held_out_sector']['mean_gate_g']:.4f}",
                f"{r['held_out_sector']['mean_survival_D']:.4f}",
            ])
    print(f"Seed table CSV saved to {args.csv_output}")


if __name__ == "__main__":
    main()

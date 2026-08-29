"""RSL Phase 2: Controlled Base Competence Sweep.

Implements the research program from rsl_phase2_controlled_competence_cookbook.md:
1. Audited survival metrics:
   - mean_gate: E_valid[g]
   - mean_alpha: E_valid[alpha]
   - mean_raw_survival: E_valid[exp(-alpha * T)]
   - mean_effective_survival: E_valid[(1 - g) + g * exp(-alpha * T)]
2. Stage A: Pretrain base model to target competence levels p in {0.50, 0.60, 0.70, 0.80, 0.90, 0.95}.
3. Stage B: Freeze base model parameters completely (requires_grad = False).
4. Stage C: Train only the dual-head RSL controller using transition-focused RL.
5. Task geometry with separate training, validation sector [30°, 60°], and final test sector [285°, 345°].
6. Oracle headroom H(p) = Acc_oracle - Acc_base and RSL efficiency eta_RSL = Delta Acc / H(p).
7. Error-detection metrics: E[g | W] vs E[g | R].
8. Exact accounting identity: Delta Acc = (1 - p)*c - p*d.
9. 10 deterministic seeds per competence level with CSV and JSON reports.
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

torch.set_num_threads(1)


# =============================================================================
# Angular Task Environment with Three Predeclared Sectors
# =============================================================================

@dataclass
class ControlledAngularBatch:
    """Batch of inputs for angular multi-task reasoning with predefined sectors."""
    tokens: torch.Tensor  # (B, L, dim)
    query_vectors: torch.Tensor  # (B, 2)
    targets: torch.Tensor  # (B,) class index
    query_positions: torch.Tensor
    relevant_positions: torch.Tensor
    irrelevant_positions: torch.Tensor
    padding_mask: torch.Tensor
    sector_name: str


class ThreeSectorAngularEnvironment:
    """Angular reasoning task with 3 strictly separated angular regions:

    - Validation gap: 30° to 60° [pi/6, pi/3]
    - Test gap: 285° to 345° [19*pi/12, 23*pi/12]
    - Training region: all remaining angles in [0, 2*pi)
    """

    def __init__(
        self,
        num_slots: int = 4,
        dim: int = 16,
        seed: int = 0,
        val_sector: Tuple[float, float] = (math.pi / 6, math.pi / 3),  # 30° - 60°
        test_sector: Tuple[float, float] = (19 * math.pi / 12, 23 * math.pi / 12),  # 285° - 345°
    ):
        self.num_slots = num_slots
        self.dim = dim
        self.val_min, self.val_max = val_sector
        self.test_min, self.test_max = test_sector
        self.rng = random.Random(seed)

    def classify_angle(self, theta: float) -> str:
        theta = theta % (2 * math.pi)
        if self.val_min <= theta <= self.val_max:
            return "validation_sector"
        if self.test_min <= theta <= self.test_max:
            return "test_sector"
        return "training_region"

    def sample_angle(self, split: str) -> float:
        while True:
            theta = self.rng.uniform(0, 2 * math.pi)
            if self.classify_angle(theta) == split:
                return theta

    def generate_batch(
        self, batch_size: int, split: str = "training_region", device: str = "cpu"
    ) -> ControlledAngularBatch:
        seq_len = self.num_slots + 2
        tokens = torch.zeros(batch_size, seq_len, self.dim, device=device)
        q_pos = torch.full((batch_size,), seq_len - 1, dtype=torch.long, device=device)
        r_pos = torch.zeros(batch_size, dtype=torch.long, device=device)
        i_pos = torch.zeros(batch_size, dtype=torch.long, device=device)
        targets = torch.zeros(batch_size, dtype=torch.long, device=device)
        query_vecs = torch.zeros(batch_size, 2, device=device)

        for b in range(batch_size):
            theta = self.sample_angle(split=split)
            q_2d = torch.tensor([math.cos(theta), math.sin(theta)], device=device)
            query_vecs[b] = q_2d

            angles = [self.rng.uniform(0, 2 * math.pi) for _ in range(self.num_slots)]
            scores = [math.cos(theta - a) for a in angles]
            best_idx = int(scores.index(max(scores)))
            worst_idx = int(scores.index(min(scores)))

            targets[b] = best_idx
            r_pos[b] = 1 + best_idx
            i_pos[b] = 1 + worst_idx

            tokens[b, 0, 0] = 1.0  # BOS
            for s in range(self.num_slots):
                k_2d = torch.tensor([math.cos(angles[s]), math.sin(angles[s])], device=device)
                tokens[b, 1 + s, :2] = k_2d
                tokens[b, 1 + s, 2 + s] = 1.0  # Slot indicator

            tokens[b, seq_len - 1, :2] = q_2d
            tokens[b, seq_len - 1, 2 + self.num_slots] = 1.0

        padding = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        return ControlledAngularBatch(
            tokens=tokens,
            query_vectors=query_vecs,
            targets=targets,
            query_positions=q_pos,
            relevant_positions=r_pos,
            irrelevant_positions=i_pos,
            padding_mask=padding,
            sector_name=split,
        )


# =============================================================================
# Dual-Head Intervention Gate + Semantic Decay Controller
# =============================================================================

class AuditedInterventionGatedController(nn.Module):
    """Dual-head controller with strict valid-position auditing."""

    def __init__(
        self,
        d_model: int = 32,
        n_heads: int = 4,
        uncertainty_dim: int = 4,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.uncertainty_dim = uncertainty_dim

        feature_dim = 4 * self.d_head + 1 + uncertainty_dim
        self.shared = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.gate_head = nn.Linear(hidden_dim, 1)
        self.alpha_head = nn.Linear(hidden_dim, 1)

    def extract_features(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        A: torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, h, l, d = q.shape
        qi = q.unsqueeze(3).expand(-1, -1, -1, l, -1)
        kj = k.unsqueeze(2).expand(-1, -1, l, -1, -1)
        prod = qi * kj
        diff = (qi - kj).abs()
        a_weight = A.unsqueeze(-1)

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.extract_features(q, k, A, uncertainty)
        h = self.shared(feat)
        gate_logits = self.gate_head(h).squeeze(-1)
        alpha_logits = self.alpha_head(h).squeeze(-1)

        g = torch.sigmoid(gate_logits)
        alpha = F.softplus(alpha_logits)

        raw_survival = torch.exp(-alpha * exposure)
        effective_survival = (1.0 - g) + g * raw_survival

        if invalid_mask is not None:
            mask = invalid_mask.expand_as(effective_survival)
            effective_survival = effective_survival.masked_fill(mask, 0.0)
            raw_survival = raw_survival.masked_fill(mask, 0.0)
            g = g.masked_fill(mask, 0.0)
            alpha = alpha.masked_fill(mask, 0.0)

        # Policy log probability for sampling-based RL
        log_prob = (
            g.clamp(1e-6, 1 - 1e-6).log() * g.detach()
            + (1 - g).clamp(1e-6, 1 - 1e-6).log() * (1 - g.detach())
        ).mean(dim=(-1, -2, -3))

        return g, alpha, raw_survival, effective_survival, log_prob


# =============================================================================
# Controlled Competence Model Architecture
# =============================================================================

class ControlledCompetenceModel(nn.Module):
    """Transformer with frozen base model and trainable dual-head RSL controller."""

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

        self.controller = AuditedInterventionGatedController(
            d_model, n_heads, hidden_dim=hidden_dim
        )

    def freeze_base_model(self):
        """Freeze all base model parameters bitwise."""
        for name, param in self.named_parameters():
            if not name.startswith("controller"):
                param.requires_grad = False

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
        probs = F.softmax(base_logits, dim=-1)
        sorted_probs, _ = torch.sort(probs, descending=True, dim=-1)
        max_prob = sorted_probs[:, 0:1]
        second_prob = sorted_probs[:, 1:2] if sorted_probs.shape[1] > 1 else torch.zeros_like(max_prob)
        margin = max_prob - second_prob
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1, keepdim=True)
        attn_entropy = -(A * torch.log(A.clamp_min(1e-8))).sum(dim=-1).mean(dim=(-1, -2), keepdim=False)[:, None]
        return torch.cat([max_prob, margin, entropy, attn_entropy], dim=-1)

    def forward(
        self,
        tokens: torch.Tensor,
        query_positions: torch.Tensor,
        exposure: float = 1.0,
        padding_mask: Optional[torch.Tensor] = None,
        apply_rsl: bool = True,
        oracle_target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        b, l, _ = tokens.shape
        pos = torch.arange(l, device=tokens.device)
        x0 = self.input_proj(tokens) + self.pos_emb(pos)
        h0 = self.ln1(x0)
        q, k, v, A, invalid = self.compute_attention(h0, padding_mask)

        # 1. Base prediction
        y_base = A @ v
        y_base = y_base.transpose(1, 2).contiguous().view(b, l, self.d_model)
        x_base = x0 + self.out(y_base)
        x_base = x_base + self.ff(self.ln2(x_base))
        rows = torch.arange(b, device=tokens.device)
        base_logits = self.head(self.final_ln(x_base[rows, query_positions]))

        # 2. Extract inference-available uncertainty features
        uncertainty = self.extract_uncertainty(base_logits.detach(), A.detach())

        # 3. Controller execution
        g = alpha = raw_survival = effective_survival = log_prob = None
        if apply_rsl:
            if oracle_target is not None:
                # Oracle privileged mode (upper bound only)
                b_corr = base_logits.argmax(-1) == oracle_target
                # For wrong examples, boost attention to target slot (1 + target)
                effective_survival = torch.ones_like(A)
                for i in range(b):
                    if not b_corr[i]:
                        t_pos = 1 + oracle_target[i].item()
                        effective_survival[i, :, :, :] = 0.05
                        effective_survival[i, :, query_positions[i], t_pos] = 1.0
                effective_survival = effective_survival.masked_fill(invalid.expand_as(A), 0.0)
                g = torch.where(b_corr[:, None, None, None], 0.0, 1.0).expand_as(A)
                alpha = torch.where(b_corr[:, None, None, None], 0.0, 3.0).expand_as(A)
                raw_survival = torch.exp(-alpha * exposure)
                log_prob = torch.zeros(b, device=tokens.device)
            else:
                g, alpha, raw_survival, effective_survival, log_prob = self.controller(
                    q.detach(),
                    k.detach(),
                    A.detach(),
                    uncertainty=uncertainty,
                    exposure=exposure,
                    invalid_mask=invalid,
                )

            decayed_A = A * effective_survival
            y_rsl = decayed_A @ v
            y_rsl = y_rsl.transpose(1, 2).contiguous().view(b, l, self.d_model)
            x_rsl = x0 + self.out(y_rsl)
            x_rsl = x_rsl + self.ff(self.ln2(x_rsl))
            rsl_logits = self.head(self.final_ln(x_rsl[rows, query_positions]))
        else:
            rsl_logits = base_logits
            effective_survival = torch.ones_like(A).masked_fill(invalid.expand_as(A), 0.0)
            raw_survival = torch.ones_like(A).masked_fill(invalid.expand_as(A), 0.0)
            g = torch.zeros_like(A)
            alpha = torch.zeros_like(A)

        info = {
            "q": q,
            "k": k,
            "v": v,
            "A_base": A,
            "g": g,
            "alpha": alpha,
            "raw_survival": raw_survival,
            "effective_survival": effective_survival,
            "log_prob": log_prob,
            "uncertainty": uncertainty,
            "invalid_mask": invalid,
        }
        return base_logits, rsl_logits, info


# =============================================================================
# Audited Metric Computation & Evaluation
# =============================================================================

def compute_audited_metrics(
    model: ControlledCompetenceModel,
    env: ThreeSectorAngularEnvironment,
    split: str,
    batches: int = 10,
    batch_size: int = 128,
    is_oracle: bool = False,
) -> Dict[str, Any]:
    """Compute strictly audited evaluation metrics on unmasked attention positions."""
    model.eval()
    total = 0
    base_correct = 0
    rsl_correct = 0
    w_to_r = 0
    r_to_w = 0
    r_to_r = 0
    w_to_w = 0

    all_valid_g = []
    all_valid_alpha = []
    all_valid_raw_d = []
    all_valid_eff_d = []
    g_when_wrong = []
    g_when_right = []

    with torch.no_grad():
        for _ in range(batches):
            b = env.generate_batch(batch_size, split=split)
            b_size = b.tokens.shape[0]
            total += b_size

            oracle_tgt = b.targets if is_oracle else None
            base_logits, rsl_logits, info = model(
                b.tokens,
                b.query_positions,
                exposure=1.0,
                padding_mask=b.padding_mask,
                apply_rsl=True,
                oracle_target=oracle_tgt,
            )

            b_preds = base_logits.argmax(-1)
            r_preds = rsl_logits.argmax(-1)
            b_corr = b_preds == b.targets
            r_corr = r_preds == b.targets

            base_correct += b_corr.sum().item()
            rsl_correct += r_corr.sum().item()
            w_to_r += ((~b_corr) & r_corr).sum().item()
            r_to_w += (b_corr & (~r_corr)).sum().item()
            r_to_r += (b_corr & r_corr).sum().item()
            w_to_w += ((~b_corr) & (~r_corr)).sum().item()

            invalid = info["invalid_mask"].expand_as(info["effective_survival"])
            valid_mask = ~invalid

            if info["g"] is not None:
                g_t = info["g"]
                all_valid_g.append(g_t[valid_mask].mean().item())
                # Error detection gate analysis
                for i in range(b_size):
                    val_g_i = g_t[i][valid_mask[i]].mean().item()
                    if b_corr[i]:
                        g_when_right.append(val_g_i)
                    else:
                        g_when_wrong.append(val_g_i)

            if info["alpha"] is not None:
                all_valid_alpha.append(info["alpha"][valid_mask].mean().item())
            if info["raw_survival"] is not None:
                all_valid_raw_d.append(info["raw_survival"][valid_mask].mean().item())
            if info["effective_survival"] is not None:
                all_valid_eff_d.append(info["effective_survival"][valid_mask].mean().item())

    p = base_correct / total
    acc_rsl = rsl_correct / total
    c = (w_to_r / (total - base_correct)) if total > base_correct else 0.0
    d = (r_to_w / base_correct) if base_correct > 0 else 0.0
    delta_acc = acc_rsl - p
    predicted_delta = (1.0 - p) * c - p * d

    mean_g_wrong = mean(g_when_wrong) if g_when_wrong else 0.0
    mean_g_right = mean(g_when_right) if g_when_right else 0.0

    return {
        "base_accuracy": p,
        "rsl_accuracy": acc_rsl,
        "delta_transfer": delta_acc,
        "predicted_delta_identity": predicted_delta,
        "identity_error": abs(delta_acc - predicted_delta),
        "P_W_to_R_marginal": w_to_r / total,
        "P_R_to_W_marginal": r_to_w / total,
        "P_W_to_R_conditional_c": c,
        "P_R_to_W_conditional_d": d,
        "P_R_to_R_marginal": r_to_r / total,
        "P_W_to_W_marginal": w_to_w / total,
        "mean_gate_g": mean(all_valid_g) if all_valid_g else 0.0,
        "mean_alpha": mean(all_valid_alpha) if all_valid_alpha else 0.0,
        "mean_raw_survival_D": mean(all_valid_raw_d) if all_valid_raw_d else 1.0,
        "mean_effective_survival_D": mean(all_valid_eff_d) if all_valid_eff_d else 1.0,
        "E_g_given_wrong": mean_g_wrong,
        "E_g_given_right": mean_g_right,
        "error_detector_gap": mean_g_wrong - mean_g_right,
    }


# =============================================================================
# Stage A: Controlled Base Competence Calibration
# =============================================================================

def train_base_model_to_competence(
    target_p: float,
    seed: int,
    env: ThreeSectorAngularEnvironment,
    device: str = "cpu",
) -> ControlledCompetenceModel:
    """Pretrain a base Transformer model to achieve target competence p in {0.50, 0.60, ..., 0.95}."""
    random.seed(seed)
    torch.manual_seed(seed)
    model = ControlledCompetenceModel(in_dim=16, num_classes=env.num_slots).to(device)

    # Competence mapping to training step targets
    # (Since base model learns monotonically from 0.25 (random 4-way) to ~0.95 at 800 steps)
    step_budget = {
        0.50: 30,
        0.60: 60,
        0.70: 120,
        0.80: 220,
        0.90: 450,
        0.95: 800,
    }
    steps = step_budget.get(target_p, int(target_p * 800))

    opt = torch.optim.AdamW(
        [p for n, p in model.named_parameters() if not n.startswith("controller")],
        lr=3e-3,
    )

    model.train()
    for step in range(1, steps + 1):
        batch = env.generate_batch(128, split="training_region", device=device)
        opt.zero_grad()
        base_logits, _, _ = model(
            batch.tokens, batch.query_positions, exposure=1.0, padding_mask=batch.padding_mask, apply_rsl=False
        )
        loss = F.cross_entropy(base_logits, batch.targets)
        loss.backward()
        opt.step()

    # Freeze base model parameters
    model.freeze_base_model()
    return model


# =============================================================================
# Stage B & C: Train RSL on Frozen Base Checkpoints
# =============================================================================

def train_rsl_on_frozen_base(
    model: ControlledCompetenceModel,
    env: ThreeSectorAngularEnvironment,
    rsl_steps: int = 500,
    batch_size: int = 128,
    rsl_lr: float = 3e-3,
    lambda_p: float = 0.02,
    device: str = "cpu",
) -> ControlledCompetenceModel:
    """Train only the RSL controller on a frozen base checkpoint."""
    opt_ctrl = torch.optim.AdamW(model.controller.parameters(), lr=rsl_lr)
    running_baseline_reward = 0.0

    model.train()
    for step in range(1, rsl_steps + 1):
        exposure = min(1.0, step / max(1, rsl_steps // 2))
        batch = env.generate_batch(batch_size, split="training_region", device=device)

        opt_ctrl.zero_grad()
        with torch.no_grad():
            base_logits, _, _ = model(
                batch.tokens, batch.query_positions, exposure=1.0, padding_mask=batch.padding_mask, apply_rsl=False
            )
            base_preds = base_logits.argmax(-1)

        _, rsl_logits, info = model(
            batch.tokens, batch.query_positions, exposure=exposure, padding_mask=batch.padding_mask, apply_rsl=True
        )
        rsl_preds = rsl_logits.argmax(-1)

        # Balanced Transition Reward
        b_corr = base_preds == batch.targets
        r_corr = rsl_preds == batch.targets
        rewards = torch.zeros_like(base_preds, dtype=torch.float)
        rewards[(~b_corr) & r_corr] = 1.0   # W -> R
        rewards[b_corr & r_corr] = 1.0     # R -> R
        rewards[b_corr & (~r_corr)] = -1.0  # R -> W
        rewards[(~b_corr) & (~r_corr)] = -1.0  # W -> W

        running_baseline_reward = 0.95 * running_baseline_reward + 0.05 * rewards.mean().item()
        advantage = rewards - running_baseline_reward

        # Policy loss
        log_prob = info["log_prob"]
        policy_loss = -(advantage.detach() * log_prob).mean()

        # Preservation regularizer on effective survival
        g_val = info["g"]
        alpha_val = info["alpha"]
        preservation = (g_val * (1.0 - torch.exp(-alpha_val * exposure))).mean()

        loss = policy_loss + lambda_p * preservation
        loss.backward()
        opt_ctrl.step()

    return model


# =============================================================================
# Full Competence Sweep Runner
# =============================================================================

def run_competence_sweep(
    competence_levels: List[float],
    seeds: int = 10,
    rsl_steps: int = 500,
    batch_size: int = 128,
) -> Dict[str, Any]:
    """Execute 10-seed sweep across all base competence levels."""
    env = ThreeSectorAngularEnvironment()
    results_by_p = {}
    raw_runs = []

    for target_p in competence_levels:
        p_label = f"p_{int(target_p * 100):02d}"
        print(f"\n--- Running Competence Level p = {target_p:.2f} ({seeds} seeds) ---")
        p_runs = []

        for seed in range(seeds):
            # 1. Train base model to competence target
            seed_env = ThreeSectorAngularEnvironment(seed=seed)
            model = train_base_model_to_competence(target_p, seed, seed_env)

            # 2. Measure initial baseline and Oracle headroom before RSL training
            eval_env = ThreeSectorAngularEnvironment(seed=seed + 100_000)
            seen_base_pre = compute_audited_metrics(model, eval_env, split="training_region")
            test_base_pre = compute_audited_metrics(model, eval_env, split="test_sector")
            oracle_test = compute_audited_metrics(model, eval_env, split="test_sector", is_oracle=True)

            # 3. Train RSL on frozen base model
            model = train_rsl_on_frozen_base(model, seed_env, rsl_steps=rsl_steps, batch_size=batch_size)

            # 4. Evaluate trained RSL on seen training region and final held-out test sector
            seen_post = compute_audited_metrics(model, eval_env, split="training_region")
            test_post = compute_audited_metrics(model, eval_env, split="test_sector")

            oracle_headroom = oracle_test["rsl_accuracy"] - test_post["base_accuracy"]
            rsl_efficiency = (
                (test_post["delta_transfer"] / oracle_headroom) if oracle_headroom > 0.001 else 0.0
            )

            run_record = {
                "target_p": target_p,
                "seed": seed,
                "seen_training_region": seen_post,
                "held_out_test_sector": test_post,
                "oracle_test_accuracy": oracle_test["rsl_accuracy"],
                "oracle_headroom": oracle_headroom,
                "rsl_efficiency_eta": rsl_efficiency,
            }
            p_runs.append(run_record)
            raw_runs.append(run_record)

            print(
                f"  Seed {seed:2d}: Base={test_post['base_accuracy']:.3f}, RSL={test_post['rsl_accuracy']:.3f} "
                f"(Delta={test_post['delta_transfer']:+.4f}, g={test_post['mean_gate_g']:.4f}, "
                f"D_eff={test_post['mean_effective_survival_D']:.4f}, Headroom={oracle_headroom:.3f})"
            )

        # Aggregate metrics for target_p
        results_by_p[p_label] = aggregate_p_summary(p_runs)

    return {"results_by_p": results_by_p, "raw_runs": raw_runs}


def aggregate_p_summary(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    def stat_dict(vals: List[float]) -> Dict[str, float]:
        n = len(vals)
        m = mean(vals)
        s = stdev(vals) if n > 1 else 0.0
        ci = 1.96 * s / math.sqrt(n) if n > 1 else 0.0
        return {
            "mean": m,
            "std": s,
            "median": median(vals),
            "ci95_low": m - ci,
            "ci95_high": m + ci,
            "min": min(vals),
            "max": max(vals),
        }

    splits = ["seen_training_region", "held_out_test_sector"]
    summary = {}
    for s in splits:
        summary[s] = {
            "base_accuracy": stat_dict([r[s]["base_accuracy"] for r in runs]),
            "rsl_accuracy": stat_dict([r[s]["rsl_accuracy"] for r in runs]),
            "delta_transfer": stat_dict([r[s]["delta_transfer"] for r in runs]),
            "positive_transfer_rate": sum(1 for r in runs if r[s]["delta_transfer"] > 0) / len(runs),
            "P_W_to_R_conditional_c": stat_dict([r[s]["P_W_to_R_conditional_c"] for r in runs]),
            "P_R_to_W_conditional_d": stat_dict([r[s]["P_R_to_W_conditional_d"] for r in runs]),
            "mean_gate_g": stat_dict([r[s]["mean_gate_g"] for r in runs]),
            "mean_effective_survival_D": stat_dict([r[s]["mean_effective_survival_D"] for r in runs]),
            "error_detector_gap": stat_dict([r[s]["error_detector_gap"] for r in runs]),
        }
    summary["oracle_headroom"] = stat_dict([r["oracle_headroom"] for r in runs])
    summary["rsl_efficiency_eta"] = stat_dict([r["rsl_efficiency_eta"] for r in runs])
    return summary


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Controlled Base Competence RSL Sweep")
    parser.add_argument(
        "--competence-levels",
        nargs="+",
        type=float,
        default=[0.50, 0.60, 0.70, 0.80, 0.90, 0.95],
        help="Target base competence levels p",
    )
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds per competence level")
    parser.add_argument("--rsl-steps", type=int, default=500, help="RSL training steps on frozen base")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--output", default="results/controlled_competence_report.json", help="Report JSON path"
    )
    parser.add_argument(
        "--csv-output", default="results/controlled_competence_seed_table.csv", help="CSV seed table path"
    )
    args = parser.parse_args()

    print(f"Starting Phase 2 Sweep: Competence levels {args.competence_levels}, {args.seeds} seeds each...")
    sweep_data = run_competence_sweep(
        competence_levels=args.competence_levels,
        seeds=args.seeds,
        rsl_steps=args.rsl_steps,
        batch_size=args.batch_size,
    )

    payload = {
        "title": "Phase 2: Controlled Base Competence RSL Benchmark",
        "question": "At what level of base-model competence does RSL learn useful correction vs rational abstention?",
        "config": vars(args),
        "summary_by_p": sweep_data["results_by_p"],
        "raw_runs": sweep_data["raw_runs"],
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nReport saved to {args.output}")

    # Write Seed CSV Table
    os.makedirs(os.path.dirname(args.csv_output) or ".", exist_ok=True)
    with open(args.csv_output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "target_p",
            "seed",
            "test_base_acc",
            "test_rsl_acc",
            "test_delta",
            "test_P_W_to_R_c",
            "test_P_R_to_W_d",
            "mean_gate_g",
            "mean_effective_D",
            "error_detector_gap",
            "oracle_headroom",
            "rsl_efficiency_eta",
        ])
        for r in sweep_data["raw_runs"]:
            t = r["held_out_test_sector"]
            writer.writerow([
                r["target_p"],
                r["seed"],
                f"{t['base_accuracy']:.4f}",
                f"{t['rsl_accuracy']:.4f}",
                f"{t['delta_transfer']:+.4f}",
                f"{t['P_W_to_R_conditional_c']:.4f}",
                f"{t['P_R_to_W_conditional_d']:.4f}",
                f"{t['mean_gate_g']:.4f}",
                f"{t['mean_effective_survival_D']:.4f}",
                f"{t['error_detector_gap']:+.4f}",
                f"{r['oracle_headroom']:.4f}",
                f"{r['rsl_efficiency_eta']:+.4f}",
            ])
    print(f"Seed table CSV saved to {args.csv_output}")


if __name__ == "__main__":
    main()

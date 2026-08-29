"""Gradient-Pattern Recursive Self-Learning (RSL) Experiment.

Tests whether a secondary controller can observe backpropagation correction patterns
during supervised training, learn recurring patterns of those corrections, and later
predict useful information-retention/decay decisions on unseen examples without
access to labels or gradients at inference time.
"""

import argparse
import copy
import json
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import BatchFactory, SHIFTS, VOCAB_SIZE
from data.selective_recall import Batch
from sadt.losses import retention_loss
from sadt.model import TinySADT

# Deterministic tiny workloads
torch.set_num_threads(1)

CASES = tuple(x for x in SHIFTS if x != "iid")


class GradientRSLController(nn.Module):
    """Secondary controller predicting semantic decay from inference-time activations.

    At inference time, it receives ONLY legitimate representations (q, k, A, h)
    and predicts alpha >= 0 (and D = exp(-alpha * T)).
    No labels, no loss, and no test gradients are ever provided.
    """

    def __init__(self, d_model: int = 32, n_heads: int = 4, hidden_dim: int = 32):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        # Features: [q_i, k_j, q_i * k_j, |q_i - k_j|] = 4 * d_head
        # Plus pairwise attention weight A_ij (1) = 4 * d_head + 1
        feature_dim = 4 * self.d_head + 1
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def extract_features(
        self, q: torch.Tensor, k: torch.Tensor, A: torch.Tensor
    ) -> torch.Tensor:
        """Extract pairwise features legitimately available during inference.

        Args:
            q: Query tensor (B, H, L, d_head)
            k: Key tensor (B, H, L, d_head)
            A: Softmax attention weights (B, H, L, L)
        """
        b, h, l, d = q.shape
        qi = q.unsqueeze(3).expand(-1, -1, -1, l, -1)  # (B, H, L, L, d)
        kj = k.unsqueeze(2).expand(-1, -1, l, -1, -1)  # (B, H, L, L, d)
        prod = qi * kj
        diff = (qi - kj).abs()
        a_weight = A.unsqueeze(-1)  # (B, H, L, L, 1)
        feat = torch.cat([qi, kj, prod, diff, a_weight], dim=-1)
        return feat

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        A: torch.Tensor,
        exposure: float = 1.0,
        invalid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict decay alpha and survival factor D.

        Returns:
            alpha: Predicted decay rate >= 0, shape (B, H, L, L)
            survival: D = exp(-alpha * exposure) in [0, 1], shape (B, H, L, L)
        """
        feat = self.extract_features(q, k, A)
        raw_logits = self.net(feat).squeeze(-1)  # (B, H, L, L)
        # Logit > 0 means retain (small alpha, large D), Logit < 0 means suppress (large alpha, small D)
        # alpha = softplus(-raw_logits) guarantees alpha >= 0
        alpha = F.softplus(-raw_logits)
        # Apply exposure: D = exp(-alpha * exposure)
        survival = torch.exp(-alpha * exposure)
        if invalid_mask is not None:
            survival = survival.masked_fill(invalid_mask.expand_as(survival), 0.0)
        return alpha, survival


class GradientRSLModel(nn.Module):
    """Transformer with Gradient-Pattern RSL controller and recursive reasoning support."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        max_len: int = 32,
        d_model: int = 32,
        n_heads: int = 4,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.token = nn.Embedding(vocab_size, d_model)
        self.position = nn.Embedding(max_len, d_model)
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
        self.head = nn.Linear(d_model, vocab_size)

        self.controller = GradientRSLController(d_model, n_heads, hidden_dim)

    def compute_attention(
        self, h: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute base scaled dot-product attention without decay."""
        b, l, d = h.shape
        q, k, v = self.qkv(h).chunk(3, -1)
        shape = (b, l, self.n_heads, self.d_head)
        q, k, v = [z.view(shape).transpose(1, 2) for z in (q, k, v)]
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        causal = torch.triu(torch.ones(l, l, dtype=torch.bool, device=h.device), 1)
        invalid = causal.view(1, 1, l, l)
        if padding_mask is not None:
            invalid = invalid | padding_mask[:, None, None, :]
        A = F.softmax(scores.masked_fill(invalid, -torch.inf), dim=-1)
        # Avoid NaN for fully masked positions
        A = A.masked_fill(invalid, 0.0)
        return q, k, v, A, invalid

    def forward_pass(
        self,
        tokens: torch.Tensor,
        query_positions: torch.Tensor,
        exposure: float = 1.0,
        padding_mask: Optional[torch.Tensor] = None,
        controller_override: Optional[Any] = None,
        oracle_survival: Optional[torch.Tensor] = None,
        decay_mode: str = "rsl",  # "none", "rsl", "random", "fixed", "oracle"
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Single-pass forward execution."""
        b, l = tokens.shape
        pos = torch.arange(l, device=tokens.device)
        x = self.token(tokens) + self.position(pos)
        h = self.ln1(x)
        q, k, v, A, invalid = self.compute_attention(h, padding_mask)

        alpha = survival = None
        if decay_mode == "none":
            survival = torch.ones_like(A).masked_fill(invalid, 0.0)
            alpha = torch.zeros_like(A)
            decayed_A = A
        elif decay_mode == "fixed":
            alpha = torch.ones_like(A)
            survival = torch.exp(-alpha * exposure).masked_fill(invalid, 0.0)
            decayed_A = A * survival
        elif decay_mode == "random":
            alpha = torch.rand(A.shape, device=tokens.device, generator=generator) * 3.0
            survival = torch.exp(-alpha * exposure).masked_fill(invalid, 0.0)
            decayed_A = A * survival
        elif decay_mode == "oracle":
            if oracle_survival is None:
                raise ValueError("oracle decay mode requires oracle_survival tensor")
            survival = oracle_survival.masked_fill(invalid, 0.0)
            alpha = -torch.log(survival.clamp_min(1e-6)) / max(exposure, 1e-4)
            decayed_A = A * survival
        elif decay_mode == "rsl":
            ctrl = controller_override if controller_override is not None else self.controller
            alpha, survival = ctrl(q, k, A, exposure, invalid)
            decayed_A = A * survival
        else:
            raise ValueError(f"Unknown decay_mode: {decay_mode}")

        y = decayed_A @ v
        y = y.transpose(1, 2).contiguous().view(b, l, self.d_model)
        x = x + self.out(y)
        x = x + self.ff(self.ln2(x))
        rows = torch.arange(b, device=tokens.device)
        logits = self.head(self.final_ln(x[rows, query_positions]))

        info = {
            "q": q,
            "k": k,
            "v": v,
            "A_base": A,
            "alpha": alpha,
            "survival": survival,
            "decayed_A": decayed_A,
            "hidden": x,
            "invalid": invalid,
        }
        return logits, info

    def recursive_forward(
        self,
        tokens: torch.Tensor,
        query_positions: torch.Tensor,
        k_passes: int = 1,
        exposure: float = 1.0,
        padding_mask: Optional[torch.Tensor] = None,
        controller_override: Optional[Any] = None,
        decay_mode: str = "rsl",
        generator: Optional[torch.Generator] = None,
    ) -> List[Tuple[torch.Tensor, Dict[str, Any]]]:
        """Perform k recursive reasoning passes over internal representations.

        For pass k:
            h^{(k)} -> R_phi -> alpha^{(k)} -> D^{(k)} = exp(-alpha^{(k)} * exposure)
            resulting in filtered state for next pass.
        """
        b, l = tokens.shape
        pos = torch.arange(l, device=tokens.device)
        x = self.token(tokens) + self.position(pos)
        rows = torch.arange(b, device=tokens.device)

        pass_results = []
        for p in range(1, k_passes + 1):
            h = self.ln1(x)
            q, k, v, A, invalid = self.compute_attention(h, padding_mask)

            if decay_mode == "none":
                survival = torch.ones_like(A).masked_fill(invalid, 0.0)
                alpha = torch.zeros_like(A)
                decayed_A = A
            elif decay_mode == "fixed":
                alpha = torch.ones_like(A)
                survival = torch.exp(-alpha * exposure).masked_fill(invalid, 0.0)
                decayed_A = A * survival
            elif decay_mode == "random":
                alpha = torch.rand(A.shape, device=tokens.device, generator=generator) * 3.0
                survival = torch.exp(-alpha * exposure).masked_fill(invalid, 0.0)
                decayed_A = A * survival
            elif decay_mode == "rsl":
                ctrl = controller_override if controller_override is not None else self.controller
                alpha, survival = ctrl(q, k, A, exposure, invalid)
                decayed_A = A * survival
            else:
                raise ValueError(f"Unsupported recursive decay_mode: {decay_mode}")

            y = decayed_A @ v
            y = y.transpose(1, 2).contiguous().view(b, l, self.d_model)
            x = x + self.out(y)
            x = x + self.ff(self.ln2(x))
            logits = self.head(self.final_ln(x[rows, query_positions]))

            info = {
                "pass": p,
                "q": q,
                "k": k,
                "v": v,
                "A_base": A,
                "alpha": alpha,
                "survival": survival,
                "decayed_A": decayed_A,
                "hidden": x,
                "invalid": invalid,
            }
            pass_results.append((logits, info))

        return pass_results


# =============================================================================
# Teacher Signal Construction & Extraction
# =============================================================================

@dataclass
class TeacherSignals:
    """Rich gradient and counterfactual descriptors from supervised backpropagation."""
    target_survival: torch.Tensor  # Continuous D* in [0, 1], shape (B, H, L, L)
    target_class: torch.Tensor  # Discrete: 0=suppress, 1=neutral, 2=retain
    sensitivity: torch.Tensor  # s_ij = - (dL/dD), shape (B, H, L, L)
    grad_norm: torch.Tensor  # Gradient norm per token/query, shape (B, L)
    cosine_sim: torch.Tensor  # Cosine sim between h and grad_h, shape (B, L)
    delta_loss_rel: torch.Tensor  # Delta L when suppressing relevant value, shape (B,)
    delta_loss_irr: torch.Tensor  # Delta L when suppressing irrelevant value, shape (B,)


def compute_teacher_signals(
    model: GradientRSLModel,
    batch: Batch,
    scale: float = 4.0,
    compute_interventions: bool = True,
) -> TeacherSignals:
    r"""Derive compact, principled gradient correction descriptors and targets.

    Measures whether backpropagation wants each attention connection retained or suppressed:
        s_{ij} = - dL / dD_{ij} = - (dL / d \widetilde{A}_{ij}) * A_{ij}
    If s_{ij} > 0, retention reduces loss -> useful (retain, D* -> 1, alpha* -> 0).
    If s_{ij} < 0, retention increases loss -> harmful (suppress, D* -> 0, alpha* >> 0).
    """
    b, l = batch.tokens.shape
    pos = torch.arange(l, device=batch.tokens.device)
    x = model.token(batch.tokens) + model.position(pos)
    h = model.ln1(x)
    h.retain_grad()

    q, k, v, A, invalid = model.compute_attention(h, batch.padding_mask)

    # Initialize survival variable D with requires_grad=True to obtain exact dL/dD
    D = torch.ones_like(A, requires_grad=True)
    decayed_A = A * D

    y = decayed_A @ v
    y = y.transpose(1, 2).contiguous().view(b, l, model.d_model)
    x_out = x + model.out(y)
    x_out = x_out + model.ff(model.ln2(x_out))
    rows = torch.arange(b, device=batch.tokens.device)
    logits = model.head(model.final_ln(x_out[rows, batch.query_positions]))

    loss = F.cross_entropy(logits, batch.targets)
    # Compute gradients with graph retention for analysis
    loss.backward(retain_graph=True)

    # Gradient of loss with respect to survival factor D
    grad_D = D.grad.detach() if D.grad is not None else torch.zeros_like(A)
    grad_h = h.grad.detach() if h.grad is not None else torch.zeros_like(h)

    # Sensitivity s_ij: positive means increasing D decreases loss (useful!)
    sensitivity = -grad_D  # (B, H, L, L)
    sensitivity = sensitivity.masked_fill(invalid, 0.0)

    # Continuous target D* in [0, 1] using scaled sigmoid
    target_survival = torch.sigmoid(scale * sensitivity).masked_fill(invalid, 0.0)

    # Discrete classification target: 0=suppress (s < -0.1), 1=neutral (|s| <= 0.1), 2=retain (s > 0.1)
    target_class = torch.ones_like(sensitivity, dtype=torch.long)
    target_class[sensitivity > 0.1] = 2
    target_class[sensitivity < -0.1] = 0

    # Gradient norm and cosine similarity descriptors
    grad_norm = grad_h.norm(dim=-1)  # (B, L)
    h_norm = h.norm(dim=-1).clamp_min(1e-8)
    g_norm = grad_norm.clamp_min(1e-8)
    cosine_sim = (h * grad_h).sum(dim=-1) / (h_norm * g_norm)  # (B, L)

    delta_loss_rel = torch.zeros(b, device=batch.tokens.device)
    delta_loss_irr = torch.zeros(b, device=batch.tokens.device)

    if compute_interventions:
        with torch.no_grad():
            base_loss = F.cross_entropy(logits, batch.targets, reduction="none")

            # Intervene: zero out relevant value connection
            D_no_rel = torch.ones_like(A)
            D_no_rel[rows, :, batch.query_positions, batch.relevant_positions] = 0.0
            y_no_rel = (A * D_no_rel) @ v
            y_no_rel = y_no_rel.transpose(1, 2).contiguous().view(b, l, model.d_model)
            x_no_rel = x + model.out(y_no_rel)
            x_no_rel = x_no_rel + model.ff(model.ln2(x_no_rel))
            logits_no_rel = model.head(model.final_ln(x_no_rel[rows, batch.query_positions]))
            loss_no_rel = F.cross_entropy(logits_no_rel, batch.targets, reduction="none")
            delta_loss_rel = loss_no_rel - base_loss  # > 0 means relevant was useful

            # Intervene: zero out irrelevant value connection
            D_no_irr = torch.ones_like(A)
            D_no_irr[rows, :, batch.query_positions, batch.irrelevant_positions] = 0.0
            y_no_irr = (A * D_no_irr) @ v
            y_no_irr = y_no_irr.transpose(1, 2).contiguous().view(b, l, model.d_model)
            x_no_irr = x + model.out(y_no_irr)
            x_no_irr = x_no_irr + model.ff(model.ln2(x_no_irr))
            logits_no_irr = model.head(model.final_ln(x_no_irr[rows, batch.query_positions]))
            loss_no_irr = F.cross_entropy(logits_no_irr, batch.targets, reduction="none")
            delta_loss_irr = loss_no_irr - base_loss  # < 0 means irrelevant was harmful

    model.zero_grad()
    if h.grad is not None:
        h.grad.zero_()

    return TeacherSignals(
        target_survival=target_survival,
        target_class=target_class,
        sensitivity=sensitivity,
        grad_norm=grad_norm,
        cosine_sim=cosine_sim,
        delta_loss_rel=delta_loss_rel,
        delta_loss_irr=delta_loss_irr,
    )


# =============================================================================
# Blind Recursive Baseline (Answer Appending from Previous Experiment)
# =============================================================================

def add_self_feedback(batch: Batch, predictions: torch.Tensor) -> Batch:
    """Append [detached prediction, repeated query] without adding parameters/tokens."""
    size = batch.tokens.shape[0]
    lengths = (~batch.padding_mask).sum(1)
    width = int(lengths.max()) + 2
    tokens = torch.zeros(size, width, dtype=batch.tokens.dtype, device=batch.tokens.device)
    padding = torch.ones(size, width, dtype=torch.bool, device=batch.tokens.device)
    qpos = lengths + 1
    for row, length in enumerate(lengths.tolist()):
        tokens[row, :length] = batch.tokens[row, :length]
        tokens[row, length] = predictions[row].detach()
        tokens[row, length + 1] = batch.tokens[row, batch.query_positions[row]]
        padding[row, : length + 2] = False
    causal = torch.tril(torch.ones(width, width, dtype=torch.bool, device=tokens.device))
    valid = causal[None] & ~padding[:, None, :] & ~padding[:, :, None]
    return Batch(
        tokens=tokens,
        targets=batch.targets,
        query_positions=qpos,
        relevant_positions=batch.relevant_positions,
        irrelevant_positions=batch.irrelevant_positions,
        padding_mask=padding,
        valid_links=valid,
    )


# =============================================================================
# Evaluation Harness & Metrics
# =============================================================================

def compute_gate_metrics(
    survival: Optional[torch.Tensor], batch: Batch
) -> Tuple[float, float, float]:
    """Compute relevant and irrelevant survival and delta_D."""
    if survival is None:
        return 1.0, 1.0, 0.0
    gates = survival.mean(1)  # Average over heads: (B, L, L)
    rows = torch.arange(batch.tokens.shape[0], device=batch.tokens.device)
    rel = gates[rows, batch.query_positions, batch.relevant_positions].mean().item()
    irr = gates[rows, batch.query_positions, batch.irrelevant_positions].mean().item()
    return rel, irr, rel - irr


def evaluate_model_on_split(
    model: Any,
    factory: BatchFactory,
    split: str,
    model_type: str = "gradient_rsl",
    batches: int = 8,
    batch_size: int = 128,
    k_passes_list: Tuple[int, ...] = (1, 2, 3, 5),
    controller_override: Optional[Any] = None,
    decay_mode: str = "rsl",
    oracle_evaluation: bool = False,
) -> Dict[str, Any]:
    """Evaluate model on a given split across recursive passes k in {1, 2, 3, 5}."""
    model.eval()
    total = 0
    # Track metrics per pass k
    pass_correct: Dict[int, int] = {k: 0 for k in k_passes_list}
    pass_rel: Dict[int, List[float]] = {k: [] for k in k_passes_list}
    pass_irr: Dict[int, List[float]] = {k: [] for k in k_passes_list}
    pass_useful: Dict[int, List[float]] = {k: [] for k in k_passes_list}
    pass_harmful: Dict[int, List[float]] = {k: [] for k in k_passes_list}
    pass_conf_wrong: Dict[int, List[float]] = {k: [] for k in k_passes_list}
    pass_conf_all: Dict[int, List[float]] = {k: [] for k in k_passes_list}
    agreements: Dict[int, int] = {k: 0 for k in k_passes_list}
    wrong_to_right: Dict[int, int] = {k: 0 for k in k_passes_list}
    right_to_wrong: Dict[int, int] = {k: 0 for k in k_passes_list}

    target_pred_acc_list: List[float] = []
    target_pred_corr_list: List[float] = []

    for _ in range(batches):
        batch = factory.batch(batch_size, split)
        b_size = batch.tokens.shape[0]
        total += b_size
        rows = torch.arange(b_size, device=batch.tokens.device)

        if model_type == "blind_recursive":
            # Blind recursive baseline: passes by appending detached token
            with torch.no_grad():
                logits1, info1 = model(batch.tokens, batch.query_positions, 1.0, batch.padding_mask)
                pred1 = logits1.argmax(-1)
                probs1 = F.softmax(logits1, dim=-1)
                corr1 = pred1 == batch.targets
                pass_correct[1] += corr1.sum().item()
                r1, i1, _ = compute_gate_metrics(info1.get("survival"), batch)
                pass_rel[1].append(r1)
                pass_irr[1].append(i1)
                pass_useful[1].append(r1)
                pass_harmful[1].append(i1)
                pass_conf_all[1].extend(probs1[rows, pred1].tolist())
                if (~corr1).sum() > 0:
                    pass_conf_wrong[1].extend(probs1[~corr1, pred1[~corr1]].tolist())

                # Pass 2
                fb = add_self_feedback(batch, pred1)
                logits2, info2 = model(fb.tokens, fb.query_positions, 1.0, fb.padding_mask)
                pred2 = logits2.argmax(-1)
                probs2 = F.softmax(logits2, dim=-1)
                corr2 = pred2 == batch.targets
                for k in k_passes_list:
                    if k >= 2:
                        pass_correct[k] += corr2.sum().item()
                        r2, i2, _ = compute_gate_metrics(info2.get("survival"), fb)
                        pass_rel[k].append(r2)
                        pass_irr[k].append(i2)
                        pass_useful[k].append(r2)
                        pass_harmful[k].append(i2)
                        agreements[k] += (pred2 == pred1).sum().item()
                        wrong_to_right[k] += ((~corr1) & corr2).sum().item()
                        right_to_wrong[k] += (corr1 & (~corr2)).sum().item()
                        pass_conf_all[k].extend(probs2[rows, pred2].tolist())
                        if (~corr1).sum() > 0:
                            pass_conf_wrong[k].extend(probs2[~corr1, pred2[~corr1]].tolist())

        elif model_type in {"baseline", "learned", "fixed"}:
            # Standard single-pass models
            with torch.no_grad():
                logits, info = model(batch.tokens, batch.query_positions, 1.0, batch.padding_mask)
                pred = logits.argmax(-1)
                probs = F.softmax(logits, dim=-1)
                corr = pred == batch.targets
                r, i, _ = compute_gate_metrics(info.get("survival"), batch)
                for k in k_passes_list:
                    pass_correct[k] += corr.sum().item()
                    pass_rel[k].append(r)
                    pass_irr[k].append(i)
                    pass_useful[k].append(r)
                    pass_harmful[k].append(i)
                    agreements[k] += b_size
                    pass_conf_all[k].extend(probs[rows, pred].tolist())
                    if (~corr).sum() > 0:
                        pass_conf_wrong[k].extend(probs[~corr, pred[~corr]].tolist())

        else:
            # Gradient RSL / Random / Shuffled / Oracle models
            oracle_surv = None
            if oracle_evaluation:
                # Oracle receives privileged supervision to compute exact gradient teacher target
                teacher_sig = compute_teacher_signals(model, batch, compute_interventions=False)
                oracle_surv = teacher_sig.target_survival

            # If tracking target prediction accuracy during evaluation, compute teacher target for evaluation
            # (Strictly for diagnostic measurement; NOT fed into the model forward pass)
            with torch.enable_grad():
                diag_teacher = compute_teacher_signals(model, batch, compute_interventions=False)

            with torch.no_grad():
                max_k = max(k_passes_list)
                if oracle_evaluation:
                    # Oracle single pass
                    logits1, info1 = model.forward_pass(
                        batch.tokens,
                        batch.query_positions,
                        exposure=1.0,
                        padding_mask=batch.padding_mask,
                        oracle_survival=oracle_surv,
                        decay_mode="oracle",
                    )
                    pass_results = [(logits1, info1)] * max_k
                else:
                    pass_results = model.recursive_forward(
                        batch.tokens,
                        batch.query_positions,
                        k_passes=max_k,
                        exposure=1.0,
                        padding_mask=batch.padding_mask,
                        controller_override=controller_override,
                        decay_mode=decay_mode,
                    )

                # Diagnostic: controller prediction correlation with real teacher target
                pred_surv = pass_results[0][1]["survival"]
                if pred_surv is not None:
                    target_surv = diag_teacher.target_survival
                    # Compare on valid query-to-key pairs
                    valid = (~pass_results[0][1]["invalid"]).expand_as(pred_surv)
                    p_flat = pred_surv[valid]
                    t_flat = target_surv[valid]
                    target_pred_acc = ((p_flat > 0.5) == (t_flat > 0.5)).float().mean().item()
                    target_pred_acc_list.append(target_pred_acc)
                    # Correlation
                    p_c = p_flat - p_flat.mean()
                    t_c = t_flat - t_flat.mean()
                    denom = (p_c.norm() * t_c.norm()).clamp_min(1e-8)
                    corr = (p_c * t_c).sum() / denom
                    target_pred_corr_list.append(corr.item())

                pred1 = pass_results[0][0].argmax(-1)
                corr1 = pred1 == batch.targets

                for k_idx, k in enumerate(k_passes_list):
                    logits_k, info_k = pass_results[k - 1]
                    pred_k = logits_k.argmax(-1)
                    probs_k = F.softmax(logits_k, dim=-1)
                    corr_k = pred_k == batch.targets

                    pass_correct[k] += corr_k.sum().item()
                    agreements[k] += (pred_k == pred1).sum().item()
                    wrong_to_right[k] += ((~corr1) & corr_k).sum().item()
                    right_to_wrong[k] += (corr1 & (~corr_k)).sum().item()

                    r, i, _ = compute_gate_metrics(info_k.get("survival"), batch)
                    pass_rel[k].append(r)
                    pass_irr[k].append(i)

                    # Compute S_k = E[D_useful] - E[D_harmful] using teacher sensitivity on query positions
                    if info_k.get("survival") is not None:
                        surv_mean = info_k["survival"].mean(1)  # (B, L, L)
                        u_val = surv_mean[rows, batch.query_positions, batch.relevant_positions].mean().item()
                        h_val = surv_mean[rows, batch.query_positions, batch.irrelevant_positions].mean().item()
                        pass_useful[k].append(u_val)
                        pass_harmful[k].append(h_val)
                    else:
                        pass_useful[k].append(1.0)
                        pass_harmful[k].append(1.0)

                    pass_conf_all[k].extend(probs_k[rows, pred_k].tolist())
                    if (~corr1).sum() > 0:
                        pass_conf_wrong[k].extend(probs_k[~corr1, pred_k[~corr1]].tolist())

    out: Dict[str, Any] = {
        "accuracy": pass_correct[1] / total,
        "k_accuracies": {k: pass_correct[k] / total for k in k_passes_list},
        "accuracy_change_k5": (pass_correct[max(k_passes_list)] - pass_correct[1]) / total,
        "wrong_to_right_k5": wrong_to_right[max(k_passes_list)] / total,
        "right_to_wrong_k5": right_to_wrong[max(k_passes_list)] / total,
        "answer_agreement_k5": agreements[max(k_passes_list)] / total,
        "D_rel": mean(pass_rel[1]),
        "D_irr": mean(pass_irr[1]),
        "delta_D": mean(pass_rel[1]) - mean(pass_irr[1]),
        "S_k": {k: mean(pass_useful[k]) - mean(pass_harmful[k]) for k in k_passes_list},
        "target_pred_acc": mean(target_pred_acc_list) if target_pred_acc_list else 0.0,
        "target_pred_corr": mean(target_pred_corr_list) if target_pred_corr_list else 0.0,
        "wrong_persistence_conf": {
            k: (mean(pass_conf_wrong[k]) if pass_conf_wrong[k] else 0.0) for k in k_passes_list
        },
        "overall_confidence": {k: mean(pass_conf_all[k]) for k in k_passes_list},
    }

    # Detect self-reinforcing error loop: wrong confidence increases across recursion
    conf1 = out["wrong_persistence_conf"].get(1, 0.0)
    conf5 = out["wrong_persistence_conf"].get(max(k_passes_list), 0.0)
    out["self_reinforcing_error_loop"] = bool(conf5 > conf1 + 0.01 and out["wrong_to_right_k5"] <= out["right_to_wrong_k5"])

    return out


# =============================================================================
# Training Pipeline (Phase A Teacher -> Student RSL)
# =============================================================================

def train_seed(
    seed: int,
    steps: int = 800,
    batch_size: int = 128,
    rsl_lr: float = 3e-3,
    base_lr: float = 3e-3,
    keep_penalty: float = 0.01,
) -> Dict[str, Any]:
    """Train base Transformer and RSL controller under seed with all baselines."""
    random.seed(seed)
    torch.manual_seed(seed)

    # 1. Base Transformer (no decay)
    base_model = TinySADT(VOCAB_SIZE, mode="baseline")
    # 2. Standard Learned Alpha SADT
    sadt_model = TinySADT(VOCAB_SIZE, mode="learned")
    # 3. Fixed Decay Transformer
    fixed_model = TinySADT(VOCAB_SIZE, mode="fixed")
    # 4. Blind Recursive Model
    blind_model = TinySADT(VOCAB_SIZE, mode="baseline")
    # 5. Gradient RSL Model
    rsl_model = GradientRSLModel(VOCAB_SIZE)
    # 6. Shuffled Target RSL Controller (Control Condition)
    shuffled_controller = GradientRSLController(rsl_model.d_model, rsl_model.n_heads)
    # 7. Random Controller
    random_controller = GradientRSLController(rsl_model.d_model, rsl_model.n_heads)

    # Initialize shared base weights for fair comparison across baselines
    initial_weights = copy.deepcopy(base_model.state_dict())

    for m in (sadt_model, fixed_model, blind_model):
        shared = {
            k: v
            for k, v in initial_weights.items()
            if k in m.state_dict() and m.state_dict()[k].shape == v.shape
        }
        m.load_state_dict(shared, strict=False)

    shared_rsl = {
        k: v
        for k, v in initial_weights.items()
        if k in rsl_model.state_dict() and rsl_model.state_dict()[k].shape == v.shape
    }
    rsl_model.load_state_dict(shared_rsl, strict=False)

    factory = BatchFactory(seed)
    train_batches = [factory.batch(batch_size, "train") for _ in range(steps)]

    # Optimizers
    opt_base = torch.optim.AdamW(base_model.parameters(), lr=base_lr)
    opt_sadt = torch.optim.AdamW(sadt_model.parameters(), lr=base_lr)
    opt_fixed = torch.optim.AdamW(fixed_model.parameters(), lr=base_lr)
    opt_blind = torch.optim.AdamW(blind_model.parameters(), lr=base_lr)
    opt_rsl_base = torch.optim.AdamW(
        [p for n, p in rsl_model.named_parameters() if not n.startswith("controller")],
        lr=base_lr,
    )
    opt_rsl_ctrl = torch.optim.AdamW(rsl_model.controller.parameters(), lr=rsl_lr)
    opt_shuffled_ctrl = torch.optim.AdamW(shuffled_controller.parameters(), lr=rsl_lr)

    # Training loop
    base_model.train()
    sadt_model.train()
    fixed_model.train()
    blind_model.train()
    rsl_model.train()
    shuffled_controller.train()

    rsl_losses = []
    shuffled_losses = []

    for step, batch in enumerate(train_batches, 1):
        exposure = min(1.0, step / max(1, steps // 2))

        # --- Baseline 1: Standard Transformer ---
        opt_base.zero_grad()
        logits_base, _ = base_model(batch.tokens, batch.query_positions, 1.0, batch.padding_mask)
        loss_base = F.cross_entropy(logits_base, batch.targets)
        loss_base.backward()
        opt_base.step()

        # --- Baseline 2: Learned SADT ---
        opt_sadt.zero_grad()
        logits_sadt, info_sadt = sadt_model(
            batch.tokens, batch.query_positions, exposure, batch.padding_mask
        )
        ce_sadt = F.cross_entropy(logits_sadt, batch.targets)
        reg_sadt = retention_loss(info_sadt["survival"], batch.valid_links)
        loss_sadt = ce_sadt + keep_penalty * reg_sadt if reg_sadt is not None else ce_sadt
        loss_sadt.backward()
        opt_sadt.step()

        # --- Baseline 3: Fixed Decay ---
        opt_fixed.zero_grad()
        logits_fixed, _ = fixed_model(
            batch.tokens, batch.query_positions, exposure, batch.padding_mask
        )
        loss_fixed = F.cross_entropy(logits_fixed, batch.targets)
        loss_fixed.backward()
        opt_fixed.step()

        # --- Baseline 4: Blind Recursive ---
        opt_blind.zero_grad()
        logits_bl1, _ = blind_model(batch.tokens, batch.query_positions, 1.0, batch.padding_mask)
        fb_bl = add_self_feedback(batch, logits_bl1.argmax(-1))
        logits_bl2, _ = blind_model(fb_bl.tokens, fb_bl.query_positions, 1.0, fb_bl.padding_mask)
        loss_blind = (F.cross_entropy(logits_bl1, batch.targets) + F.cross_entropy(logits_bl2, batch.targets)) / 2
        loss_blind.backward()
        opt_blind.step()

        # --- Phase A: Gradient-Pattern RSL Controller Training ---
        # 1. Base forward & supervised loss to extract teacher gradient signals
        teacher_signals = compute_teacher_signals(
            rsl_model, batch, scale=4.0, compute_interventions=(step % 20 == 0)
        )
        target_survival = teacher_signals.target_survival.detach()

        # 2. Update base model weights with standard supervised task loss
        opt_rsl_base.zero_grad()
        logits_rsl, info_rsl = rsl_model.forward_pass(
            batch.tokens,
            batch.query_positions,
            exposure=exposure,
            padding_mask=batch.padding_mask,
            decay_mode="rsl",
        )
        loss_rsl_task = F.cross_entropy(logits_rsl, batch.targets)
        loss_rsl_task.backward()
        opt_rsl_base.step()

        # 3. Train RSL Controller to predict teacher target survival D* from inference activations (q, k, A)
        opt_rsl_ctrl.zero_grad()
        # Legitimate features: detached q, k, A from forward pass
        q_det = info_rsl["q"].detach()
        k_det = info_rsl["k"].detach()
        A_det = info_rsl["A_base"].detach()
        invalid_det = info_rsl["invalid"]

        _, pred_survival = rsl_model.controller(q_det, k_det, A_det, exposure, invalid_det)
        valid_mask = (~invalid_det).expand_as(pred_survival)
        loss_rsl_ctrl = F.mse_loss(pred_survival[valid_mask], target_survival[valid_mask])
        loss_rsl_ctrl.backward()
        opt_rsl_ctrl.step()
        rsl_losses.append(loss_rsl_ctrl.item())

        # 4. Control baseline: Train Shuffled Target Controller on permuted gradient targets
        opt_shuffled_ctrl.zero_grad()
        shuffled_target = target_survival[torch.randperm(batch_size, device=batch.tokens.device)]
        _, pred_shuffled = shuffled_controller(q_det, k_det, A_det, exposure, invalid_det)
        loss_shuffled = F.mse_loss(pred_shuffled[valid_mask], shuffled_target[valid_mask])
        loss_shuffled.backward()
        opt_shuffled_ctrl.step()
        shuffled_losses.append(loss_shuffled.item())

    # =========================================================================
    # Phase B: Evaluation on IID + 5 OOD Splits across all Baselines & Passes
    # =========================================================================
    splits = ("iid",) + CASES
    eval_results: Dict[str, Dict[str, Any]] = {}

    baseline_names = [
        "transformer_baseline",
        "learned_sadt",
        "fixed_decay",
        "blind_recursive",
        "gradient_rsl",
        "random_controller",
        "shuffled_controller",
        "oracle_controller",
    ]

    for b_name in baseline_names:
        eval_results[b_name] = {}

    for index, split in enumerate(splits):
        eval_factory = BatchFactory(seed + 100_000 + index)

        # 1. Transformer Baseline
        eval_results["transformer_baseline"][split] = evaluate_model_on_split(
            base_model, eval_factory, split, model_type="baseline"
        )
        # 2. Learned SADT
        eval_results["learned_sadt"][split] = evaluate_model_on_split(
            sadt_model, eval_factory, split, model_type="learned"
        )
        # 3. Fixed Decay
        eval_results["fixed_decay"][split] = evaluate_model_on_split(
            fixed_model, eval_factory, split, model_type="fixed"
        )
        # 4. Blind Recursive
        eval_results["blind_recursive"][split] = evaluate_model_on_split(
            blind_model, eval_factory, split, model_type="blind_recursive"
        )
        # 5. Gradient RSL
        eval_results["gradient_rsl"][split] = evaluate_model_on_split(
            rsl_model, eval_factory, split, model_type="gradient_rsl", decay_mode="rsl"
        )
        # 6. Random Controller
        eval_results["random_controller"][split] = evaluate_model_on_split(
            rsl_model,
            eval_factory,
            split,
            model_type="gradient_rsl",
            controller_override=random_controller,
            decay_mode="random",
        )
        # 7. Shuffled Controller
        eval_results["shuffled_controller"][split] = evaluate_model_on_split(
            rsl_model,
            eval_factory,
            split,
            model_type="gradient_rsl",
            controller_override=shuffled_controller,
            decay_mode="rsl",
        )
        # 8. Oracle Controller (Privileged Upper Bound)
        eval_results["oracle_controller"][split] = evaluate_model_on_split(
            rsl_model, eval_factory, split, model_type="gradient_rsl", oracle_evaluation=True
        )

    return {
        "seed": seed,
        "train_rsl_loss": mean(rsl_losses[-50:]),
        "train_shuffled_loss": mean(shuffled_losses[-50:]),
        "eval_results": eval_results,
    }


# =============================================================================
# Summary Statistics & Aggregator
# =============================================================================

def calculate_stats(values: List[float]) -> Dict[str, float]:
    """Return mean and sample standard deviation."""
    if not values:
        return {"mean": 0.0, "std": 0.0}
    return {
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
    }


def summarize_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate multi-seed metrics into comprehensive structured summary."""
    models = list(runs[0]["eval_results"].keys())
    splits = list(runs[0]["eval_results"][models[0]].keys())
    k_passes = (1, 2, 3, 5)

    summary: Dict[str, Any] = {}

    for model_name in models:
        summary[model_name] = {}
        for split in splits:
            summary[model_name][split] = {}

            # Scalar metrics
            for metric in (
                "accuracy",
                "accuracy_change_k5",
                "wrong_to_right_k5",
                "right_to_wrong_k5",
                "answer_agreement_k5",
                "D_rel",
                "D_irr",
                "delta_D",
                "target_pred_acc",
                "target_pred_corr",
            ):
                vals = [r["eval_results"][model_name][split][metric] for r in runs]
                summary[model_name][split][metric] = calculate_stats(vals)

            # Recursive metrics per k
            summary[model_name][split]["k_accuracies"] = {
                k: calculate_stats(
                    [r["eval_results"][model_name][split]["k_accuracies"][k] for r in runs]
                )
                for k in k_passes
            }
            summary[model_name][split]["S_k"] = {
                k: calculate_stats(
                    [r["eval_results"][model_name][split]["S_k"][k] for r in runs]
                )
                for k in k_passes
            }
            summary[model_name][split]["wrong_persistence_conf"] = {
                k: calculate_stats(
                    [r["eval_results"][model_name][split]["wrong_persistence_conf"][k] for r in runs]
                )
                for k in k_passes
            }

            # Self-reinforcing error loop flag: True if flagged in at least 2/5 seeds
            flag_count = sum(
                1 for r in runs if r["eval_results"][model_name][split]["self_reinforcing_error_loop"]
            )
            summary[model_name][split]["self_reinforcing_error_loop_rate"] = flag_count / len(runs)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Gradient-Pattern RSL Experiment")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds")
    parser.add_argument("--steps", type=int, default=800, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--workers", type=int, default=1, help="Parallel seed workers")
    parser.add_argument(
        "--output", default="results/gradient_rsl_report.json", help="Output report JSON"
    )
    args = parser.parse_args()

    print(f"Starting Gradient-Pattern RSL Experiment: {args.seeds} seeds, {args.steps} steps...")
    runs = []

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(train_seed, seed, args.steps, args.batch_size)
                for seed in range(args.seeds)
            ]
            for future in as_completed(futures):
                res = future.result()
                runs.append(res)
                print(f"Completed seed {res['seed']}")
    else:
        for seed in range(args.seeds):
            res = train_seed(seed, args.steps, args.batch_size)
            runs.append(res)
            print(f"Completed seed {seed}")

    runs.sort(key=lambda r: r["seed"])
    summary = summarize_runs(runs)

    payload = {
        "title": "Gradient-Pattern Recursive Self-Learning (RSL) Benchmark",
        "hypothesis": (
            "A secondary controller can observe backpropagation corrections during supervised training, "
            "learn recurring patterns of those corrections, and later predict useful information-retention/decay "
            "decisions on unseen examples without access to labels or gradients at inference time."
        ),
        "config": vars(args),
        "runs": runs,
        "summary": summary,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Report successfully saved to {args.output}")

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Model':<24} {'Split':<20} {'Acc (k=1)':<12} {'Acc (k=5)':<12} {'Delta D':<12} {'S_1 -> S_5'}")
    print("=" * 90)
    for model_name in summary:
        for split in ("iid", "unseen_combinations", "combined"):
            row = summary[model_name][split]
            acc1 = row["k_accuracies"][1]["mean"]
            acc5 = row["k_accuracies"][5]["mean"]
            d_d = row["delta_D"]["mean"]
            s1 = row["S_k"][1]["mean"]
            s5 = row["S_k"][5]["mean"]
            print(
                f"{model_name:<24} {split:<20} {acc1:<12.3f} {acc5:<12.3f} {d_d:<+12.4f} {s1:+.3f} -> {s5:+.3f}"
            )
        print("-" * 90)


if __name__ == "__main__":
    main()

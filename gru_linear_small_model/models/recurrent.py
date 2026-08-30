from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import nn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class LinearSmoothCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.candidate = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        u = torch.cat([h, x], dim=-1)
        beta = torch.sigmoid(self.gate(u))
        candidate = torch.tanh(self.candidate(u))
        return beta * h + (1.0 - beta) * candidate, {"beta": beta, "candidate": candidate}


class RecurrentClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_dim = hidden_dim
        self.output = nn.Linear(hidden_dim, num_classes)

    def step(self, x, h, diagnostics: bool = False):
        raise NotImplementedError

    def forward(self, tokens, mask=None, return_diagnostics: bool = False):
        x = self.embedding(tokens)
        h = x.new_zeros(x.shape[0], self.hidden_dim)
        history: Dict[str, list] = {}
        for t in range(x.shape[1]):
            h_new, info = self.step(x[:, t], h, return_diagnostics)
            if mask is not None:
                h = torch.where(mask[:, t, None], h_new, h)
            else:
                h = h_new
            if return_diagnostics:
                for key, value in info.items():
                    history.setdefault(key, []).append(value.detach())
        logits = self.output(h)
        if return_diagnostics:
            return logits, {key: torch.stack(value, dim=1) for key, value in history.items()}
        return logits


class GRUBaseline(RecurrentClassifier):
    def __init__(self, vocab_size, num_classes, embedding_dim, hidden_dim):
        super().__init__(vocab_size, num_classes, embedding_dim, hidden_dim)
        self.cell = nn.GRUCell(embedding_dim, hidden_dim)

    def step(self, x, h, diagnostics=False):
        return self.cell(x, h), {}


class LinearSmoothRNN(RecurrentClassifier):
    def __init__(self, vocab_size, num_classes, embedding_dim, hidden_dim):
        super().__init__(vocab_size, num_classes, embedding_dim, hidden_dim)
        self.cell = LinearSmoothCell(embedding_dim, hidden_dim)

    def step(self, x, h, diagnostics=False):
        return self.cell(x, h)


class HybridGRULinear(RecurrentClassifier):
    def __init__(
        self,
        vocab_size,
        num_classes,
        embedding_dim,
        hidden_dim,
        force_lambda: Optional[float] = None,
        random_fusion: bool = False,
        scalar_fusion: bool = False,
    ):
        super().__init__(vocab_size, num_classes, embedding_dim, hidden_dim)
        self.gru = nn.GRUCell(embedding_dim, hidden_dim)
        self.linear = LinearSmoothCell(embedding_dim, hidden_dim)
        self.force_lambda = force_lambda
        self.random_fusion = random_fusion
        fusion_out = 1 if scalar_fusion else hidden_dim
        self.fusion = nn.Linear(embedding_dim + 3 * hidden_dim, fusion_out)
        nn.init.zeros_(self.fusion.bias)

    def step(self, x, h, diagnostics=False):
        h_gru = self.gru(x, h)
        h_linear, linear_info = self.linear(x, h)
        if self.random_fusion:
            lam = torch.rand_like(h_gru)
        elif self.force_lambda is not None:
            lam = torch.full_like(h_gru, self.force_lambda)
        else:
            lam = torch.sigmoid(self.fusion(torch.cat([h, x, h_gru, h_linear], dim=-1)))
        h_new = lam * h_gru + (1.0 - lam) * h_linear
        return h_new, {"lambda": lam, "h_gru": h_gru, "h_linear": h_linear, **linear_info}


class SmoothedGRU(RecurrentClassifier):
    def __init__(self, vocab_size, num_classes, embedding_dim, hidden_dim):
        super().__init__(vocab_size, num_classes, embedding_dim, hidden_dim)
        self.gru = nn.GRUCell(embedding_dim, hidden_dim)
        self.gate = nn.Linear(embedding_dim + 2 * hidden_dim, hidden_dim)

    def step(self, x, h, diagnostics=False):
        h_gru = self.gru(x, h)
        beta = torch.sigmoid(self.gate(torch.cat([h, x, h_gru], dim=-1)))
        return beta * h + (1.0 - beta) * h_gru, {"beta": beta}


class DualStateHybrid(nn.Module):
    def __init__(self, vocab_size, num_classes, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRUCell(embedding_dim, hidden_dim)
        self.linear = LinearSmoothCell(embedding_dim, hidden_dim)
        self.output = nn.Linear(2 * hidden_dim, num_classes)
        self.hidden_dim = hidden_dim

    def forward(self, tokens, mask=None, return_diagnostics=False):
        x = self.embedding(tokens)
        hg = x.new_zeros(x.shape[0], self.hidden_dim)
        hl = x.new_zeros(x.shape[0], self.hidden_dim)
        beta_history = []
        for t in range(x.shape[1]):
            ng = self.gru(x[:, t], hg)
            nl, info = self.linear(x[:, t], hl)
            if mask is not None:
                active = mask[:, t, None]
                hg, hl = torch.where(active, ng, hg), torch.where(active, nl, hl)
            else:
                hg, hl = ng, nl
            if return_diagnostics:
                beta_history.append(info["beta"].detach())
        logits = self.output(torch.cat([hg, hl], dim=-1))
        return (logits, {"beta": torch.stack(beta_history, dim=1)}) if return_diagnostics else logits


def _parameter_matched_gru_dim(vocab_size, classes, embedding_dim, hybrid_dim):
    hybrid = HybridGRULinear(vocab_size, classes, embedding_dim, hybrid_dim)
    target = count_parameters(hybrid)
    best_dim, best_delta = hybrid_dim, math.inf
    for dim in range(hybrid_dim, hybrid_dim * 4 + 1):
        delta = abs(count_parameters(GRUBaseline(vocab_size, classes, embedding_dim, dim)) - target)
        if delta < best_delta:
            best_dim, best_delta = dim, delta
    return best_dim


def build_models(vocab_size: int, classes: int, embedding_dim: int, hidden_dim: int):
    matched_dim = _parameter_matched_gru_dim(vocab_size, classes, embedding_dim, hidden_dim)
    args = (vocab_size, classes, embedding_dim, hidden_dim)
    return {
        "gru": GRUBaseline(*args),
        "linear": LinearSmoothRNN(*args),
        "hybrid": HybridGRULinear(*args),
        "fixed_half_hybrid": HybridGRULinear(*args, force_lambda=0.5),
        "forced_gru_hybrid": HybridGRULinear(*args, force_lambda=1.0),
        "forced_linear_hybrid": HybridGRULinear(*args, force_lambda=0.0),
        "random_fusion_hybrid": HybridGRULinear(*args, random_fusion=True),
        "scalar_fusion_hybrid": HybridGRULinear(*args, scalar_fusion=True),
        "param_matched_gru": GRUBaseline(vocab_size, classes, embedding_dim, matched_dim),
        "dual_state_hybrid": DualStateHybrid(*args),
        "smoothed_gru": SmoothedGRU(*args),
    }

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


MODES = {"baseline", "learned", "renorm", "pre_softmax", "random", "fixed"}


class SemanticDecayAttention(nn.Module):
    def __init__(self, d_model=32, n_heads=4, mode="learned", alpha_hidden=16):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown attention mode: {mode}")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.mode, self.n_heads, self.d_head = mode, n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        if mode in {"learned", "renorm", "pre_softmax"}:
            self.alpha_net = nn.Sequential(
                nn.Linear(3 * self.d_head, alpha_hidden), nn.Tanh(), nn.Linear(alpha_hidden, 1)
            )

    def _alpha(self, q, k, generator=None):
        b, h, l, d = q.shape
        if self.mode == "random":
            return torch.rand(b, h, l, l, device=q.device, generator=generator)
        if self.mode == "fixed":
            return torch.ones(b, h, l, l, device=q.device)
        qi = q.unsqueeze(3).expand(-1, -1, -1, l, -1)
        kj = k.unsqueeze(2).expand(-1, -1, l, -1, -1)
        return F.softplus(self.alpha_net(torch.cat([qi, kj, qi * kj], -1)).squeeze(-1))

    def forward(self, x, exposure=1.0, padding_mask=None, intervention=None,
                query_positions=None, relevant_positions=None, generator=None):
        b, l, d_model = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        shape = (b, l, self.n_heads, self.d_head)
        q, k, v = [z.view(shape).transpose(1, 2) for z in (q, k, v)]
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        causal = torch.triu(torch.ones(l, l, dtype=torch.bool, device=x.device), 1)
        invalid = causal.view(1, 1, l, l)
        if padding_mask is not None:
            invalid = invalid | padding_mask[:, None, None, :]

        alpha = survival = None
        if self.mode == "baseline":
            attention = F.softmax(scores.masked_fill(invalid, -torch.inf), -1)
        else:
            alpha = self._alpha(q, k, generator)
            survival = torch.exp(-alpha * exposure).masked_fill(invalid, 0.0)
            if intervention == "shuffle":
                survival = survival.roll(1, dims=0)
            elif intervention == "random":
                survival = torch.rand(survival.shape, device=x.device, generator=generator).masked_fill(invalid, 0.0)
            elif intervention == "force_one":
                survival = torch.ones_like(survival).masked_fill(invalid, 0.0)
            elif intervention == "zero_relevant":
                if query_positions is None or relevant_positions is None:
                    raise ValueError("zero_relevant requires query and relevant positions")
                rows = torch.arange(b, device=x.device)
                survival[rows, :, query_positions, relevant_positions] = 0.0

            if self.mode == "pre_softmax":
                biased = scores + survival.clamp_min(1e-30).log()
                attention = F.softmax(biased.masked_fill(invalid, -torch.inf), -1)
            else:
                base = F.softmax(scores.masked_fill(invalid, -torch.inf), -1)
                attention = base * survival
                if self.mode == "renorm":
                    attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-8)
        y = attention @ v
        y = y.transpose(1, 2).contiguous().view(b, l, d_model)
        return self.out(y), {"alpha": alpha, "survival": survival, "attention": attention}

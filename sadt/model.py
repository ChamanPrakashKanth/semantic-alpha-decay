import torch
import torch.nn as nn

from .attention import SemanticDecayAttention


class TinySADT(nn.Module):
    """One attention path: the prediction must pass through the tested mechanism."""
    def __init__(self, vocab_size, max_len=32, d_model=32, n_heads=4, mode="learned"):
        super().__init__()
        self.token = nn.Embedding(vocab_size, d_model)
        self.position = nn.Embedding(max_len, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SemanticDecayAttention(d_model, n_heads, mode)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
        self.final_ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens, query_positions, exposure=1.0, padding_mask=None,
                intervention=None, relevant_positions=None, generator=None):
        pos = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.token(tokens) + self.position(pos)
        a, info = self.attn(self.ln1(x), exposure, padding_mask, intervention,
                            query_positions, relevant_positions, generator)
        x = x + a
        x = x + self.ff(self.ln2(x))
        rows = torch.arange(tokens.shape[0], device=tokens.device)
        return self.head(self.final_ln(x[rows, query_positions])), info

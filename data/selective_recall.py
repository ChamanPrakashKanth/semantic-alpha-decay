import random
from dataclasses import dataclass

import torch

PAD, BOS, QUERY_A, QUERY_B, KEY_A, KEY_B, SEP = range(7)
VALUES = tuple(range(7, 15))
NOISE = tuple(range(15, 23))
VOCAB_SIZE = 23


@dataclass
class Batch:
    tokens: torch.Tensor
    targets: torch.Tensor
    query_positions: torch.Tensor
    relevant_positions: torch.Tensor
    irrelevant_positions: torch.Tensor
    padding_mask: torch.Tensor
    valid_links: torch.Tensor


SHIFTS = ("iid", "unseen_combinations", "unseen_layout", "more_distractors",
          "reversed_order", "combined")


class BatchFactory:
    """Deterministic generator with one isolated distribution shift per split."""
    def __init__(self, seed=0, max_len=24):
        self.rng, self.max_len = random.Random(seed), max_len

    @staticmethod
    def _held_out(va, vb):
        return ((va - VALUES[0]) * len(VALUES) + vb - VALUES[0]) % 5 == 0

    def _example(self, split):
        if split == "train":
            split = "iid"
        elif split == "ood":  # Backward-compatible alias.
            split = "combined"
        if split not in SHIFTS:
            raise ValueError(f"unknown split: {split}")
        held_out = split in {"unseen_combinations", "combined"}
        while True:
            va, vb = self.rng.sample(VALUES, 2)
            if self._held_out(va, vb) == held_out:
                break
        query = self.rng.choice((QUERY_A, QUERY_B))
        pairs = [(KEY_A, va, "a"), (KEY_B, vb, "b")]
        self.rng.shuffle(pairs)
        noise_n = self.rng.randint(4, 7) if split in {"more_distractors", "combined"} else self.rng.randint(0, 3)
        noise = [self.rng.choice(NOISE) for _ in range(noise_n)]
        reverse = split in {"reversed_order", "combined"}
        rendered = [([value, key] if reverse else [key, value]) for key, value, _ in pairs]
        if split not in {"unseen_layout", "combined"}:
            cut1, cut2 = sorted((self.rng.randint(0, noise_n), self.rng.randint(0, noise_n)))
            groups = [noise[:cut1], noise[cut1:cut2], noise[cut2:]]
            seq = [BOS] + groups[0]
            for i, pair in enumerate(rendered):
                seq += pair + groups[i + 1]
            seq += [query, SEP]
        else:
            # Unseen layout only: separator moves between pairs and query becomes final.
            midpoint = noise_n // 2
            seq = [BOS] + noise[:midpoint] + rendered[0] + [SEP]
            seq += noise[midpoint:] + rendered[1] + [query]
        qpos = seq.index(query)
        wanted = "a" if query == QUERY_A else "b"
        value = va if wanted == "a" else vb
        irrelevant_value = vb if wanted == "a" else va
        rpos = seq.index(value)
        ipos = seq.index(irrelevant_value)
        return seq, value, qpos, rpos, ipos

    def batch(self, size, split="train"):
        examples = [self._example(split) for _ in range(size)]
        length = max(len(e[0]) for e in examples)
        tokens = torch.full((size, length), PAD, dtype=torch.long)
        qpos, rpos, ipos, targets = [], [], [], []
        for i, (seq, target, q, r, irr) in enumerate(examples):
            tokens[i, :len(seq)] = torch.tensor(seq)
            targets.append(target); qpos.append(q); rpos.append(r); ipos.append(irr)
        padding = tokens.eq(PAD)
        causal = torch.tril(torch.ones(length, length, dtype=torch.bool))
        valid = causal[None] & ~padding[:, None, :] & ~padding[:, :, None]
        return Batch(tokens, torch.tensor(targets), torch.tensor(qpos),
                     torch.tensor(rpos), torch.tensor(ipos), padding, valid)

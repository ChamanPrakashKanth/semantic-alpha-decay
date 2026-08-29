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
    padding_mask: torch.Tensor
    valid_links: torch.Tensor


class BatchFactory:
    """Deterministic batches. OOD uses held-out value pairs and unseen layouts."""
    def __init__(self, seed=0, max_len=24):
        self.rng, self.max_len = random.Random(seed), max_len

    @staticmethod
    def _held_out(va, vb):
        return ((va - VALUES[0]) * len(VALUES) + vb - VALUES[0]) % 5 == 0

    def _example(self, split):
        while True:
            va, vb = self.rng.sample(VALUES, 2)
            if self._held_out(va, vb) == (split == "ood"):
                break
        query = self.rng.choice((QUERY_A, QUERY_B))
        pairs = [(KEY_A, va, "a"), (KEY_B, vb, "b")]
        self.rng.shuffle(pairs)
        noise_n = self.rng.randint(0, 3) if split == "train" else self.rng.randint(4, 7)
        noise = [self.rng.choice(NOISE) for _ in range(noise_n)]
        if split == "train":
            cut1, cut2 = sorted((self.rng.randint(0, noise_n), self.rng.randint(0, noise_n)))
            groups = [noise[:cut1], noise[cut1:cut2], noise[cut2:]]
            seq = [BOS] + groups[0]
            for i, (key, value, _) in enumerate(pairs):
                seq += [key, value] + groups[i + 1]
        else:
            # Held-out layout: every value precedes its key and distractors separate them.
            seq = [BOS] + noise[:2]
            for key, value, _ in pairs:
                seq += [value] + noise[2:3] + [key]
            seq += noise[3:]
        seq += [query, SEP]
        qpos = len(seq) - 2
        wanted = "a" if query == QUERY_A else "b"
        value = va if wanted == "a" else vb
        rpos = seq.index(value)
        return seq, value, qpos, rpos

    def batch(self, size, split="train"):
        examples = [self._example(split) for _ in range(size)]
        length = max(len(e[0]) for e in examples)
        tokens = torch.full((size, length), PAD, dtype=torch.long)
        qpos, rpos, targets = [], [], []
        for i, (seq, target, q, r) in enumerate(examples):
            tokens[i, :len(seq)] = torch.tensor(seq)
            targets.append(target); qpos.append(q); rpos.append(r)
        padding = tokens.eq(PAD)
        causal = torch.tril(torch.ones(length, length, dtype=torch.bool))
        valid = causal[None] & ~padding[:, None, :] & ~padding[:, :, None]
        return Batch(tokens, torch.tensor(targets), torch.tensor(qpos),
                     torch.tensor(rpos), padding, valid)

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch


@dataclass(frozen=True)
class Vocabulary:
    num_entities: int

    @property
    def arrow(self) -> int:
        return self.num_entities

    @property
    def query(self) -> int:
        return self.num_entities + 1

    @property
    def sep(self) -> int:
        return self.num_entities + 2

    @property
    def pad(self) -> int:
        return self.num_entities + 3

    @property
    def size(self) -> int:
        return self.num_entities + 4


@dataclass
class RelationExample:
    tokens: List[int]
    target: int
    chain_length: int
    distractors: int
    token_roles: List[int]
    edges: List[tuple]
    query: int
    camouflage_distractors: int = 0


class RelationChainGenerator:
    """Deterministic generator with shuffled true edges and unreachable distractors.

    Token roles: 0=special/padding, 1=true-chain entity, 2=distractor entity.
    """

    def __init__(self, num_entities: int = 24, seed: int = 0, balance_destinations: bool = True):
        self.vocab = Vocabulary(num_entities)
        self.rng = random.Random(seed)
        self.balance_destinations = balance_destinations

    def get_state(self):
        return self.rng.getstate()

    def set_state(self, state):
        self.rng.setstate(state)

    def sample_example(self, chain_length: int, distractors: int) -> RelationExample:
        if chain_length < 1 or chain_length >= self.vocab.num_entities:
            raise ValueError("chain_length must leave at least one entity outside the chain")
        chain_nodes = self.rng.sample(range(self.vocab.num_entities), chain_length + 1)
        true_edges = list(zip(chain_nodes[:-1], chain_nodes[1:]))
        outside = [e for e in range(self.vocab.num_entities) if e not in chain_nodes]
        if distractors and not outside:
            raise ValueError("no unreachable entities available for distractors")

        distractor_edges = []
        used = set(true_edges)
        # Without this camouflage, picking a random displayed destination has
        # 1/(chain edges + distractors) accuracy, which is a severe shortcut for
        # short chains. Giving every entity one destination occurrence restores
        # the candidate-only baseline to 1/num_entities while keeping all added
        # sources unreachable from the query.
        if self.balance_destinations:
            if not outside:
                raise ValueError("destination balancing requires an entity outside the chain")
            present_destinations = {dst for _, dst in true_edges}
            for dst in range(self.vocab.num_entities):
                if dst in present_destinations:
                    continue
                choices = [src for src in outside if src != dst and (src, dst) not in used]
                if not choices:
                    raise RuntimeError("could not generate destination-balanced camouflage")
                edge = (self.rng.choice(choices), dst)
                used.add(edge)
                distractor_edges.append(edge)
        camouflage_count = len(distractor_edges)
        attempts = 0
        while len(distractor_edges) < camouflage_count + distractors:
            attempts += 1
            if attempts > 10000:
                raise RuntimeError("could not generate unique distractors")
            # A source outside the true reachable set cannot form a competing path
            # from the query, regardless of its destination.
            edge = (self.rng.choice(outside), self.rng.randrange(self.vocab.num_entities))
            if edge[0] == edge[1] or edge in used:
                continue
            used.add(edge)
            distractor_edges.append(edge)

        marked_edges = [(a, b, 1) for a, b in true_edges]
        marked_edges += [(a, b, 2) for a, b in distractor_edges]
        self.rng.shuffle(marked_edges)

        tokens: List[int] = []
        roles: List[int] = []
        for src, dst, role in marked_edges:
            tokens.extend([src, self.vocab.arrow, dst, self.vocab.sep])
            roles.extend([role, 0, role, 0])
        tokens.extend([self.vocab.query, chain_nodes[0]])
        roles.extend([0, 1])
        return RelationExample(
            tokens=tokens,
            target=chain_nodes[-1],
            chain_length=chain_length,
            distractors=distractors,
            token_roles=roles,
            edges=[(a, b) for a, b, _ in marked_edges],
            query=chain_nodes[0],
            camouflage_distractors=camouflage_count,
        )

    def sample_batch(
        self,
        batch_size: int,
        chain_lengths: Sequence[int],
        distractor_counts: Sequence[int],
        device: torch.device | str = "cpu",
    ) -> Dict[str, torch.Tensor]:
        examples = [
            self.sample_example(self.rng.choice(chain_lengths), self.rng.choice(distractor_counts))
            for _ in range(batch_size)
        ]
        max_len = max(len(e.tokens) for e in examples)
        tokens = torch.full((batch_size, max_len), self.vocab.pad, dtype=torch.long)
        roles = torch.zeros((batch_size, max_len), dtype=torch.long)
        mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
        for i, example in enumerate(examples):
            n = len(example.tokens)
            tokens[i, :n] = torch.tensor(example.tokens)
            roles[i, :n] = torch.tensor(example.token_roles)
            mask[i, :n] = True
        return {
            "tokens": tokens.to(device),
            "mask": mask.to(device),
            "targets": torch.tensor([e.target for e in examples], device=device),
            "roles": roles.to(device),
            "chain_lengths": torch.tensor([e.chain_length for e in examples], device=device),
            "distractors": torch.tensor([e.distractors for e in examples], device=device),
        }

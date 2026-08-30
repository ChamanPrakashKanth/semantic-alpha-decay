from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn.functional as F

from .data import RelationChainGenerator


@dataclass(frozen=True)
class CurriculumStage:
    stage: int
    name: str
    chain_lengths: Sequence[int]
    edge_order: str
    distractors: Sequence[int]
    destination_coverage: int
    threshold: float
    max_steps: int


def curriculum_stages(preset: str = "pilot") -> List[CurriculumStage]:
    max_steps = {
        "pilot": [250, 400, 600, 800, 1000, 1200, 1500],
        "full": [400, 600, 800, 1200, 1500, 2000, 2500],
    }[preset]
    specs = [
        ("one_hop_ordered", [1], "ordered", [0], 0, 0.90),
        ("two_hop_ordered", [2], "ordered", [0], 0, 0.90),
        ("two_three_partial_shuffle", [2, 3], "partial", [0], 0, 0.85),
        ("two_three_full_shuffle", [2, 3], "shuffled", [0], 0, 0.80),
        ("shuffled_one_distractor", [2, 3], "shuffled", [1], 0, 0.75),
        ("partial_destination_balance", [2, 3], "shuffled", [1, 2], 12, 0.70),
        ("full_corrected_generator", [2, 3], "shuffled", [0, 1], 24, 0.70),
    ]
    return [
        CurriculumStage(i, name, lengths, order, noise, coverage, threshold, max_steps[i])
        for i, (name, lengths, order, noise, coverage, threshold) in enumerate(specs)
    ]


def make_batch(
    stage: CurriculumStage,
    examples: int,
    seed: int,
    num_entities: int,
    device: torch.device,
):
    return RelationChainGenerator(num_entities, seed, balance_destinations=False).sample_batch(
        examples,
        stage.chain_lengths,
        stage.distractors,
        device,
        edge_order=stage.edge_order,
        destination_coverage=stage.destination_coverage,
    )


@torch.no_grad()
def score(model, batch):
    model.eval()
    logits = model(batch["tokens"], batch["mask"])
    return {
        "accuracy": float(logits.argmax(-1).eq(batch["targets"]).float().mean()),
        "loss": float(F.cross_entropy(logits, batch["targets"])),
    }

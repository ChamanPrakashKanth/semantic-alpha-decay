from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple


@dataclass
class ExperimentConfig:
    num_entities: int = 24
    embedding_dim: int = 32
    hidden_dim: int = 32
    batch_size: int = 128
    learning_rate: float = 3e-3
    weight_decay: float = 0.0
    gradient_clip_norm: float = 1.0
    steps: int = 1500
    seeds: Tuple[int, ...] = tuple(range(10))
    eval_examples_per_mode: int = 5000
    train_chain_lengths: Tuple[int, ...] = (2, 3)
    train_distractors: Tuple[int, ...] = (0, 1)
    device: str = "auto"
    deterministic: bool = True
    shortcut_balanced_destinations: bool = True

    @classmethod
    def smoke(cls) -> "ExperimentConfig":
        return cls(
            embedding_dim=24,
            hidden_dim=24,
            batch_size=64,
            steps=120,
            seeds=(0,),
            eval_examples_per_mode=512,
        )

    @classmethod
    def exploratory(cls) -> "ExperimentConfig":
        return cls(
            embedding_dim=24,
            hidden_dim=24,
            batch_size=64,
            steps=1000,
            seeds=tuple(range(5)),
            eval_examples_per_mode=2000,
        )

    def evaluation_modes(self) -> Dict[str, Tuple[List[int], List[int]]]:
        return {
            "iid": ([2, 3], [0, 1]),
            "noise_ood": ([2, 3], [3, 4, 5, 6]),
            "length_ood": ([4, 5], [0, 1]),
            "combined_ood": ([4, 5], [3, 4, 5, 6]),
        }

    def to_dict(self):
        result = asdict(self)
        result["seeds"] = list(self.seeds)
        result["train_chain_lengths"] = list(self.train_chain_lengths)
        result["train_distractors"] = list(self.train_distractors)
        return result

from __future__ import annotations

import random
import time
from typing import Dict, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from .config import ExperimentConfig
from .data import RelationChainGenerator
from .models import count_parameters


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_models_shared_data(
    models: Dict[str, torch.nn.Module],
    generator: RelationChainGenerator,
    config: ExperimentConfig,
    device: torch.device,
    progress_every: int = 0,
):
    for model in models.values():
        model.to(device).train()
    optimizers = {
        name: torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        for name, model in models.items()
    }
    last_losses = {}
    started = time.perf_counter()
    for step in range(1, config.steps + 1):
        # This object is generated once and passed unchanged to every model.
        batch = generator.sample_batch(
            config.batch_size,
            config.train_chain_lengths,
            config.train_distractors,
            device,
        )
        for name, model in models.items():
            optimizer = optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["tokens"], batch["mask"])
            loss = F.cross_entropy(logits, batch["targets"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            last_losses[name] = float(loss.detach())
        if progress_every and (step == 1 or step % progress_every == 0):
            compact = " ".join(f"{k}={v:.3f}" for k, v in last_losses.items())
            print(f"step {step}/{config.steps} {compact}", flush=True)
    return {
        "last_losses": last_losses,
        "training_seconds": time.perf_counter() - started,
        "parameter_counts": {name: count_parameters(model) for name, model in models.items()},
    }


def overfit_tiny_batch(model, generator, device, steps: int = 300, threshold: float = 0.95):
    batch = generator.sample_batch(16, [2], [0], device)
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["tokens"], batch["mask"])
        loss = F.cross_entropy(logits, batch["targets"])
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        accuracy = (model(batch["tokens"], batch["mask"]).argmax(-1) == batch["targets"]).float().mean()
    return float(accuracy), float(loss.detach()), bool(accuracy >= threshold)

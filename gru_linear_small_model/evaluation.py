from __future__ import annotations

import time
from typing import Dict, Sequence

import torch
import torch.nn.functional as F

from .data import RelationChainGenerator
from .diagnostics import complementarity, fusion_diagnostics, rescue_damage


@torch.no_grad()
def evaluate_mode(
    models: Dict[str, torch.nn.Module],
    generator: RelationChainGenerator,
    chain_lengths: Sequence[int],
    distractors: Sequence[int],
    examples: int,
    batch_size: int,
    device: torch.device,
):
    for model in models.values():
        model.eval()
    records = {name: {"logits": [], "latency": 0.0} for name in models}
    targets, masks, roles, lengths, hybrid_lambdas = [], [], [], [], []
    heuristic_predictions = {
        "last_destination": [],
        "first_destination": [],
        "most_recent_entity": [],
        "most_frequent_entity": [],
        "random_class": [],
    }
    random_generator = torch.Generator(device="cpu").manual_seed(739391)
    completed = 0
    while completed < examples:
        n = min(batch_size, examples - completed)
        batch = generator.sample_batch(n, chain_lengths, distractors, device)
        targets.append(batch["targets"].cpu())
        masks.append(batch["mask"].cpu())
        roles.append(batch["roles"].cpu())
        lengths.append(batch["chain_lengths"].cpu())
        cpu_tokens, cpu_mask = batch["tokens"].cpu(), batch["mask"].cpu()
        valid_lengths = cpu_mask.sum(1)
        heuristic_predictions["first_destination"].append(cpu_tokens[:, 2])
        heuristic_predictions["last_destination"].append(
            cpu_tokens.gather(1, (valid_lengths - 4).unsqueeze(1)).squeeze(1)
        )
        heuristic_predictions["most_recent_entity"].append(
            cpu_tokens.gather(1, (valid_lengths - 1).unsqueeze(1)).squeeze(1)
        )
        frequent = []
        for row, active in zip(cpu_tokens, cpu_mask):
            entities = row[active & row.lt(generator.vocab.num_entities)]
            frequent.append(int(torch.bincount(entities, minlength=generator.vocab.num_entities).argmax()))
        heuristic_predictions["most_frequent_entity"].append(torch.tensor(frequent))
        heuristic_predictions["random_class"].append(
            torch.randint(generator.vocab.num_entities, (n,), generator=random_generator)
        )
        for name, model in models.items():
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            if name == "hybrid":
                logits, info = model(batch["tokens"], batch["mask"], return_diagnostics=True)
                hybrid_lambdas.append(info["lambda"].cpu())
            else:
                logits = model(batch["tokens"], batch["mask"])
            if device.type == "cuda":
                torch.cuda.synchronize()
            records[name]["latency"] += time.perf_counter() - started
            records[name]["logits"].append(logits.cpu())
        completed += n

    y = torch.cat(targets)
    result = {"models": {}}
    logits_by_name = {}
    for name, record in records.items():
        logits = torch.cat(record["logits"])
        logits_by_name[name] = logits
        result["models"][name] = {
            "accuracy": float(logits.argmax(-1).eq(y).float().mean()),
            "loss": float(F.cross_entropy(logits, y)),
            "mean_inference_latency_ms_per_example": 1000.0 * record["latency"] / examples,
        }
    pg = logits_by_name["gru"].softmax(-1)
    pl = logits_by_name["linear"].softmax(-1)
    probability_logits = ((pg + pl) / 2).clamp_min(1e-12).log()
    logit_average = (logits_by_name["gru"] + logits_by_name["linear"]) / 2
    for name, logits in {
        "probability_ensemble": probability_logits,
        "logit_ensemble": logit_average,
    }.items():
        result["models"][name] = {
            "accuracy": float(logits.argmax(-1).eq(y).float().mean()),
            "loss": float(F.cross_entropy(logits, y)),
            "mean_inference_latency_ms_per_example": (
                result["models"]["gru"]["mean_inference_latency_ms_per_example"]
                + result["models"]["linear"]["mean_inference_latency_ms_per_example"]
            ),
        }
    for name, parts in heuristic_predictions.items():
        pred = torch.cat(parts)
        logits = torch.zeros(examples, generator.vocab.num_entities)
        logits.scatter_(1, pred[:, None], 1.0)
        result["models"][name] = {
            "accuracy": float(pred.eq(y).float().mean()),
            "loss": float(F.cross_entropy(logits, y)),
            "mean_inference_latency_ms_per_example": 0.0,
        }
    gp, lp = logits_by_name["gru"].argmax(-1), logits_by_name["linear"].argmax(-1)
    hp = logits_by_name["hybrid"].argmax(-1)
    result["complementarity"] = complementarity(gp, lp, y)
    result["rescue_damage"] = rescue_damage(gp, hp, y)

    # Batch sequence lengths vary, so pad diagnostic tensors before concatenation.
    max_t = max(t.shape[1] for t in masks)
    def pad2(t, fill=0):
        return F.pad(t, (0, max_t - t.shape[1]), value=fill)
    def pad3(t):
        return F.pad(t, (0, 0, 0, max_t - t.shape[1]))
    all_masks = torch.cat([pad2(t, False) for t in masks])
    all_roles = torch.cat([pad2(t, 0) for t in roles])
    all_lambdas = torch.cat([pad3(t) for t in hybrid_lambdas])
    result["fusion_diagnostics"] = fusion_diagnostics(
        all_lambdas,
        all_masks,
        all_roles,
        torch.cat(lengths),
        gp,
        lp,
        y,
    )
    return result

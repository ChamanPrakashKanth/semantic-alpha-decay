"""One-step on-policy self-correction using the model's own detached answer token."""

import argparse
import copy
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from statistics import mean, stdev

import torch
import torch.nn.functional as F

from data import BatchFactory, SHIFTS, VOCAB_SIZE
from data.selective_recall import Batch
from sadt.losses import retention_loss
from sadt.model import TinySADT

MODES = ("baseline", "learned", "constrained")
CASES = tuple(x for x in SHIFTS if x != "iid")
torch.set_num_threads(1)


def add_self_feedback(batch, predictions):
    """Append [detached prediction, repeated query] without adding parameters/tokens."""
    size = batch.tokens.shape[0]
    lengths = (~batch.padding_mask).sum(1)
    width = int(lengths.max()) + 2
    tokens = torch.zeros(size, width, dtype=batch.tokens.dtype, device=batch.tokens.device)
    padding = torch.ones(size, width, dtype=torch.bool, device=batch.tokens.device)
    qpos = lengths + 1
    for row, length in enumerate(lengths.tolist()):
        tokens[row, :length] = batch.tokens[row, :length]
        tokens[row, length] = predictions[row].detach()
        tokens[row, length + 1] = batch.tokens[row, batch.query_positions[row]]
        padding[row, :length + 2] = False
    causal = torch.tril(torch.ones(width, width, dtype=torch.bool, device=tokens.device))
    valid = causal[None] & ~padding[:, None, :] & ~padding[:, :, None]
    return Batch(tokens=tokens, targets=batch.targets, query_positions=qpos,
                 relevant_positions=batch.relevant_positions,
                 irrelevant_positions=batch.irrelevant_positions,
                 padding_mask=padding, valid_links=valid)


def gate_metrics(info, batch):
    if info["survival"] is None:
        return 1.0, 1.0
    gates = info["survival"].mean(1)
    rows = torch.arange(batch.tokens.shape[0], device=batch.tokens.device)
    rel = gates[rows, batch.query_positions, batch.relevant_positions].mean().item()
    irr = gates[rows, batch.query_positions, batch.irrelevant_positions].mean().item()
    return rel, irr


def evaluate_recursive(model, factory, split, batches=8, batch_size=128):
    model.eval(); first_correct = second_correct = total = 0
    agreements = wrong_to_right = right_to_wrong = 0
    first_rel = []; first_irr = []; second_rel = []; second_irr = []
    with torch.no_grad():
        for _ in range(batches):
            batch = factory.batch(batch_size, split)
            logits1, info1 = model(batch.tokens, batch.query_positions, 1.0, batch.padding_mask)
            prediction = logits1.argmax(-1)
            feedback = add_self_feedback(batch, prediction)
            logits2, info2 = model(feedback.tokens, feedback.query_positions, 1.0, feedback.padding_mask)
            prediction2 = logits2.argmax(-1)
            correct1, correct2 = prediction == batch.targets, prediction2 == batch.targets
            first_correct += correct1.sum().item(); second_correct += correct2.sum().item()
            agreements += (prediction2 == prediction).sum().item()
            wrong_to_right += ((~correct1) & correct2).sum().item()
            right_to_wrong += (correct1 & (~correct2)).sum().item(); total += batch_size
            r, i = gate_metrics(info1, batch); first_rel.append(r); first_irr.append(i)
            r, i = gate_metrics(info2, feedback); second_rel.append(r); second_irr.append(i)
    d1r, d1i, d2r, d2i = map(mean, (first_rel, first_irr, second_rel, second_irr))
    first_acc, second_acc = first_correct / total, second_correct / total
    return {"first_accuracy": first_acc, "recursive_accuracy": second_acc,
            "accuracy_change": second_acc - first_acc,
            "answer_agreement": agreements / total,
            "wrong_to_right": wrong_to_right / total,
            "right_to_wrong": right_to_wrong / total,
            "first_D_rel": d1r, "first_D_irr": d1i, "first_delta_D": d1r - d1i,
            "recursive_D_rel": d2r, "recursive_D_irr": d2i, "recursive_delta_D": d2r - d2i}


def train_recursive(seed, mode, steps=800, batch_size=128, keep_penalty=.01):
    random.seed(seed); torch.manual_seed(seed)
    baseline = TinySADT(VOCAB_SIZE, mode="baseline")
    initial = copy.deepcopy(baseline.state_dict())
    model = TinySADT(VOCAB_SIZE, mode=mode)
    shared = {k: v for k, v in initial.items()
              if k in model.state_dict() and model.state_dict()[k].shape == v.shape}
    model.load_state_dict(shared, strict=False)
    factory = BatchFactory(seed)
    training = [factory.batch(batch_size, "train") for _ in range(steps)]
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    for step, batch in enumerate(training, 1):
        exposure = min(1.0, step / max(1, steps // 2))
        logits1, info1 = model(batch.tokens, batch.query_positions, exposure, batch.padding_mask)
        feedback = add_self_feedback(batch, logits1.argmax(-1))
        logits2, info2 = model(feedback.tokens, feedback.query_positions, exposure, feedback.padding_mask)
        loss = (F.cross_entropy(logits1, batch.targets) + F.cross_entropy(logits2, batch.targets)) / 2
        keep1, keep2 = retention_loss(info1["survival"], batch.valid_links), retention_loss(info2["survival"], feedback.valid_links)
        if keep1 is not None:
            loss = loss + keep_penalty * (keep1 + keep2) / 2
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    cases = {case: evaluate_recursive(model, BatchFactory(seed + 100_000 + index), case, batch_size=batch_size)
             for index, case in enumerate(("iid",) + CASES)}
    return {"seed": seed, "mode": mode, "cases": cases}


def run_seed(seed, steps, batch_size, keep_penalty):
    return [train_recursive(seed, mode, steps, batch_size, keep_penalty) for mode in MODES]


def stat(values):
    return {"mean": mean(values), "std": stdev(values) if len(values) > 1 else 0.0}


def summarize(runs):
    metrics = ("first_accuracy", "recursive_accuracy", "accuracy_change",
               "answer_agreement", "wrong_to_right", "right_to_wrong",
               "first_D_rel", "first_D_irr", "first_delta_D",
               "recursive_D_rel", "recursive_D_irr", "recursive_delta_D")
    return {mode: {case: {metric: stat([r["cases"][case][metric] for r in runs if r["mode"] == mode])
                          for metric in metrics} for case in ("iid",) + CASES} for mode in MODES}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5); p.add_argument("--steps", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=128); p.add_argument("--keep-penalty", type=float, default=.01)
    p.add_argument("--workers", type=int, default=1); p.add_argument("--output", default="results/recursive_report.json")
    args = p.parse_args(); runs = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = [pool.submit(run_seed, seed, args.steps, args.batch_size, args.keep_penalty) for seed in range(args.seeds)]
            for job in as_completed(jobs): runs.extend(job.result())
    else:
        for seed in range(args.seeds): runs.extend(run_seed(seed, args.steps, args.batch_size, args.keep_penalty))
    runs.sort(key=lambda r: (r["seed"], r["mode"]))
    payload = {"method": "append detached first answer and repeat query; supervise second answer",
               "config": vars(args), "runs": runs, "summary": summarize(runs)}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f: json.dump(payload, f, indent=2)
    for mode in MODES:
        print(mode)
        for case in ("iid",) + CASES:
            row = payload["summary"][mode][case]
            print(f"  {case:20s} first={row['first_accuracy']['mean']:.3f} recursive={row['recursive_accuracy']['mean']:.3f} change={row['accuracy_change']['mean']:+.3f} delta_D={row['recursive_delta_D']['mean']:+.4f}")


if __name__ == "__main__": main()

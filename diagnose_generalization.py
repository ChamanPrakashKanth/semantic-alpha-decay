"""Locate where semantic selectivity stops transferring. No scale or tuning sweep."""

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
from sadt.losses import retention_loss
from sadt.model import TinySADT
from train import evaluate

MODES = ("baseline", "fixed", "learned", "constrained")
CASES = tuple(x for x in SHIFTS if x != "iid")
torch.set_num_threads(1)


def fit_and_diagnose(seed, mode, steps=800, batch_size=128, keep_penalty=.01):
    random.seed(seed); torch.manual_seed(seed)
    baseline = TinySADT(VOCAB_SIZE, mode="baseline")
    initial = copy.deepcopy(baseline.state_dict())
    model = TinySADT(VOCAB_SIZE, mode=mode)
    shared = {k: v for k, v in initial.items()
              if k in model.state_dict() and model.state_dict()[k].shape == v.shape}
    model.load_state_dict(shared, strict=False)

    factory = BatchFactory(seed)
    batches = [factory.batch(batch_size, "train") for _ in range(steps)]
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    for step, batch in enumerate(batches, 1):
        logits, info = model(batch.tokens, batch.query_positions,
                             min(1.0, step / max(1, steps // 2)), batch.padding_mask)
        ce = F.cross_entropy(logits, batch.targets)
        keep = retention_loss(info["survival"], batch.valid_links)
        loss = ce if keep is None else ce + keep_penalty * keep
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    cases = {}
    for index, case in enumerate(("iid",) + CASES):
        cases[case] = evaluate(model, BatchFactory(seed + 100_000 + index), case,
                               batches=8, batch_size=batch_size)
    return {"seed": seed, "mode": mode, "cases": cases}


def run_seed(seed, modes, steps, batch_size, keep_penalty):
    return [fit_and_diagnose(seed, mode, steps, batch_size, keep_penalty) for mode in modes]


def stats(values):
    return {"mean": mean(values), "std": stdev(values) if len(values) > 1 else 0.0}


def summarize(runs):
    result = {}
    for mode in MODES:
        rows = [r for r in runs if r["mode"] == mode]
        result[mode] = {}
        for case in ("iid",) + CASES:
            result[mode][case] = {metric: stats([r["cases"][case][metric] for r in rows])
                                  for metric in ("accuracy", "cross_entropy", "D_rel", "D_irr", "delta_D")}
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5); p.add_argument("--steps", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=128); p.add_argument("--keep-penalty", type=float, default=.01)
    p.add_argument("--workers", type=int, default=1); p.add_argument("--output", default="results/generalization_report.json")
    args = p.parse_args(); runs = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = [pool.submit(run_seed, seed, MODES, args.steps, args.batch_size, args.keep_penalty)
                    for seed in range(args.seeds)]
            for job in as_completed(jobs): runs.extend(job.result())
    else:
        for seed in range(args.seeds): runs.extend(run_seed(seed, MODES, args.steps, args.batch_size, args.keep_penalty))
    runs.sort(key=lambda r: (r["seed"], r["mode"]))
    payload = {"question": "Does D_rel > D_irr remain true under isolated OOD shifts?",
               "config": vars(args), "cases": list(CASES), "runs": runs, "summary": summarize(runs)}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f: json.dump(payload, f, indent=2)
    for mode in MODES:
        print(mode)
        for case in CASES:
            row = payload["summary"][mode][case]
            print(f"  {case:20s} acc={row['accuracy']['mean']:.3f} delta_D={row['delta_D']['mean']:+.4f}")


if __name__ == "__main__": main()

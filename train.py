import argparse
import copy
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from statistics import mean, stdev

import torch
import torch.nn.functional as F

from data import BatchFactory, VOCAB_SIZE
from sadt.losses import retention_loss
from sadt.model import TinySADT

MODES = ("baseline", "learned", "learned_no_penalty", "random", "fixed", "renorm", "pre_softmax")

# Tiny tensor workloads are faster and more reproducible without large thread pools.
torch.set_num_threads(1)


def evaluate(model, factory, split, batches=8, batch_size=128, intervention=None):
    model.eval(); losses = []; correct = total = 0; rel = []; irr = []
    with torch.no_grad():
        for _ in range(batches):
            b = factory.batch(batch_size, split)
            logits, info = model(b.tokens, b.query_positions, 1.0, b.padding_mask,
                                 intervention, b.relevant_positions)
            losses.append(F.cross_entropy(logits, b.targets).item())
            correct += (logits.argmax(-1) == b.targets).sum().item(); total += batch_size
            if info["survival"] is not None:
                g = info["survival"].mean(1)
                rows = torch.arange(batch_size)
                rel.append(g[rows, b.query_positions, b.relevant_positions].mean().item())
                irr.append(g[rows, b.query_positions, b.irrelevant_positions].mean().item())
            else:
                rel.append(1.0); irr.append(1.0)
    out = {"accuracy": correct / total, "cross_entropy": mean(losses)}
    if rel:
        d_rel, d_irr = mean(rel), mean(irr)
        out.update(D_rel=d_rel, D_irr=d_irr, delta_D=d_rel - d_irr,
                   relevant_survival=d_rel, irrelevant_survival=d_irr)
    return out


def run(seed, mode, steps, batch_size, keep_penalty):
    random.seed(seed); torch.manual_seed(seed)
    base = TinySADT(VOCAB_SIZE, mode="baseline")
    initial = copy.deepcopy(base.state_dict())
    actual = "learned" if mode == "learned_no_penalty" else mode
    model = TinySADT(VOCAB_SIZE, mode=actual)
    common = {k: v for k, v in initial.items() if k in model.state_dict() and model.state_dict()[k].shape == v.shape}
    model.load_state_dict(common, strict=False)
    factory = BatchFactory(seed)
    batches = [factory.batch(batch_size, "train") for _ in range(steps)]
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    last_ce = last_reg = 0.0
    model.train()
    for step, b in enumerate(batches, 1):
        logits, info = model(b.tokens, b.query_positions, min(1.0, step / max(1, steps // 2)), b.padding_mask)
        ce = F.cross_entropy(logits, b.targets)
        reg = retention_loss(info["survival"], b.valid_links)
        weight = keep_penalty if mode not in {"baseline", "learned_no_penalty"} else 0.0
        loss = ce if reg is None else ce + weight * reg
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        last_ce, last_reg = ce.item(), 0.0 if reg is None else reg.item()
    eval_factory = BatchFactory(seed + 100_000)
    result = {"seed": seed, "mode": mode, "train_ce": last_ce, "train_regularizer": last_reg,
              "iid": evaluate(model, eval_factory, "train"), "ood": evaluate(model, eval_factory, "ood")}
    if actual in {"learned", "renorm", "pre_softmax"}:
        result["interventions"] = {name: evaluate(model, BatchFactory(seed + 200_000), "ood", intervention=name)
                                   for name in ("zero_relevant", "shuffle", "random", "force_one")}
    return result


def summarize(results):
    summary = {}
    for mode in sorted({r["mode"] for r in results}):
        rows = [r for r in results if r["mode"] == mode]
        summary[mode] = {}
        for split in ("iid", "ood"):
            for metric in ("accuracy", "cross_entropy", "relevant_survival", "irrelevant_survival"):
                vals = [r[split][metric] for r in rows if metric in r[split]]
                if vals:
                    summary[mode][split + "_" + metric] = {
                        "mean": mean(vals), "std": stdev(vals) if len(vals) > 1 else 0.0
                    }
        names = sorted({name for r in rows for name in r.get("interventions", {})})
        for name in names:
            vals = [r["interventions"][name]["accuracy"] for r in rows]
            summary[mode]["intervention_" + name + "_ood_accuracy"] = {
                "mean": mean(vals), "std": stdev(vals) if len(vals) > 1 else 0.0
            }
    return summary


def run_seed(seed, modes, steps, batch_size, keep_penalty):
    return [run(seed, mode, steps, batch_size, keep_penalty) for mode in modes]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5); p.add_argument("--steps", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=128); p.add_argument("--keep-penalty", type=float, default=.01)
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES)); p.add_argument("--output", default="results/report.json")
    p.add_argument("--workers", type=int, default=1, help="parallel seed workers")
    a = p.parse_args(); results = []
    if a.workers > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            jobs = [pool.submit(run_seed, seed, a.modes, a.steps, a.batch_size, a.keep_penalty)
                    for seed in range(a.seeds)]
            for job in as_completed(jobs):
                results.extend(job.result())
    else:
        for seed in range(a.seeds):
            results.extend(run_seed(seed, a.modes, a.steps, a.batch_size, a.keep_penalty))
    results.sort(key=lambda r: (r["seed"], r["mode"]))
    for r in results:
        print(r["seed"], r["mode"], "iid", r["iid"]["accuracy"], "ood", r["ood"]["accuracy"])
    payload = {"config": vars(a), "runs": results, "summary": summarize(results)}
    import os; os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
    with open(a.output, "w", encoding="utf-8") as f: json.dump(payload, f, indent=2)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__": main()

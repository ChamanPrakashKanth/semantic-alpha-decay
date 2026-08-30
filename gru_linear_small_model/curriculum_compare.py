from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

from .curriculum import CurriculumStage, curriculum_stages, make_batch, score
from .data import RelationChainGenerator
from .models import GRUBaseline, HybridGRULinear, count_parameters
from .training import resolve_device, seed_everything


def compact_stages() -> List[CurriculumStage]:
    caps = [150, 250, 350, 500, 600, 800, 1000]
    return [replace(stage, max_steps=caps[stage.stage]) for stage in curriculum_stages("pilot")]


def _fingerprint(batch) -> str:
    digest = hashlib.sha256()
    digest.update(batch["tokens"].detach().cpu().numpy().tobytes())
    digest.update(batch["targets"].detach().cpu().numpy().tobytes())
    return digest.hexdigest()[:16]


def _score_hybrid(model, batch):
    model.eval()
    with torch.no_grad():
        logits, info = model(batch["tokens"], batch["mask"], return_diagnostics=True)
        valid = batch["mask"].unsqueeze(-1).expand_as(info["lambda"])
        values = info["lambda"][valid]
        return {
            "accuracy": float(logits.argmax(-1).eq(batch["targets"]).float().mean()),
            "loss": float(F.cross_entropy(logits, batch["targets"])),
            "mean_lambda": float(values.mean()),
            "std_lambda": float(values.std(unbiased=False)),
        }


def _plot(path: Path, reports: List[Dict], chance: float):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    xs = [item["stage"] for item in reports]
    plt.figure(figsize=(8, 4.5))
    for model_name, label in [("gru", "GRU"), ("hybrid", "Hybrid")]:
        ys = [item["corrected_development"][model_name]["accuracy"] for item in reports]
        plt.plot(xs, ys, marker="o", label=label)
    plt.axhline(chance, linestyle="--", color="gray", label=f"chance ({chance:.3f})")
    plt.xlabel("Curriculum stage attempted")
    plt.ylabel("Final corrected development accuracy")
    plt.xticks(xs)
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _diagnosis(report: Dict) -> str:
    lines = [
        "# Low-compute curriculum comparison",
        "",
        "GRU and Hybrid were trained together on the exact same minibatch at every step. GRU validation alone controlled stage advancement, so the Hybrid could not alter the shared schedule.",
        "",
        f"Chance accuracy: **{report['chance_accuracy']:.2%}**.",
        f"Total shared training steps: **{report['total_steps']}**.",
        "",
        "| Stage | Steps | GRU stage val | Hybrid stage val | GRU corrected dev | Hybrid corrected dev | Mean lambda | Advanced |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for item in report["stage_reports"]:
        lines.append(
            f"| {item['stage']} | {item['steps_used']} | "
            f"{item['stage_validation']['gru']['accuracy']:.2%} | "
            f"{item['stage_validation']['hybrid']['accuracy']:.2%} | "
            f"{item['corrected_development']['gru']['accuracy']:.2%} | "
            f"{item['corrected_development']['hybrid']['accuracy']:.2%} | "
            f"{item['corrected_development']['hybrid']['mean_lambda']:.3f} | "
            f"{'yes' if item['advanced'] else 'no'} |"
        )
    failed = report["stage_reports"][-1]
    lines += ["", "## Verdict", ""]
    if report["all_stages_passed"]:
        lines.append("The GRU cleared every curriculum gate; corrected-development transfer determines whether a larger comparison is justified.")
    else:
        lines.append(
            f"GRU reached its learnability boundary at stage {failed['stage']} (`{failed['name']}`). The comparison stopped there to conserve compute."
        )
    gru_final = failed["corrected_development"]["gru"]["accuracy"]
    hybrid_final = failed["corrected_development"]["hybrid"]["accuracy"]
    lines.append(
        f"At the stopping point, corrected-development accuracy was {gru_final:.2%} for GRU and {hybrid_final:.2%} for Hybrid."
    )
    lines += [
        "",
        "The reserved corrected test seed was never generated or evaluated. This remains a development-stage learnability result, not a final architecture claim.",
    ]
    return "\n".join(lines) + "\n"


def run_comparison(
    output_dir: Path,
    seed: int = 0,
    stages_override: Sequence[CurriculumStage] = (),
    num_entities: int = 24,
    embedding_dim: int = 24,
    hidden_dim: int = 24,
    batch_size: int = 32,
    validation_examples: int = 256,
    final_probe_examples: int = 512,
    validate_every: int = 25,
    required_consecutive_passes: int = 2,
    learning_rate: float = 3e-3,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    device = resolve_device("auto")
    stages = list(stages_override) if stages_override else compact_stages()
    model_args = (num_entities + 4, num_entities, embedding_dim, hidden_dim)
    models = {
        "gru": GRUBaseline(*model_args).to(device),
        "hybrid": HybridGRULinear(*model_args).to(device),
    }
    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
        for name, model in models.items()
    }
    corrected_development = make_batch(
        stages[-1], final_probe_examples, 880000 + seed, num_entities, device
    )
    initial = {
        "gru": score(models["gru"], corrected_development),
        "hybrid": _score_hybrid(models["hybrid"], corrected_development),
    }
    reports = []
    total_steps = 0
    started = time.perf_counter()
    for stage in stages:
        generator = RelationChainGenerator(
            num_entities, 20000 + seed * 100 + stage.stage, balance_destinations=False
        )
        validation = make_batch(
            stage, validation_examples, 10000 + seed * 100 + stage.stage, num_entities, device
        )
        history = []
        first_batch_fingerprint = None
        consecutive = 0
        advanced = False
        stage_started = time.perf_counter()
        for local_step in range(1, stage.max_steps + 1):
            batch = generator.sample_batch(
                batch_size,
                stage.chain_lengths,
                stage.distractors,
                device,
                edge_order=stage.edge_order,
                destination_coverage=stage.destination_coverage,
            )
            if first_batch_fingerprint is None:
                first_batch_fingerprint = _fingerprint(batch)
            # Both models consume this exact batch object before it is discarded.
            for name, model in models.items():
                model.train()
                optimizer = optimizers[name]
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch["tokens"], batch["mask"])
                loss = F.cross_entropy(logits, batch["targets"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_steps += 1
            if local_step % validate_every == 0 or local_step == stage.max_steps:
                current = {
                    "local_step": local_step,
                    "total_step": total_steps,
                    "gru": score(models["gru"], validation),
                    "hybrid": _score_hybrid(models["hybrid"], validation),
                }
                history.append(current)
                print(
                    f"stage {stage.stage} step {local_step}/{stage.max_steps} "
                    f"gru={current['gru']['accuracy']:.2%} hybrid={current['hybrid']['accuracy']:.2%}",
                    flush=True,
                )
                consecutive = consecutive + 1 if current["gru"]["accuracy"] >= stage.threshold else 0
                if consecutive >= required_consecutive_passes:
                    advanced = True
                    break
        stage_validation = {
            "gru": score(models["gru"], validation),
            "hybrid": _score_hybrid(models["hybrid"], validation),
        }
        corrected = {
            "gru": score(models["gru"], corrected_development),
            "hybrid": _score_hybrid(models["hybrid"], corrected_development),
        }
        reports.append({
            **asdict(stage),
            "steps_used": local_step,
            "total_steps": total_steps,
            "advanced": advanced,
            "stage_validation": stage_validation,
            "corrected_development": corrected,
            "validation_history": history,
            "training_stream_seed": 20000 + seed * 100 + stage.stage,
            "first_shared_batch_fingerprint": first_batch_fingerprint,
            "stage_training_seconds": time.perf_counter() - stage_started,
        })
        if not advanced:
            break
    all_passed = len(reports) == len(stages) and all(item["advanced"] for item in reports)
    result = {
        "experiment": "low_compute_gru_hybrid_curriculum_comparison",
        "seed": seed,
        "device": str(device),
        "chance_accuracy": 1.0 / num_entities,
        "shared_batch_training": True,
        "stage_controller": "gru_validation_only",
        "models": {
            name: {"parameter_count": count_parameters(model)} for name, model in models.items()
        },
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "batch_size": batch_size,
        "validation_examples": validation_examples,
        "final_probe_examples": final_probe_examples,
        "initial_corrected_development": initial,
        "stage_reports": reports,
        "all_stages_passed": all_passed,
        "total_steps": total_steps,
        "wall_clock_seconds": time.perf_counter() - started,
        "corrected_development_seed": 880000 + seed,
        "reserved_untouched_test_seed": 990000 + seed,
        "reserved_test_was_evaluated": False,
    }
    (output_dir / "comparison_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    schedule = {
        "controller": "gru_validation_only",
        "shared_batches": True,
        "seed": seed,
        "stages": [
            {
                "stage": item["stage"],
                "name": item["name"],
                "steps_used": item["steps_used"],
                "training_stream_seed": item["training_stream_seed"],
                "first_batch_fingerprint": item["first_shared_batch_fingerprint"],
                "advanced": item["advanced"],
            }
            for item in reports
        ],
    }
    (output_dir / "shared_schedule.json").write_text(json.dumps(schedule, indent=2), encoding="utf-8")
    with (output_dir / "stage_results.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "stage", "name", "steps_used", "advanced",
            "gru_stage_accuracy", "hybrid_stage_accuracy",
            "gru_corrected_accuracy", "hybrid_corrected_accuracy", "hybrid_mean_lambda",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in reports:
            writer.writerow({
                "stage": item["stage"],
                "name": item["name"],
                "steps_used": item["steps_used"],
                "advanced": item["advanced"],
                "gru_stage_accuracy": item["stage_validation"]["gru"]["accuracy"],
                "hybrid_stage_accuracy": item["stage_validation"]["hybrid"]["accuracy"],
                "gru_corrected_accuracy": item["corrected_development"]["gru"]["accuracy"],
                "hybrid_corrected_accuracy": item["corrected_development"]["hybrid"]["accuracy"],
                "hybrid_mean_lambda": item["corrected_development"]["hybrid"]["mean_lambda"],
            })
    (output_dir / "diagnosis.md").write_text(_diagnosis(result), encoding="utf-8")
    torch.save(
        {"model_state_dict": {name: model.state_dict() for name, model in models.items()}, "report": result},
        output_dir / "gru_hybrid_curriculum.pt",
    )
    _plot(output_dir / "corrected_transfer_curve.png", reports, result["chance_accuracy"])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results" / "curriculum_compare_compact",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_comparison(args.output, seed=args.seed)


if __name__ == "__main__":
    main()

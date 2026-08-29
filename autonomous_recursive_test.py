"""Autonomous Recursive Teacher-Free Evaluation for RSL.

Tests the user's core hypothesis:
"Training RSL with RL isn't itself novel self-learning. The interesting claim
begins only if the learned correction dynamics continue to produce useful
recursive correction after the RL teacher has been removed."

Evaluates:
1. Multi-pass autonomous recursion (k in {1, 2, 3, 5}) without any external reward teacher.
2. Tracks whether internal representation updates produce:
   - Monotonic accuracy improvements (Acc_{k+1} >= Acc_k)
   - Rescue vs damage transition dynamics: c_k = P_k(W -> R) vs d_k = P_k(R -> W)
   - Self-reinforcing error loop detection (d_k > c_k)
   - Representation drift ||h^(k) - h^(k-1)||
   - Gate / survival evolution under multi-pass recursion.
3. Tests both learned-gate and forced-intervention regimes (g in {0.0, 0.2, 0.5, 0.8, 1.0}).
"""

import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from statistics import mean, median, stdev
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from controlled_competence_rsl import (
    ControlledCompetenceModel,
    ThreeSectorAngularEnvironment,
    train_base_model_to_competence,
    train_rsl_on_frozen_base,
)

torch.set_num_threads(1)


# =============================================================================
# Multi-Pass Recursive Forward Execution (Teacher-Free)
# =============================================================================

def recursive_teacher_free_forward(
    model: ControlledCompetenceModel,
    tokens: torch.Tensor,
    query_positions: torch.Tensor,
    k_passes: int = 5,
    padding_mask: Optional[torch.Tensor] = None,
    forced_gate: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Execute k recursive internal passes through the model and controller.

    No ground-truth labels, losses, or external rewards are used.
    """
    b, l, _ = tokens.shape
    pos = torch.arange(l, device=tokens.device)
    x0 = model.input_proj(tokens) + model.pos_emb(pos)
    h_current = model.ln1(x0)
    rows = torch.arange(b, device=tokens.device)

    pass_results = []

    for step in range(1, k_passes + 1):
        q, k, v, A, invalid = model.compute_attention(h_current, padding_mask)

        # Base pass prediction at step k
        y_step = A @ v
        y_step = y_step.transpose(1, 2).contiguous().view(b, l, model.d_model)
        x_step = x0 + model.out(y_step)
        x_step = x_step + model.ff(model.ln2(x_step))
        logits_k = model.head(model.final_ln(x_step[rows, query_positions]))

        # Legitimate inference uncertainty features
        uncertainty = model.extract_uncertainty(logits_k.detach(), A.detach())

        # Controller prediction
        g, alpha, raw_d, eff_d, _ = model.controller(
            q.detach(),
            k.detach(),
            A.detach(),
            uncertainty=uncertainty,
            exposure=1.0,
            invalid_mask=invalid,
        )

        if forced_gate is not None:
            # Override gate to test intervention dynamics under forced activity
            g = torch.full_like(g, forced_gate)
            eff_d = (1.0 - g) + g * raw_d
            eff_d = eff_d.masked_fill(invalid.expand_as(eff_d), 0.0)

        # Apply decay to update representation for next pass
        decayed_A = A * eff_d
        y_rsl = decayed_A @ v
        y_rsl = y_rsl.transpose(1, 2).contiguous().view(b, l, model.d_model)
        x_rsl = x0 + model.out(y_rsl)
        x_rsl = x_rsl + model.ff(model.ln2(x_rsl))
        rsl_logits_k = model.head(model.final_ln(x_rsl[rows, query_positions]))

        # State update for next recursive pass: update internal representations
        h_next = model.ln1(x_rsl)
        drift = (h_next - h_current).norm(dim=-1).mean().item()
        h_current = h_next.detach()

        valid_mask = ~invalid.expand_as(eff_d)
        pass_results.append({
            "pass": step,
            "logits": rsl_logits_k,
            "predictions": rsl_logits_k.argmax(-1),
            "mean_gate": g[valid_mask].mean().item(),
            "mean_effective_D": eff_d[valid_mask].mean().item(),
            "representation_drift": drift,
        })

    return pass_results


# =============================================================================
# Autonomous Multi-Pass Benchmark
# =============================================================================

def evaluate_autonomous_recursion(
    model: ControlledCompetenceModel,
    env: ThreeSectorAngularEnvironment,
    split: str = "test_sector",
    k_passes: int = 5,
    batches: int = 10,
    batch_size: int = 128,
    forced_gate: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluate autonomous multi-pass recursion without external teacher feedback."""
    model.eval()
    total = 0

    pass_stats = [
        {"correct": 0, "w_to_r": 0, "r_to_w": 0, "gates": [], "eff_d": [], "drifts": []}
        for _ in range(k_passes)
    ]
    base_correct_total = 0

    with torch.no_grad():
        for _ in range(batches):
            b = env.generate_batch(batch_size, split=split)
            b_size = b.tokens.shape[0]
            total += b_size

            # Initial 1-pass base model prediction
            base_logits, _, _ = model(
                b.tokens, b.query_positions, exposure=1.0, padding_mask=b.padding_mask, apply_rsl=False
            )
            base_preds = base_logits.argmax(-1)
            b_corr = base_preds == b.targets
            base_correct_total += b_corr.sum().item()

            # Execute autonomous recursive reasoning
            pass_outputs = recursive_teacher_free_forward(
                model,
                b.tokens,
                b.query_positions,
                k_passes=k_passes,
                padding_mask=b.padding_mask,
                forced_gate=forced_gate,
            )

            prev_corr = b_corr
            for k in range(k_passes):
                preds_k = pass_outputs[k]["predictions"]
                corr_k = preds_k == b.targets
                pass_stats[k]["correct"] += corr_k.sum().item()
                # Transitions relative to previous pass
                pass_stats[k]["w_to_r"] += ((~prev_corr) & corr_k).sum().item()
                pass_stats[k]["r_to_w"] += (prev_corr & (~corr_k)).sum().item()
                pass_stats[k]["gates"].append(pass_outputs[k]["mean_gate"])
                pass_stats[k]["eff_d"].append(pass_outputs[k]["mean_effective_D"])
                pass_stats[k]["drifts"].append(pass_outputs[k]["representation_drift"])
                prev_corr = corr_k

    base_p = base_correct_total / total
    summary_passes = []
    prev_acc = base_p

    for k in range(k_passes):
        acc_k = pass_stats[k]["correct"] / total
        delta_from_base = acc_k - base_p
        delta_step = acc_k - prev_acc
        w_to_r = pass_stats[k]["w_to_r"]
        r_to_w = pass_stats[k]["r_to_w"]
        c_k = (w_to_r / (total - int(prev_acc * total))) if (total > int(prev_acc * total)) else 0.0
        d_k = (r_to_w / int(prev_acc * total)) if (int(prev_acc * total) > 0) else 0.0

        is_error_loop = (acc_k < prev_acc) and (r_to_w > w_to_r)

        summary_passes.append({
            "pass_k": k + 1,
            "accuracy": acc_k,
            "delta_from_base": delta_from_base,
            "delta_step": delta_step,
            "P_W_to_R_c": c_k,
            "P_R_to_W_d": d_k,
            "damage_to_rescue_ratio": (d_k / c_k) if c_k > 1e-6 else float("inf"),
            "mean_gate": mean(pass_stats[k]["gates"]),
            "mean_effective_D": mean(pass_stats[k]["eff_d"]),
            "representation_drift": mean(pass_stats[k]["drifts"]),
            "error_loop_detected": is_error_loop,
        })
        prev_acc = acc_k

    return {
        "base_accuracy": base_p,
        "forced_gate": forced_gate,
        "passes": summary_passes,
    }


# =============================================================================
# Multi-Seed Autonomous Recursion Experiment
# =============================================================================

def run_autonomous_recursive_experiment(
    seeds: int = 10,
    rsl_steps: int = 400,
    k_passes: int = 5,
    competence_p: float = 0.70,  # Moderate competence with room for self-correction
) -> Dict[str, Any]:
    """Train base + RSL model and test autonomous teacher-free recursion across seeds."""
    env = ThreeSectorAngularEnvironment()
    print(f"\n=======================================================")
    print(f"Autonomous Teacher-Free Recursion Test ({seeds} seeds, k={k_passes} passes)")
    print(f"Base Competence p ~ {competence_p:.2f}")
    print(f"=======================================================")

    learned_gate_runs = []
    forced_intervention_runs = {0.2: [], 0.5: [], 1.0: []}

    for seed in range(seeds):
        seed_env = ThreeSectorAngularEnvironment(seed=seed)
        model = train_base_model_to_competence(competence_p, seed, seed_env)
        model = train_rsl_on_frozen_base(model, seed_env, rsl_steps=rsl_steps)

        eval_env = ThreeSectorAngularEnvironment(seed=seed + 100_000)

        # 1. Autonomous recursion with learned gate policy
        learned_res = evaluate_autonomous_recursion(
            model, eval_env, split="test_sector", k_passes=k_passes, forced_gate=None
        )
        learned_gate_runs.append({"seed": seed, "results": learned_res})

        # 2. Autonomous recursion with forced intervention gates
        for g_val in [0.2, 0.5, 1.0]:
            forced_res = evaluate_autonomous_recursion(
                model, eval_env, split="test_sector", k_passes=k_passes, forced_gate=g_val
            )
            forced_intervention_runs[g_val].append({"seed": seed, "results": forced_res})

        p1_acc = learned_res["passes"][0]["accuracy"]
        pk_acc = learned_res["passes"][-1]["accuracy"]
        f1_acc = forced_intervention_runs[0.5][-1]["results"]["passes"][-1]["accuracy"]
        print(
            f"Seed {seed:2d}: Base={learned_res['base_accuracy']:.3f} -> "
            f"Learned k=1:{p1_acc:.3f}, k={k_passes}:{pk_acc:.3f} | Forced (g=0.5) k={k_passes}:{f1_acc:.3f}"
        )

    return {
        "learned_gate_runs": learned_gate_runs,
        "forced_intervention_runs": forced_intervention_runs,
    }


def aggregate_recursive_passes(runs: List[Dict[str, Any]], k_passes: int = 5) -> List[Dict[str, Any]]:
    summary = []
    for k in range(k_passes):
        accs = [r["results"]["passes"][k]["accuracy"] for r in runs]
        deltas = [r["results"]["passes"][k]["delta_from_base"] for r in runs]
        cs = [r["results"]["passes"][k]["P_W_to_R_c"] for r in runs]
        ds = [r["results"]["passes"][k]["P_R_to_W_d"] for r in runs]
        drifts = [r["results"]["passes"][k]["representation_drift"] for r in runs]
        eff_ds = [r["results"]["passes"][k]["mean_effective_D"] for r in runs]
        loops = [r["results"]["passes"][k]["error_loop_detected"] for r in runs]

        summary.append({
            "pass_k": k + 1,
            "accuracy_mean": mean(accs),
            "accuracy_std": stdev(accs) if len(accs) > 1 else 0.0,
            "delta_mean": mean(deltas),
            "P_W_to_R_c_mean": mean(cs),
            "P_R_to_W_d_mean": mean(ds),
            "effective_D_mean": mean(eff_ds),
            "drift_mean": mean(drifts),
            "error_loop_rate": sum(1 for x in loops if x) / len(loops),
        })
    return summary


def main():
    parser = argparse.ArgumentParser(description="Autonomous Teacher-Free Recursion Benchmark")
    parser.add_argument("--seeds", type=int, default=10, help="Number of deterministic seeds")
    parser.add_argument("--rsl-steps", type=int, default=400, help="RSL training steps")
    parser.add_argument("--k-passes", type=int, default=5, help="Recursive passes k")
    parser.add_argument("--competence-p", type=float, default=0.70, help="Base competence p")
    parser.add_argument(
        "--output", default="results/autonomous_recursion_report.json", help="JSON report path"
    )
    parser.add_argument(
        "--csv-output", default="results/autonomous_recursion_table.csv", help="CSV table path"
    )
    args = parser.parse_args()

    data = run_autonomous_recursive_experiment(
        seeds=args.seeds,
        rsl_steps=args.rsl_steps,
        k_passes=args.k_passes,
        competence_p=args.competence_p,
    )

    learned_summary = aggregate_recursive_passes(data["learned_gate_runs"], k_passes=args.k_passes)
    forced_05_summary = aggregate_recursive_passes(
        data["forced_intervention_runs"][0.5], k_passes=args.k_passes
    )

    payload = {
        "title": "Autonomous Teacher-Free Recursive Self-Learning Benchmark",
        "scientific_question": (
            "Does the learned internal correction dynamics continue to produce useful recursive correction "
            "after the RL reward teacher is removed, or does multi-pass recursion collapse into error amplification?"
        ),
        "config": vars(args),
        "learned_gate_summary": learned_summary,
        "forced_g_05_summary": forced_05_summary,
        "raw_runs": data,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nReport saved to {args.output}")

    # Write CSV Table
    os.makedirs(os.path.dirname(args.csv_output) or ".", exist_ok=True)
    with open(args.csv_output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "mode",
            "pass_k",
            "accuracy_mean",
            "accuracy_std",
            "delta_mean",
            "P_W_to_R_c",
            "P_R_to_W_d",
            "effective_D",
            "drift",
            "error_loop_rate",
        ])
        for s in learned_summary:
            writer.writerow([
                "learned_gate",
                s["pass_k"],
                f"{s['accuracy_mean']:.4f}",
                f"{s['accuracy_std']:.4f}",
                f"{s['delta_mean']:+.4f}",
                f"{s['P_W_to_R_c_mean']:.4f}",
                f"{s['P_R_to_W_d_mean']:.4f}",
                f"{s['effective_D_mean']:.4f}",
                f"{s['drift_mean']:.4f}",
                f"{s['error_loop_rate']:.2f}",
            ])
        for s in forced_05_summary:
            writer.writerow([
                "forced_gate_0.5",
                s["pass_k"],
                f"{s['accuracy_mean']:.4f}",
                f"{s['accuracy_std']:.4f}",
                f"{s['delta_mean']:+.4f}",
                f"{s['P_W_to_R_c_mean']:.4f}",
                f"{s['P_R_to_W_d_mean']:.4f}",
                f"{s['effective_D_mean']:.4f}",
                f"{s['drift_mean']:.4f}",
                f"{s['error_loop_rate']:.2f}",
            ])
    print(f"CSV table saved to {args.csv_output}")


if __name__ == "__main__":
    main()

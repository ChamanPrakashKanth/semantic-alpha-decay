from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import torch

from .config import ExperimentConfig
from .data import RelationChainGenerator
from .evaluation import evaluate_mode
from .models import build_models
from .training import resolve_device, seed_everything, train_models_shared_data


def aggregate(values):
    values = [float(v) for v in values]
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    half = 1.96 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "median": statistics.median(values),
        "ci95": [mean - half, mean + half],
        "raw_seed_values": values,
    }


def _write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_chain_lengths(path: Path, rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    wanted = ["gru", "linear", "probability_ensemble", "hybrid", "param_matched_gru", "dual_state_hybrid"]
    plt.figure(figsize=(8, 5))
    for name in wanted:
        subset = sorted((r for r in rows if r["model"] == name), key=lambda r: r["chain_length"])
        if subset:
            plt.plot([r["chain_length"] for r in subset], [r["accuracy"] for r in subset], marker="o", label=name)
    plt.xlabel("Chain length")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _plot_fusion(path: Path, rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    grouped = {}
    for row in rows:
        grouped.setdefault(row["chain_length"], []).append(row["mean_lambda"])
    xs = sorted(grouped)
    ys = [statistics.fmean(grouped[x]) for x in xs]
    plt.figure(figsize=(7, 4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Chain length")
    plt.ylabel("Mean fusion lambda")
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _plot_fusion_conditions(path: Path, rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    names = [
        "gru_correct_linear_wrong",
        "gru_wrong_linear_correct",
        "both_correct",
        "both_wrong",
    ]
    values = []
    for name in names:
        available = [r[name] for r in rows if r.get(name) is not None]
        values.append(statistics.fmean(available) if available else 0.0)
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(names)), values)
    plt.xticks(range(len(names)), ["G✓/L✗", "G✗/L✓", "both ✓", "both ✗"])
    plt.ylabel("Mean fusion lambda")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def make_diagnosis(report):
    modes = report["modes"]
    chance = 1.0 / report["config"]["num_entities"]
    n_eval = report["config"]["eval_examples_per_mode"]
    sampling_band = 3.0 * math.sqrt(chance * (1.0 - chance) / n_eval)
    trainable = ["gru", "linear", "hybrid", "param_matched_gru", "dual_state_hybrid"]
    best_observed = max(
        modes[mode][name]["accuracy"]["mean"]
        for mode in modes
        for name in trainable
    )
    learned_signal = best_observed > chance + sampling_band
    shortcut_names = [
        "last_destination",
        "first_destination",
        "most_recent_entity",
        "most_frequent_entity",
        "random_class",
    ]
    max_shortcut = max(
        modes[mode][name]["accuracy"]["mean"]
        for mode in modes
        for name in shortcut_names
    )
    lines = [
        "# Experiment diagnosis",
        "",
        f"Protocol: **{report['protocol_label']}** with {len(report['config']['seeds'])} seed(s).",
        f"Class chance is {chance:.3%}.",
        "",
    ]
    for mode in ["iid", "noise_ood", "length_ood", "combined_ood"]:
        if mode not in modes:
            continue
        m = modes[mode]
        h, g = m["hybrid"]["accuracy"]["mean"], m["gru"]["accuracy"]["mean"]
        pm = m["param_matched_gru"]["accuracy"]["mean"]
        ens = m["probability_ensemble"]["accuracy"]["mean"]
        lines.append(
            f"- {mode}: Hybrid {h:.3%}, GRU {g:.3%}, parameter-matched GRU {pm:.3%}, "
            f"probability ensemble {ens:.3%}."
        )
    long_mode = modes.get("length_ood", modes["iid"])
    gain = long_mode["hybrid"]["accuracy"]["mean"] - long_mode["gru"]["accuracy"]["mean"]
    pm_gain = long_mode["hybrid"]["accuracy"]["mean"] - long_mode["param_matched_gru"]["accuracy"]["mean"]
    lines += [
        "",
        "## Answers",
        "",
        f"Learning signal beyond a conservative chance band: **{'yes' if learned_signal else 'no'}**.",
        f"Largest listed shortcut accuracy: **{max_shortcut:.3%}**.",
        "",
        f"1. Hybrid beat GRU on mean length OOD: **{'yes' if gain > 0 else 'no'}** ({gain:+.3%}).",
        f"2. Hybrid beat parameter-matched GRU there: **{'yes' if pm_gain > 0 else 'no'}** ({pm_gain:+.3%}).",
        "3. LinearSmooth and all ablations are reported in `aggregate_report.json`.",
        "4. Whether Hybrid matched the external ensemble is shown mode by mode above.",
        "5. Gains, if any, must be interpreted by evaluation mode rather than IID alone.",
        "6. The chain-length CSV records extrapolation through length 10.",
        "7. Fusion collapse is explicitly flagged in each mode's raw diagnostics.",
        "8. Complementarity counts and conditional rescue rates are preserved per seed.",
        "9. Oracle accuracy gives the available routing headroom.",
        "10. The strongest permitted claim is limited to this synthetic bounded-state benchmark.",
        "11. These results do not establish general reasoning, intelligence, or a Transformer replacement.",
    ]
    if report["protocol_label"] != "serious":
        lines += ["", "This run is exploratory and cannot satisfy the cookbook's serious success criteria."]
    if not learned_signal:
        lines += [
            "",
            "Because no trainable model cleared the conservative chance band, relative rankings in this run are sampling noise and must not be interpreted as architectural gains.",
        ]
    return "\n".join(lines) + "\n"


def run(config: ExperimentConfig, output_dir: Path, protocol_label: str, progress_every: int = 0):
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)
    seed_reports = []
    flat_rows, chain_rows, fusion_chain_rows, fusion_condition_rows = [], [], [], []
    for seed in config.seeds:
        print(f"seed {seed} on {device}", flush=True)
        seed_everything(seed, config.deterministic)
        models = build_models(config.num_entities + 4, config.num_entities, config.embedding_dim, config.hidden_dim)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        training = train_models_shared_data(
            models,
            RelationChainGenerator(
                config.num_entities,
                seed * 1000 + 1,
                balance_destinations=config.shortcut_balanced_destinations,
            ),
            config,
            device,
            progress_every,
        )
        mode_reports = {}
        for mode_index, (mode, (lengths, noise)) in enumerate(config.evaluation_modes().items()):
            mode_report = evaluate_mode(
                models,
                RelationChainGenerator(
                    config.num_entities,
                    seed * 1000 + 100 + mode_index,
                    balance_destinations=config.shortcut_balanced_destinations,
                ),
                lengths,
                noise,
                config.eval_examples_per_mode,
                config.batch_size,
                device,
            )
            mode_reports[mode] = mode_report
            for name, metrics in mode_report["models"].items():
                flat_rows.append({"seed": seed, "mode": mode, "model": name, **metrics})
            fusion_condition_rows.append({
                "seed": seed,
                "mode": mode,
                **mode_report["fusion_diagnostics"]["lambda_by_branch_correctness"],
            })
        chain_reports = {}
        per_length_examples = max(128, min(config.eval_examples_per_mode, 1000))
        for length in [2, 3, 4, 5, 6, 8, 10]:
            result = evaluate_mode(
                models,
                RelationChainGenerator(
                    config.num_entities,
                    seed * 1000 + 500 + length,
                    balance_destinations=config.shortcut_balanced_destinations,
                ),
                [length],
                [0, 1],
                per_length_examples,
                config.batch_size,
                device,
            )
            chain_reports[str(length)] = result
            for name, metrics in result["models"].items():
                chain_rows.append({"seed": seed, "chain_length": length, "model": name, **metrics})
            fusion_chain_rows.append({
                "seed": seed,
                "chain_length": length,
                "mean_lambda": result["fusion_diagnostics"]["mean_lambda"],
            })
        peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        seed_reports.append({
            "seed": seed,
            "training": training,
            "peak_cuda_memory_bytes": peak,
            "modes": mode_reports,
            "chain_lengths": chain_reports,
        })
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()

    aggregate_modes = {}
    model_names = sorted({r["model"] for r in flat_rows})
    for mode in config.evaluation_modes():
        aggregate_modes[mode] = {}
        for name in model_names:
            rows = [r for r in flat_rows if r["mode"] == mode and r["model"] == name]
            aggregate_modes[mode][name] = {
                metric: aggregate([r[metric] for r in rows])
                for metric in ["accuracy", "loss", "mean_inference_latency_ms_per_example"]
            }
    paired = {}
    for mode in config.evaluation_modes():
        paired[mode] = {}
        for reference in ["gru", "linear", "param_matched_gru", "probability_ensemble"]:
            deltas = []
            for seed in config.seeds:
                h = next(r["accuracy"] for r in flat_rows if r["seed"] == seed and r["mode"] == mode and r["model"] == "hybrid")
                b = next(r["accuracy"] for r in flat_rows if r["seed"] == seed and r["mode"] == mode and r["model"] == reference)
                deltas.append(h - b)
            paired[mode][f"hybrid_minus_{reference}"] = aggregate(deltas)
    parameter_counts = seed_reports[0]["training"]["parameter_counts"]
    report = {
        "protocol_label": protocol_label,
        "config": config.to_dict(),
        "device": str(device),
        "models": {name: {"parameter_count": count} for name, count in parameter_counts.items()},
        "modes": aggregate_modes,
        "paired_differences": paired,
        "parameter_counts": parameter_counts,
        "latency": {mode: {name: data["mean_inference_latency_ms_per_example"] for name, data in models.items()} for mode, models in aggregate_modes.items()},
        "chain_length_results": chain_rows,
        "fusion_diagnostics": [s["modes"] for s in seed_reports],
        "complementarity": {str(s["seed"]): {m: r["complementarity"] for m, r in s["modes"].items()} for s in seed_reports},
        "oracle_headroom": {str(s["seed"]): {m: r["complementarity"]["oracle_accuracy"] for m, r in s["modes"].items()} for s in seed_reports},
        "raw_seed_reports": seed_reports,
    }
    (output_dir / "aggregate_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "diagnosis.md").write_text(make_diagnosis(report), encoding="utf-8")
    _write_csv(output_dir / "seed_results.csv", flat_rows)
    _write_csv(output_dir / "chain_length_curve.csv", chain_rows)
    _write_csv(output_dir / "fusion_gate_by_chain_length.csv", fusion_chain_rows)
    _write_csv(output_dir / "fusion_gate_by_condition.csv", fusion_condition_rows)
    (output_dir / "complementarity_report.json").write_text(json.dumps(report["complementarity"], indent=2), encoding="utf-8")
    (output_dir / "oracle_headroom.json").write_text(json.dumps(report["oracle_headroom"], indent=2), encoding="utf-8")
    _plot_chain_lengths(output_dir / "chain_length_curve.png", chain_rows)
    _plot_fusion(output_dir / "fusion_gate_vs_chain_length.png", fusion_chain_rows)
    _plot_fusion_conditions(output_dir / "fusion_gate_by_condition.png", fusion_condition_rows)
    return report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["smoke", "exploratory", "serious"], default="smoke")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / "smoke")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seeds", type=int)
    parser.add_argument("--eval-examples", type=int)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def main():
    args = parse_args()
    config = {
        "smoke": ExperimentConfig.smoke,
        "exploratory": ExperimentConfig.exploratory,
        "serious": ExperimentConfig,
    }[args.preset]()
    if args.steps is not None:
        config.steps = args.steps
    if args.seeds is not None:
        config.seeds = tuple(range(args.seeds))
    if args.eval_examples is not None:
        config.eval_examples_per_mode = args.eval_examples
    run(config, args.output, args.preset, args.progress_every)


if __name__ == "__main__":
    main()

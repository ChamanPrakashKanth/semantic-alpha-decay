from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import RelationChainGenerator
from .models import build_models
from .training import overfit_tiny_batch, resolve_device, seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / "tiny_overfit.json")
    args = parser.parse_args()
    seed_everything(0)
    device = resolve_device("auto")
    results = {}
    for index, (name, model) in enumerate(build_models(28, 24, 16, 16).items()):
        accuracy, loss, passed = overfit_tiny_batch(
            # Tiny-batch overfitting is a cell/gradient sanity check, so omit the
            # 24-destination camouflage used by the scientific benchmark.
            model, RelationChainGenerator(24, 100 + index, balance_destinations=False), device, args.steps
        )
        results[name] = {"accuracy": accuracy, "loss": loss, "passed": passed}
        print(name, results[name], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if not all(item["passed"] for item in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

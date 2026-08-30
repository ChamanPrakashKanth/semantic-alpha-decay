# GRU + LinearSmooth benchmark

This project implements the controlled synthetic experiment described in
`gru_linear_small_model_agent_cookbook.md`. It is deliberately framed as a test
of bounded-state sequence processing, not general reasoning.

## Run

From the parent workspace:

```powershell
python -m gru_linear_small_model.run_experiment --preset smoke
python -m pytest gru_linear_small_model/tests -q
```

Larger protocols:

```powershell
python -m gru_linear_small_model.run_experiment --preset exploratory --output gru_linear_small_model/results/exploratory
python -m gru_linear_small_model.run_experiment --preset serious --output gru_linear_small_model/results/serious
```

The serious preset is 10 seeds, 1,500 training steps, and 5,000 examples per
evaluation mode. It is intentionally expensive. Smoke results must not be
reported as confirmatory evidence.

## Implemented models

- GRU and LinearSmooth baselines
- external probability and logit ensembles
- learned vector-gated Hybrid
- fixed-half, forced-GRU, forced-Linear, random, and scalar-gate ablations
- parameter-matched GRU
- DualStateHybrid
- SmoothedGRU

All trainable models see the exact same generated minibatch at each step. All
models are evaluated on the exact same examples for a given seed and mode.

The generator adds unreachable camouflage edges so that every entity appears
exactly once as an edge destination before nominal noise is added. This is a
necessary shortcut control: with only a 2–3 edge chain, choosing a displayed
destination scores 33–50% despite doing no composition.

## Low-compute curriculum comparison

To locate the learnability boundary without running the full model suite, train
only GRU and Hybrid together. Both consume the same batch at every step; GRU
validation controls the shared stage transitions.

```powershell
python -m gru_linear_small_model.curriculum_compare
```

The compact seed-0 run used 24-dimensional states, batch size 32, and stopped
after 550 shared steps. Both models mastered one-hop and ordered two-hop chains.
At partially shuffled two-to-three-hop chains, GRU reached 75.00% and Hybrid
75.39%, below the 85% advancement gate. Corrected-development performance
remained at chance (GRU 4.10%, Hybrid 5.66%), so the run stopped before the
expensive stages. The reserved corrected test seed was not evaluated.

See `results/curriculum_compare_compact/diagnosis.md` and
`results/curriculum_compare_compact/comparison_report.json`.

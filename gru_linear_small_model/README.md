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

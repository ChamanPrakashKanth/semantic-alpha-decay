# Gradient-Pattern Recursive Self-Learning (RSL)

Trains a secondary controller to observe training-time backpropagation corrections (sensitivity $s_{ij} = -\langle \nabla_{\widetilde{A}} L, A \rangle$ and counterfactual loss interventions $\Delta L_{ij}$) and predict information retention/decay at inference time without access to labels or gradients.

```bash
python gradient_rsl.py --seeds 5 --steps 800 --output results/gradient_rsl_report.json
```

Evaluates 9 models and controls across 6 distribution shifts and $k \in \{1, 2, 3, 5\}$ recursive reasoning passes.

See `results/gradient_rsl_diagnosis.md` for the five-seed conclusion and `results/gradient_rsl_report.json` for every raw measurement.

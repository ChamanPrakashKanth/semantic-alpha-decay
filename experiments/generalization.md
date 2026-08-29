# Semantic generalization diagnosis

Train exactly four unchanged-size models on the same batches: baseline, fixed
decay, current learned alpha, and the constrained controller
`softplus(w^T |q_i-k_j| + b)`.

```bash
python diagnose_generalization.py --seeds 5 --steps 800 --workers 5
```

The five cases isolate held-out value combinations, unseen layout, added
distractors, reversed key/value order, and all shifts combined. The report logs
accuracy, cross-entropy, `D_rel`, `D_irr`, and `delta_D` for every case and seed.

See `results/generalization_diagnosis.md` for the five-seed conclusion and
`results/generalization_report.json` for every raw measurement.

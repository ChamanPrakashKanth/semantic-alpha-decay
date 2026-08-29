# Balanced Transition-Focused Reinforcement Learning (Transition RL)

Evaluates whether balanced transition rewards ($+1$ for $W \to R$ and $R \to R$, $-1$ for $R \to W$ and $W \to W$) with an explicit dual-head intervention gate $g \in [0, 1]$ and decay $\alpha \ge 0$ can learn an error-rescue policy on an angular task manifold and transfer to a never-rewarded held-out sector.

```bash
python balanced_rsl.py --seeds 10 --steps 800 --output results/balanced_rsl_report.json --csv-output results/balanced_rsl_seed_table.csv
```

See `results/balanced_rsl_diagnosis.md` for the 10-seed conclusion, accounting identity analysis, and `results/balanced_rsl_seed_table.csv` for per-seed raw measurements.

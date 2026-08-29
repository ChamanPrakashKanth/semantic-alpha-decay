# Phase 2: Controlled Base Competence & Autonomous Recursion

Evaluates RSL across 6 controlled base competence levels $p \in \{0.50, 0.60, 0.70, 0.80, 0.90, 0.95\}$ and tests whether learned internal dynamics produce useful autonomous multi-pass recursion once the RL teacher is removed.

```bash
# Run 60-run competence sweep
python controlled_competence_rsl.py --seeds 10 --rsl-steps 400

# Run autonomous teacher-free multi-pass recursion test
python autonomous_recursive_test.py --seeds 10 --k-passes 5 --competence-p 0.70
```

See `results/controlled_competence_diagnosis.md` for complete analysis and `results/controlled_competence_seed_table.csv` and `results/autonomous_recursion_table.csv` for raw data.

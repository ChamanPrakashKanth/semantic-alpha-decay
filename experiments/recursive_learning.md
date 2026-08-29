# One-step recursive output learning

The same unchanged model predicts once, appends its detached answer token to the
sequence, repeats the original query, and predicts again. Both passes are
supervised against the real target. This tests whether the model can learn the
pattern of its own answers and correct them without trusting erroneous
pseudo-labels or adding parameters.

```bash
python recursive_learning.py --seeds 5 --steps 800 --workers 5
```

See `results/recursive_diagnosis.md` for the conclusion and
`results/recursive_report.json` for all seed-level measurements.

# Where semantic generalization breaks

Five seeds, 800 steps, unchanged Transformer size and optimizer. `D_irr` is the
survival of the competing value (not an average over punctuation/noise), making
`delta_D = D_rel - D_irr` a direct semantic selectivity test.

## Accuracy (mean +/- standard deviation)

| Model | IID | Unseen combinations | Unseen layout | More distractors | Reversed order | Combined |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | .652 +/- .050 | .099 +/- .010 | .589 +/- .052 | .587 +/- .024 | .570 +/- .022 | .204 +/- .047 |
| Fixed decay | .638 +/- .043 | .100 +/- .018 | .577 +/- .027 | .563 +/- .029 | .566 +/- .031 | .230 +/- .036 |
| Learned alpha | .663 +/- .104 | .117 +/- .056 | .638 +/- .098 | .610 +/- .024 | .562 +/- .032 | .228 +/- .027 |
| Constrained alpha | .654 +/- .037 | .102 +/- .017 | .618 +/- .012 | .595 +/- .022 | .565 +/- .029 | .206 +/- .063 |

## Learned-alpha survival (mean +/- standard deviation)

| Case | D_rel | D_irr | delta_D | Seeds with delta_D > 0 |
|---|---:|---:|---:|---:|
| IID | .6577 +/- .0483 | .6618 +/- .0445 | -.0041 +/- .0108 | 2/5 |
| Unseen combinations | .6491 +/- .0344 | .6484 +/- .0465 | +.0007 +/- .0170 | 2/5 |
| Unseen layout | .6366 +/- .0473 | .6370 +/- .0435 | -.0004 +/- .0059 | 2/5 |
| More distractors | .6113 +/- .0325 | .6044 +/- .0289 | +.0069 +/- .0076 | 4/5 |
| Reversed order | .5790 +/- .0827 | .5871 +/- .0736 | -.0082 +/- .0126 | 1/5 |
| Combined | .5904 +/- .0413 | .6021 +/- .0369 | -.0117 +/- .0093 | 1/5 |

## Diagnosis

1. **Prediction generalization breaks first on unseen value combinations.** All
   four models fall from roughly 64-66% IID accuracy to roughly 10-12%, near
   chance for the eight possible values. Layout, distractors, and reversal alone
   retain roughly 56-64% accuracy. This identifies compositional value binding,
   not sequence layout, as the primary prediction failure.
2. **The current alpha controller is not learning transferable semantic
   selectivity.** `D_rel > D_irr` is not reliable even IID (2/5 seeds), and mean
   `delta_D` is slightly negative. Reversed order and the combined shift make it
   more negative, showing sensitivity to presentation direction rather than a
   stable relevant-value rule.
3. **This is an alpha-controller failure, not merely a downstream Transformer
   failure.** If the gate generalized while prediction failed, `delta_D` would
   stay positive on unseen combinations. It does not: the mean is nearly zero
   with only 2/5 positive seeds.
4. **The low-capacity controller avoids a strong wrong preference but becomes
   almost inactive.** Its IID `D_rel` and `D_irr` are both about .982 and
   `delta_D` remains approximately zero. It matches baseline accuracy without
   learning semantic selection.

The immediate milestone is therefore unmet: semantic selectivity does not yet
generalize. Scaling would currently amplify a controller that is largely
non-selective and becomes directionally wrong under reversal.

# Low-compute curriculum comparison

GRU and Hybrid were trained together on the exact same minibatch at every step. GRU validation alone controlled stage advancement, so the Hybrid could not alter the shared schedule.

Chance accuracy: **4.17%**.
Total shared training steps: **550**.

| Stage | Steps | GRU stage val | Hybrid stage val | GRU corrected dev | Hybrid corrected dev | Mean lambda | Advanced |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 150 | 100.00% | 97.27% | 4.10% | 5.08% | 0.547 | yes |
| 1 | 50 | 100.00% | 100.00% | 4.10% | 4.10% | 0.575 | yes |
| 2 | 350 | 75.00% | 75.39% | 4.10% | 5.66% | 0.506 | no |

## Verdict

GRU reached its learnability boundary at stage 2 (`two_three_partial_shuffle`). The comparison stopped there to conserve compute.
At the stopping point, corrected-development accuracy was 4.10% for GRU and 5.66% for Hybrid.

The reserved corrected test seed was never generated or evaluated. This remains a development-stage learnability result, not a final architecture claim.

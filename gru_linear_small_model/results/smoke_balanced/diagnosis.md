# Experiment diagnosis

Protocol: **smoke** with 1 seed(s).
Class chance is 4.167%.

- iid: Hybrid 3.906%, GRU 4.492%, parameter-matched GRU 3.711%, probability ensemble 3.906%.
- noise_ood: Hybrid 4.492%, GRU 2.734%, parameter-matched GRU 3.711%, probability ensemble 3.516%.
- length_ood: Hybrid 5.273%, GRU 5.664%, parameter-matched GRU 3.711%, probability ensemble 5.273%.
- combined_ood: Hybrid 4.102%, GRU 4.688%, parameter-matched GRU 2.930%, probability ensemble 4.492%.

## Answers

Learning signal beyond a conservative chance band: **no**.
Largest listed shortcut accuracy: **7.422%**.

1. Hybrid beat GRU on mean length OOD: **no** (-0.391%).
2. Hybrid beat parameter-matched GRU there: **yes** (+1.562%).
3. LinearSmooth and all ablations are reported in `aggregate_report.json`.
4. Whether Hybrid matched the external ensemble is shown mode by mode above.
5. Gains, if any, must be interpreted by evaluation mode rather than IID alone.
6. The chain-length CSV records extrapolation through length 10.
7. Fusion collapse is explicitly flagged in each mode's raw diagnostics.
8. Complementarity counts and conditional rescue rates are preserved per seed.
9. Oracle accuracy gives the available routing headroom.
10. The strongest permitted claim is limited to this synthetic bounded-state benchmark.
11. These results do not establish general reasoning, intelligence, or a Transformer replacement.

This run is exploratory and cannot satisfy the cookbook's serious success criteria.

Because no trainable model cleared the conservative chance band, relative rankings in this run are sampling noise and must not be interpreted as architectural gains.

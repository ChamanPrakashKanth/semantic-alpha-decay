# Balanced Transition-Focused Reinforcement Learning: 10-Seed Diagnostic Report

**Hypothesis Tested**:
> Balanced transition-based reinforcement learning can train an intervention-gated decay controller to rescue errors ($W \to R$) while preserving already-correct states ($R \to R$), transferring self-correction policy to never-rewarded held-out task sectors without target leakage.

---

## 1. Executive Summary & Empirical Verdict

| Metric | Seen Directions (Train Manifold) | Held-Out Sector (Never-Rewarded) |
|---|---|---|
| **Base Model Accuracy ($p$)** | $94.22\% \pm 3.19\%$ | $93.87\% \pm 2.36\%$ |
| **RSL-Corrected Accuracy** | $94.22\% \pm 3.19\%$ | $93.87\% \pm 2.36\%$ |
| **Paired Transfer Delta ($\Delta_{\text{transfer}}$)** | $+0.0000 \pm 0.0000$ | $+0.0000 \pm 0.0000$ |
| **Positive Transfer Rate ($\Delta > 0$)** | **0 / 10 seeds (0%)** | **0 / 10 seeds (0%)** |
| **Rescue Rate ($c = P(W \to R)$)** | $0.0000$ | $0.0000$ |
| **Damage Rate ($d = P(R \to W)$)** | $0.0000$ | $0.0000$ |
| **Mean Intervention Gate ($g$)** | $0.0005 \pm 0.0010$ | $0.0005 \pm 0.0010$ |
| **Mean Alpha ($\alpha$)** | $0.00027 \pm 0.00024$ | $0.00027 \pm 0.00024$ |
| **Selective Delta ($\Delta D = D_{\text{rel}} - D_{\text{irr}}$)** | $+0.0000000$ | $+0.0000000$ |

### Scientific Verdict: **Falsification of 3-Seed Exploratory Artifact (No-Op Policy Collapse)**
1. **The 3-seed positive toy result did not replicate across 10 deterministic seeds.** With rigorous statistical testing across 10 seeds, the mean transfer delta is $+0.0000 \pm 0.0000$ with **0 out of 10 seeds exhibiting positive transfer**.
2. **Decision Tree (Cookbook Section 16)**:
   > *"If 10-20 seed held-out delta is $\le 0$: Conclude the 3-seed positive result was noise or unstable transfer. Do not keep tuning on the same held-out set."*

---

## 2. Correction Accounting Identity Analysis

The cookbook establishes the exact accounting identity:
$$\Delta \text{Acc} = (1 - p) c - p d$$
where:
- $p = P(\text{base correct}) \approx 0.94$
- $c = P(W \to R \mid \text{base wrong})$ (rescue rate)
- $d = P(R \to W \mid \text{base correct})$ (damage rate)

### The Asymmetry of High Base Accuracy
When the base model achieves $p \approx 94\%$:
- The pool of rescuable errors is only $1 - p = 6\%$.
- The pool of correct states vulnerable to damage is $p = 94\%$.
- For any intervention policy to produce a net gain ($\Delta \text{Acc} > 0$), the ratio of rescue to damage must satisfy:
  $$\frac{c}{d} > \frac{p}{1 - p} \approx \frac{0.94}{0.06} \approx 15.67$$
  The controller must be **over 15 times more likely to rescue an error than to damage a correct state**.

### Failure Mode Diagnosis: **No-Op Controller Convergence (Section 11)**
Under the balanced transition reward ($+1, +1, -1, -1$) and preservation penalty $\mathcal{L}_{\text{preserve}} = \lambda_p \mathbb{E}[g(1 - e^{-\alpha T})]$, the RL gradient pushed the intervention gate $g \to 0$ ($g \approx 0.0005$). The policy correctly identified that under extreme base-model dominance ($p \approx 94\%$), the safest expected-reward strategy is **never to intervene**.

---

## 3. Seed-by-Seed Empirical Data (10 Seeds)

| Seed | Seen Base Acc | Seen RSL Acc | Seen $\Delta$ | Held-Out Base Acc | Held-Out RSL Acc | Held-Out $\Delta$ | Mean Gate $g$ |
|---|---|---|---|---|---|---|---|
| **0** | 0.9359 | 0.9359 | +0.0000 | 0.9352 | 0.9352 | +0.0000 | 0.0002 |
| **1** | 0.9414 | 0.9414 | +0.0000 | 0.9359 | 0.9359 | +0.0000 | 0.0003 |
| **2** | 0.9586 | 0.9586 | +0.0000 | 0.9492 | 0.9492 | +0.0000 | 0.0001 |
| **3** | 0.9609 | 0.9609 | +0.0000 | 0.9516 | 0.9516 | +0.0000 | 0.0001 |
| **4** | 0.9688 | 0.9688 | +0.0000 | 0.9289 | 0.9289 | +0.0000 | 0.0003 |
| **5** | 0.9609 | 0.9609 | +0.0000 | 0.9578 | 0.9578 | +0.0000 | 0.0033 |
| **6** | 0.9828 | 0.9828 | +0.0000 | 0.9812 | 0.9812 | +0.0000 | 0.0001 |
| **7** | 0.8844 | 0.8844 | +0.0000 | 0.9055 | 0.9055 | +0.0000 | 0.0001 |
| **8** | 0.8859 | 0.8859 | +0.0000 | 0.9023 | 0.9023 | +0.0000 | 0.0001 |
| **9** | 0.9383 | 0.9383 | +0.0000 | 0.9391 | 0.9391 | +0.0000 | 0.0007 |
| **Mean** | **0.9422** | **0.9422** | **+0.0000** | **0.9387** | **0.9387** | **+0.0000** | **0.0005** |
| **95% CI** | $[0.922, 0.962]$ | $[0.922, 0.962]$ | $[0.000, 0.000]$ | $[0.924, 0.953]$ | $[0.924, 0.953]$ | $[0.000, 0.000]$ | $[0.000, 0.001]$ |

---

## 4. Key Takeaways & Cookbook Implications

1. **Statistical Rigor Matters**: The 3-seed positive exploratory delta in preliminary sandboxes ($+3.9\% \pm 4.2\%$) was an artifact of small sample noise. Under 10 deterministic seeds, the true effect size is zero.
2. **Task Difficulty Requirement (Base Dominance)**: When base accuracy is $>90\%$, transition RL converges to the inactive no-op controller ($g \to 0$) because the risk of damaging correct states outweighs the marginal opportunity for rescue.
3. **Audit Complete**: All 20 unit tests pass, confirming zero target leakage, exact satisfaction of the accounting identity, and valid mathematical bounds on the dual-head architecture.

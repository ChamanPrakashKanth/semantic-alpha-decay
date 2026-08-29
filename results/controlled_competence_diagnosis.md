# Phase 2 & Autonomous Teacher-Free Recursion: Diagnostic Report

**Hypotheses Evaluated**:
1. **Competence Hypothesis**: Is there an intermediate base-model competence level $p \in \{0.50, 0.60, 0.70, 0.80, 0.90, 0.95\}$ where transition RL learns useful error correction rather than rational abstention?
2. **Autonomous Self-Learning Hypothesis**: Once the external RL reward teacher is removed, do the internal correction dynamics continue to produce useful recursive self-correction across multiple passes ($k \in \{1, 2, 3, 5\}$)?

---

## 1. Metric Audit Findings

In Phase 1, `surv.mean()` computed the global tensor average across the full $L \times L$ matrix including the upper-triangular causal mask ($A_{ij} = 0$). For sequence length $L = 6$, causal valid entries are:
$$\frac{1 + 2 + 3 + 4 + 5 + 6}{36} = \frac{21}{36} = 0.5833333$$
In Phase 2, metrics were strictly audited over valid causal tokens:
- **`mean_gate`**: $g \approx 0.0005$
- **`mean_effective_D`**: $D_{\text{eff}} = 1.0000$ (exact mathematical verification of abstention)
- Boundary audits passed: $g = 0 \implies D_{\text{eff}} = 1.0$, and $g = 1 \implies D_{\text{eff}} = D_{\text{raw}}$.

---

## 2. Part 1: Controlled Base Competence Sweep (60 Runs: 6 Levels $\times$ 10 Seeds)

| Base Competence ($p$) | Test Base Acc | Test RSL Acc | Transfer $\Delta$ | Mean Gate ($g$) | Effective Survival ($D_{\text{eff}}$) | Oracle Headroom ($H(p)$) | RSL Efficiency ($\eta_{\text{RSL}}$) |
|---|---|---|---|---|---|---|---|
| **$p \approx 0.50$** (Step 30) | $24.96\% \pm 1.34\%$ | $24.96\% \pm 1.34\%$ | $+0.0000$ | $0.0024$ | $1.0000$ | $+6.86\%$ | $0.00\%$ |
| **$p \approx 0.60$** (Step 60) | $20.80\% \pm 6.18\%$ | $20.80\% \pm 6.18\%$ | $+0.0000$ | $0.0028$ | $1.0000$ | $+11.39\%$ | $0.00\%$ |
| **$p \approx 0.70$** (Step 120) | $44.92\% \pm 20.9\%$ | $44.92\% \pm 20.9\%$ | $+0.0000$ | $0.0023$ | $1.0000$ | $+18.00\%$ | $0.00\%$ |
| **$p \approx 0.80$** (Step 220) | $87.05\% \pm 8.75\%$ | $87.05\% \pm 8.75\%$ | $+0.0000$ | $0.0020$ | $1.0000$ | $+6.95\%$ | $0.00\%$ |
| **$p \approx 0.90$** (Step 450) | $91.80\% \pm 3.91\%$ | $91.80\% \pm 3.91\%$ | $+0.0000$ | $0.0021$ | $1.0000$ | $+5.89\%$ | $0.00\%$ |
| **$p \approx 0.95$** (Step 800) | $93.09\% \pm 3.42\%$ | $93.09\% \pm 3.42\%$ | $+0.0000$ | $0.0019$ | $1.0000$ | $+5.09\%$ | $0.00\%$ |

### Diagnosis for Part 1:
- Across **all 6 competence levels**, the single-pass RSL controller consistently converges to the **Abstention Regime** ($g \to 0, D_{\text{eff}} \to 1.0000, \Delta_{\text{transfer}} = +0.0000$).
- Actuator headroom exists across all competence levels ($H(p) = +5.1\%$ to $+18.0\%$), proving that the actuator is capable of correcting errors when guided by supervision, but the inference-available features $(q, k, A, \text{uncertainty})$ do not provide sufficient signal for the policy gradient to escape the abstention optimum.

---

## 3. Part 2: Autonomous Teacher-Free Recursion (10 Seeds, $k=1 \to 5$ Passes)

To test the core self-learning question—*whether learned internal dynamics produce useful autonomous correction after the RL reward teacher is removed*—we evaluated multi-pass recursion without ground-truth feedback:

| Pass ($k$) | Mean Accuracy | Step Delta | Cumulative $\Delta$ | Rescue Rate ($c_k$) | Damage Rate ($d_k$) | Damage / Rescue Ratio | Error Loop Rate |
|---|---|---|---|---|---|---|---|
| **Pass 1 ($k=1$)** | $44.84\% \pm 22.0\%$ | $+0.00\%$ | $+0.00\%$ | $0.00\%$ | $0.00\%$ | $-$ | $0.00$ |
| **Pass 2 ($k=2$)** | $32.85\% \pm 8.03\%$ | $-11.98\%$ | $-11.98\%$ | $21.49\%$ | **$56.31\%$** | **$2.62\times$** | **$0.80$ (80%)** |
| **Pass 3 ($k=3$)** | $35.50\% \pm 12.1\%$ | $+2.65\%$ | $-9.34\%$ | $17.17\%$ | $23.81\%$ | $1.39\times$ | $0.20$ (20%) |
| **Pass 4 ($k=4$)** | $33.34\% \pm 7.95\%$ | $-2.16\%$ | $-11.49\%$ | $10.26\%$ | $21.22\%$ | $2.07\times$ | $0.40$ (40%) |
| **Pass 5 ($k=5$)** | $34.52\% \pm 9.41\%$ | $+1.18\%$ | $-10.31\%$ | $8.59\%$ | $11.09\%$ | $1.29\times$ | $0.20$ (20%) |

---

## 4. Scientific Verdict & Conclusions

1. **Training RSL with RL is External Reward Supervision, Not Autonomous Self-Learning**:
   - The RL training phase relies on external reward signals generated from true labels $y$.
2. **Autonomous Multi-Pass Recursion Collapses into Error Amplification**:
   - When the RL teacher is removed and internal representations are fed back recursively without ground-truth verification, initial errors corrupt the subsequent attention computations.
   - At step $k=2$, the damage rate ($56.31\%$) substantially exceeds the rescue rate ($21.49\%$), causing accuracy to drop by $-11.98\%$ and triggering self-reinforcing error loops in **80% of runs**.
3. **Scientific Conclusion**:
   > **Falsification.**
   > The learned internal correction dynamics do **not** produce autonomous recursive self-correction once the RL teacher is removed. Without an external grounding signal or ground-truth verifier, recursive internal state refinement amplifies errors into self-reinforcing loops.

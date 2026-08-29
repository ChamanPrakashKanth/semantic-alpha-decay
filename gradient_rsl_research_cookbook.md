# Gradient-Pattern RSL Research Cookbook

## Purpose

This file is an execution guide for AI coding/research agents continuing
the `semantic-alpha-decay` project.

The project began as Semantic Alpha Decay Transformer (SADT), using
post-softmax attention survival

\[ A\_{ij} = `\operatorname{softmax}`{=tex}*j(S*{ij}), `\qquad`{=tex}
D\_{ij}=e\^{-`\alpha`{=tex}*{ij}T}, `\qquad`{=tex}
`\widetilde `{=tex}A*{ij}=A\_{ij}D\_{ij}. \]

Current evidence does **not** support the claim that the learned alpha
controller is a transferable semantic reasoning engine.

The strongest current direction is instead:

> Train a controller to learn when an internal intervention is likely to
> correct a wrong state while preserving an already-correct state, then
> test whether that correction policy transfers to tasks that received
> no RL reward.

Do not optimize experiments to make this hypothesis look successful. Try
to falsify it.

------------------------------------------------------------------------

## 1. Established Results: Do Not Re-Litigate Without a Reason

### 1.1 The decay actuator can be useful

An Oracle controller with privileged correction information improves the
base model substantially on several benchmark splits.

Interpretation:

\[ `\text{correct control information}`{=tex} + e\^{-`\alpha `{=tex}T}
`\rightarrow`{=tex} `\text{useful intervention}`{=tex}. \]

Therefore the actuator is not the main unresolved problem.

### 1.2 Learned semantic alpha did not generalize

The learned SADT controller showed task-dependent selectivity but failed
to learn a robust transferable semantic relevance rule.

Use the wording:

> The learned decay controller acquires task-dependent
> information-selection behavior, but OOD evidence indicates that this
> behavior is dominated by distribution-specific shortcuts rather than
> transferable semantic relevance.

### 1.3 Blind recursion failed

Appending/reusing the model's own previous answer did not create
self-correction.

Observed pathology:

\[ `\text{previous answer}`{=tex} `\rightarrow`{=tex}
`\text{persistence prior}`{=tex}. \]

Useful recursion requires information gain or a learned corrective
intervention.

### 1.4 Gradient-imitation RSL failed OOD

A controller could fit training-time gradient-derived correction targets
with very low error, but OOD selectivity collapsed:

\[ `\Delta `{=tex}D\_{`\mathrm{OOD}`{=tex}}`\approx 0`{=tex}. \]

A shuffled-target controller behaved similarly.

Interpretation:

\[ `\text{memorizing correction patterns}`{=tex} `\neq`{=tex}
`\text{learning a transferable correction rule}`{=tex}. \]

### 1.5 Ungrounded recursive RSL amplified errors

Previously observed:

\[ P(`\text{right}`{=tex}`\rightarrow`{=tex}`\text{wrong}`{=tex}) \>
P(`\text{wrong}`{=tex}`\rightarrow`{=tex}`\text{right}`{=tex}). \]

This is a positive-feedback failure mode.

------------------------------------------------------------------------

## 2. Current Hypothesis

The new hypothesis is **not** that the controller can reconstruct the
true supervised gradient without the target.

Instead:

\[ `\boxed{
\text{RL experience across diverse tasks}
\rightarrow
\text{learned intervention policy}
\rightarrow
\text{partial correction transfer to unseen tasks}
}`{=tex} \]

The controller should learn two behaviors simultaneously:

1.  **Intervene when intervention is likely to rescue an error.**
2.  **Do not intervene aggressively when the current state is already
    good.**

This is crucial. Earlier transition-RL became too aggressive and learned
near-total suppression.

------------------------------------------------------------------------

## 3. Important Toy Results So Far

These are exploratory sandbox results, not publication-grade evidence.

### 3.1 Ordinary RL transfer

RL trained on several task rules and tested on a held-out rule.

Held-out result:

-   RSL off: about 55.9%
-   RSL on: about 54.4%
-   transfer: about -1.6%
-   wrong-\>right: about 4.7%
-   right-\>wrong: about 6.2%

Conclusion: ordinary final-answer RL did not produce transferable
correction.

### 3.2 Transition-focused RL

Reward explicitly targeted correction transitions.

Example reward:

\[ R(W`\rightarrow `{=tex}R)=+1.5 \]

\[ R(R`\rightarrow `{=tex}W)=-1.5. \]

On trained task families, correction became strong. On the held-out
task, net transfer remained approximately zero.

Conclusion: RL can teach self-correction on familiar task structure, but
this alone does not establish transfer.

### 3.3 Many-task transition RL

Train on many random linear rules

\[ q=(a,b) \]

while withholding an angular sector of rule space.

The aggressive reward version produced approximately:

-   held-out (P(W`\rightarrow `{=tex}R)): 59%
-   held-out (P(R`\rightarrow `{=tex}W)): 6.9%
-   net accuracy change: approximately zero
-   mean survival (D): approximately 0.005

Diagnosis:

> The controller learned to intervene far too aggressively. It rescued
> many errors but damaged enough already-correct states to erase the
> gain.

### 3.4 Balanced transition RL

The next version balanced preservation and correction:

\[ R(W`\rightarrow `{=tex}R)=+1 \]

\[ R(R`\rightarrow `{=tex}R)=+1 \]

\[ R(R`\rightarrow `{=tex}W)=-1 \]

\[ R(W`\rightarrow `{=tex}W)=-1. \]

A light unnecessary-decay penalty was also added:

\[ L\_{`\mathrm{preserve}`{=tex}} = `\mathbb `{=tex}E\[1-D\]. \]

Exploratory 3-seed result:

#### Seen directions

-   RSL off: (0.893`\pm0.054`{=tex})
-   RSL on: (0.929`\pm0.021`{=tex})
-   delta: (+0.036`\pm0.047`{=tex})
-   (P(W`\rightarrow `{=tex}R)=0.622)
-   (P(R`\rightarrow `{=tex}W)=0.042)
-   mean (D=0.408)

#### Never-rewarded held-out sector

-   RSL off: (0.854`\pm0.049`{=tex})
-   RSL on: (0.893`\pm0.027`{=tex})
-   delta: (+0.039`\pm0.042`{=tex})
-   (P(W`\rightarrow `{=tex}R)=0.387)
-   (P(R`\rightarrow `{=tex}W)=0.037)
-   mean (D=0.621)

This is encouraging but **not statistically sufficient**. Three seeds
and a noisy (+3.9%`\pm4.2`{=tex}%) effect do not justify a transfer
claim.

------------------------------------------------------------------------

# 4. What the Next Agent Must Do

## Step A - Reproduce the balanced result properly

Do not immediately redesign the architecture.

First reproduce the balanced transition-RL experiment with:

-   at least 10 deterministic seeds;
-   preferably 20 seeds if runtime permits;
-   identical train/held-out task geometry;
-   fixed evaluation sets per seed;
-   saved raw per-seed metrics;
-   confidence intervals;
-   paired comparison between RSL-on and RSL-off.

Report:

\[ `\Delta`{=tex}*{`\mathrm{transfer}`{=tex}} =
Acc*{`\mathrm{RSL-on}`{=tex}} - Acc\_{`\mathrm{RSL-off}`{=tex}}. \]

Also report the fraction of seeds for which:

\[ `\Delta`{=tex}\_{`\mathrm{transfer}`{=tex}}\>0. \]

Do not report only the mean.

------------------------------------------------------------------------

## Step B - Measure the correction accounting identity

For every evaluation split compute:

\[ p = P(`\text{base correct}`{=tex}) \]

\[ c=P(W`\rightarrow `{=tex}R) \]

\[ d=P(R`\rightarrow `{=tex}W). \]

The expected net accuracy change should approximately satisfy:

\[ `\boxed{
\Delta Acc
=
(1-p)c-pd
}`{=tex} \]

up to sampling/definition details.

Use this identity to diagnose every result.

A controller can have a large (P(W`\rightarrow `{=tex}R)) and still hurt
accuracy when (p) is large and (P(R`\rightarrow `{=tex}W)) is not
sufficiently small.

------------------------------------------------------------------------

## Step C - Test reward balance systematically

Do not choose reward coefficients based on which gives the highest
held-out score.

Predefine a sweep such as:

    W-\>R   R-\>R   R-\>W   W-\>W
  ------- ------- ------- -------
       +1      +1      -1      -1
       +1      +1      -2      -1
       +2      +1      -2      -1
       +1      +2      -2      -1

Also sweep the preservation penalty:

\[ `\lambda`{=tex}\_p `\in`{=tex} {0, 0.01, 0.05, 0.1}. \]

Choose hyperparameters using **training/validation task families only**.

Never choose them based on the final held-out sector.

------------------------------------------------------------------------

## Step D - Add an explicit intervention gate

Current alpha can modify every state.

Test a two-part controller:

\[
g=`\sigma`{=tex}(f_g(h,`\text{confidence}`{=tex},`\text{context}`{=tex}))
\]

\[
`\alpha`{=tex}=`\operatorname{positive}`{=tex}(f\_`\alpha`{=tex}(`\cdot`{=tex}))
\]

and

\[ D = (1-g)+g e\^{-`\alpha `{=tex}T}. \]

Interpretation:

-   (g`\approx0`{=tex}): leave attention alone;
-   (g`\approx1`{=tex}): apply learned decay.

This separates:

1.  **Should I intervene?**
2.  **How strongly should I intervene?**

This may be more appropriate than forcing alpha itself to solve both
decisions.

Compare against the one-head alpha controller.

------------------------------------------------------------------------

## Step E - Use confidence and uncertainty carefully

Inference-available features may include:

-   maximum predicted probability;
-   top-1/top-2 margin;
-   entropy;
-   attention entropy;
-   layer/head disagreement;
-   change in prediction between recursive passes;
-   representation change;
-   alpha history;
-   survival history.

No ground-truth target, true loss, or supervised gradient may enter the
controller at held-out evaluation.

Add tests that fail if target leakage occurs.

------------------------------------------------------------------------

# 5. Harder Transfer Tests

The angular-sector toy task is only the first test.

If the balanced result survives, progressively increase difficulty.

## Level 1 - Interpolation gap

Train on most (q=(a,b)) directions and hold out one contiguous angular
sector.

This is the current experiment.

## Level 2 - Larger gap

Increase the held-out sector width.

Example:

\[
60^`\circ `{=tex}`\rightarrow 90`{=tex}^`\circ `{=tex}`\rightarrow 120`{=tex}\^`\circ`{=tex}.
\]

Measure how transfer decays with distance from the training
distribution.

## Level 3 - Different coefficient magnitudes

Train primarily on unit-norm (q), then test:

\[ q=(2,-1),`\quad `{=tex}(3,-2),`\quad `{=tex}(0.5,-1.5). \]

Normalize and non-normalize in separate experiments.

## Level 4 - Nonlinear held-out rule

Train only on linear rules:

\[ q\^`\top `{=tex}x. \]

Then test a never-trained nonlinear family, for example:

\[ x_1x_2, \]

or another clearly specified synthetic relation.

Expect transfer to be much harder.

## Level 5 - Different task family

Only after the above tests work should the agent test transfer from
selective-recall/value-binding training to a structurally different
synthetic task.

A gain here would be much stronger evidence.

------------------------------------------------------------------------

# 6. Recursive RSL Test

Do not assume more passes are better.

Evaluate:

\[ k`\in`{=tex}{1,2,3,5}. \]

For each pass record:

\[ Acc_k \]

\[ P_k(W`\rightarrow `{=tex}R) \]

\[ P_k(R`\rightarrow `{=tex}W) \]

\[ D_k \]

and selectivity/intervention statistics.

The desired behavior is not merely increasing activity.

Ideally:

\[ Acc\_{k+1}`\ge `{=tex}Acc_k \]

and

\[ P(W`\rightarrow `{=tex}R) \> P(R`\rightarrow `{=tex}W) \]

without collapse of survival toward zero.

Flag a self-reinforcing error loop when later passes increasingly damage
previously correct examples.

------------------------------------------------------------------------

# 7. Required Baselines

Every serious experiment should include:

1.  Transformer / base model with RSL disabled.
2.  Fixed decay.
3.  Random decay controller.
4.  Shuffled-policy or shuffled-reward control.
5.  Learned SADT controller.
6.  Gradient-pattern imitation RSL.
7.  Ordinary final-answer RL RSL.
8.  Transition-reward RSL.
9.  Balanced transition-reward RSL.
10. Intervention-gated RSL, if implemented.
11. Oracle controller using privileged target/correction information
    **only as an invalid-inference upper bound**.

Do not compare only against a weak baseline.

------------------------------------------------------------------------

# 8. Required Metrics

At minimum save per seed:

-   base accuracy;
-   RSL accuracy;
-   paired accuracy delta;
-   wrong-\>right rate;
-   right-\>wrong rate;
-   right-\>right rate;
-   wrong-\>wrong rate;
-   mean alpha;
-   alpha standard deviation;
-   mean survival (D);
-   survival standard deviation;
-   intervention frequency, if a gate is used;
-   prediction confidence;
-   calibration/error-detection metrics if practical;
-   recursive metrics per pass;
-   training reward;
-   validation reward;
-   held-out reward only for reporting, never tuning.

For selective-recall tasks also retain:

\[ D\_{`\mathrm{relevant}`{=tex}}, `\qquad`{=tex}
D\_{`\mathrm{irrelevant}`{=tex}}, `\qquad`{=tex} `\Delta `{=tex}D =
D\_{`\mathrm{relevant}`{=tex}}-D\_{`\mathrm{irrelevant}`{=tex}}. \]

------------------------------------------------------------------------

# 9. Statistical Discipline

The current positive toy result is exploratory.

For the next run:

-   use \>=10 seeds;
-   report mean, standard deviation, median, and 95% confidence
    interval;
-   use paired seed-level RSL-on vs RSL-off differences;
-   report all seeds;
-   do not discard failed seeds;
-   do not rerun only negative seeds;
-   separate exploratory hyperparameter search from final confirmation;
-   freeze the final configuration before evaluating the final held-out
    set.

A positive result should survive reasonable changes in:

-   seed;
-   task gap;
-   number of slots;
-   noise;
-   model width;
-   reward scale.

------------------------------------------------------------------------

# 10. Leakage Rules

At held-out inference, the RSL controller must **not** access:

-   target (y);
-   correctness indicator;
-   supervised loss;
-   true gradient;
-   counterfactual loss using (y);
-   Oracle relevance mask;
-   reward computed from the true answer.

The evaluation harness may use (y) **after prediction** to score
metrics.

Add explicit unit tests proving that the RSL inference API can execute
without a target argument.

------------------------------------------------------------------------

# 11. Failure Modes to Watch

## Aggressive suppression

Symptom:

\[ D`\rightarrow0`{=tex}. \]

Interpretation: the policy learned "intervene everywhere."

Fix experimentally with preservation costs, intervention gates, or
reward rebalancing. Do not hide the failure.

## No-op controller

Symptom:

\[ D`\rightarrow1`{=tex}. \]

Interpretation: the safest policy became "never intervene."

This can happen when the right-\>wrong penalty is too strong.

## Shortcut learning

High training correction with no held-out correction indicates
task-specific policy memorization.

## Base-model dominance

If the base model already reaches near 100%, net transfer becomes
difficult to measure. Increase task difficulty so there is room for
correction.

## Random perturbation advantage

If random decay beats learned RSL, test controlled perturbation
strength. It may indicate shortcut-breaking regularization rather than
learned correction.

## Recursive instability

If:

\[ P(R`\rightarrow `{=tex}W) \]

grows with pass number, recursion is amplifying controller errors.

------------------------------------------------------------------------

# 12. Architecture Principle

Keep the conceptual components separate.

### Base model

Produces ordinary attention/state/prediction.

### Intervention detector

Estimates whether changing the current computation is warranted.

### Decay strength controller

Chooses the magnitude of suppression.

### Actuator

\[ D=e\^{-`\alpha `{=tex}T}. \]

### Recursive loop

Feeds the modified computation into another pass only when explicitly
being tested.

This decomposition makes failures interpretable.

------------------------------------------------------------------------

# 13. Current Scientific Interpretation

Do **not** claim that RSL has learned a universal notion of right and
wrong.

The strongest defensible statement from the current exploratory evidence
is:

> Balanced transition-based reinforcement learning can train the decay
> controller to rescue errors while preserving correct states on
> familiar synthetic task families. A small exploratory experiment also
> produced a positive mean accuracy delta on a never-rewarded held-out
> region of task space, but the effect is noisy and requires larger
> multi-seed confirmation before it can be interpreted as transferable
> self-correction.

If the larger experiment fails, report the failure.

If it succeeds only on nearby task rules, call it **local task-space
transfer**, not general reasoning.

If it succeeds across structurally different task families, then
investigate a stronger self-correction claim.

------------------------------------------------------------------------

# 14. Recommended Next Implementation Files

Create or extend:

``` text
balanced_rsl.py
tests/test_balanced_rsl.py
results/balanced_rsl_report.json
results/balanced_rsl_diagnosis.md
results/balanced_rsl_seed_table.csv
```

Optional later:

``` text
intervention_gate_rsl.py
tests/test_intervention_gate_rsl.py
results/intervention_gate_report.json
```

Update README only after results are reproduced.

------------------------------------------------------------------------

# 15. Minimal Experimental Pseudocode

``` python
for training_step in range(num_steps):
    task = sample_training_task_excluding_holdout()
    x, y = sample_batch(task)

    base_state, base_logits = transformer(x, task)
    base_action = base_logits.argmax(-1)

    alpha, gate = rsl_controller(
        inference_available_features(base_state, base_logits)
    )

    D = (1 - gate) + gate * torch.exp(-alpha * T)
    corrected_logits = apply_decay(base_state, D)

    corrected_action = sample_policy(corrected_logits)

    reward = transition_reward(
        base_action=base_action,
        corrected_action=corrected_action,
        target=y,
    )

    optimize_joint_policy(reward)
```

At final held-out evaluation:

``` python
with torch.no_grad():
    # y is NOT passed into the model/controller.
    base_prediction = model.predict(x, rsl=False)
    rsl_prediction = model.predict(x, rsl=True)

# Only now use y for scoring.
score(base_prediction, rsl_prediction, y)
```

------------------------------------------------------------------------

# 16. Decision Tree for the Next Agent

### If 10-20 seed held-out delta is \<= 0

Conclude the 3-seed positive result was noise or unstable transfer.

Do not keep tuning on the same held-out set.

### If delta is positive but only near the training boundary

Call it local interpolation/generalization.

Increase the held-out gap.

### If delta remains positive across a wide unseen sector

Add intervention-gate ablation and harder rule families.

### If nonlinear/different-family transfer fails

Conclude the policy transfers within a task manifold but not across task
structure.

### If structurally different task transfer succeeds

Run leakage audits, random/shuffled controls, larger models, and
independent replication before making stronger claims.

------------------------------------------------------------------------

# 17. Core Research Question

The project should now answer:

\[ `\boxed{
\text{Can balanced reinforcement learning make an internal decay controller learn a correction policy that improves never-rewarded tasks without access to the target at inference?}
}`{=tex} \]

Secondary question:

\[ `\boxed{
\text{Can it learn when NOT to intervene?}
}`{=tex} \]

The second question is as important as the first.

------------------------------------------------------------------------

## Final Rule for Agents

**Do not confuse activity with intelligence.**

A controller that changes many attention weights is not necessarily
correcting anything.

The desired result is:

\[ `\boxed{
\text{selective intervention}
+
\text{error rescue}
+
\text{correct-state preservation}
+
\text{held-out transfer}
}`{=tex} \]

All four must be measured separately.

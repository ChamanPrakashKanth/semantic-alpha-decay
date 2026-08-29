# RSL Research Cookbook --- Phase 2: Controlled Base Competence

## Status

This document supersedes the earlier exploratory interpretation of the
3-seed balanced-transition result.

The 10-seed replication falsified that apparent positive transfer
result.

### Current empirical result

With the dual-head intervention-gated RSL architecture:

-   seen base accuracy: \~94.22%
-   held-out base accuracy: \~93.87%
-   RSL accuracy change: exactly 0 across the reported 10 seeds
-   positive-transfer seeds: 0/10
-   rescue rate (P(W`\to `{=tex}R)=0)
-   damage rate (P(R`\to `{=tex}W)=0)
-   mean intervention gate (g`\approx0.0005`{=tex})

The learned controller converged to an almost complete **no-op /
abstention policy**.

The previous exploratory (+3.9%`\pm4.2`{=tex}%) 3-seed result must not
be cited as evidence of transferable self-correction.

------------------------------------------------------------------------

# 1. Current Scientific Interpretation

The correction accounting identity is

\[ `\boxed{
\Delta Acc=(1-p)c-pd
}`{=tex} \]

where

-   (p=P(`\text{base prediction is correct}`{=tex})),
-   (c=P(W`\to `{=tex}R)),
-   (d=P(R`\to `{=tex}W)).

Positive net improvement requires

\[ (1-p)c\>pd \]

or

\[ `\boxed{
\frac{c}{d}>\frac{p}{1-p}.
}`{=tex} \]

For (p=0.94),

\[ `\frac{p}{1-p}`{=tex}`\approx15.67`{=tex}. \]

Therefore, when the base model is already \~94% accurate, RSL must
rescue errors at more than \~15.7 times its correct-state damage rate
merely to break even.

The controller instead learned:

\[ g`\rightarrow0`{=tex}. \]

This is not evidence of self-correction.

It is evidence that under the current reward/task distribution,
**abstention is the safest learned policy**.

------------------------------------------------------------------------

# 2. Immediate Code Audit Before New Experiments

The reported values include:

\[ g`\approx0.0005`{=tex}, `\qquad`{=tex}
`\alpha`{=tex}`\approx0.00027`{=tex}, `\qquad`{=tex}
D`\approx0.5833`{=tex}. \]

If effective survival is implemented as

\[ `\boxed{
D_{\mathrm{eff}}=(1-g)+g e^{-\alpha T},
}`{=tex} \]

then tiny (g) and tiny (`\alpha`{=tex}) should normally produce

\[ D\_{`\mathrm{eff}`{=tex}}`\approx1`{=tex}. \]

Therefore the next agent MUST inspect what the reported
`Mean Net Survival D` actually represents.

Check whether the report is logging:

1.  raw (e\^{-`\alpha `{=tex}T}),
2.  gated effective survival,
3.  attention-weight mean,
4.  a normalized statistic,
5.  another intermediate tensor.

Add separate metrics:

``` text
mean_gate
mean_alpha
mean_raw_survival
mean_effective_survival
```

with

``` python
raw_survival = torch.exp(-alpha * T)
effective_survival = (1.0 - gate) + gate * raw_survival
```

Add a unit test verifying:

``` python
gate = 0
=> effective_survival == 1
```

and:

``` python
gate = 1
=> effective_survival == raw_survival
```

Do not begin the Phase-2 benchmark until this audit passes.

------------------------------------------------------------------------

# 3. New Central Research Question

Do not ask only:

> Can RSL improve a nearly solved task?

Instead ask:

\[ `\boxed{
\text{At what level of base-model competence does RSL learn useful correction rather than rational abstention?}
}`{=tex} \]

The next experiment must separate:

1.  base-model learning,
2.  RSL learning,
3.  transfer.

------------------------------------------------------------------------

# 4. Phase-2 Experimental Design

## Stage A --- Train the base model alone

Train a base Transformer/scorer without RSL.

Create checkpoints with controlled held-out validation accuracies near:

\[ `\boxed{
p\in\{0.50,0.60,0.70,0.80,0.90,0.95\}.
}`{=tex} \]

Exact values do not need to be perfect. Use narrow target bands, e.g. ±2
percentage points.

Possible methods:

-   stop training at different checkpoints;
-   vary model width;
-   vary training-set size;
-   inject controlled label noise during base training;
-   vary training steps.

Prefer checkpointing/stopping over artificial corruption at inference.

Save every base checkpoint.

------------------------------------------------------------------------

## Stage B --- Freeze the base model

This is mandatory for the primary experiment.

After choosing a base checkpoint:

``` python
for p in base_model.parameters():
    p.requires_grad = False
```

RSL training must not improve the base model itself.

This isolates:

\[ `\boxed{
\text{Can RSL learn to correct a fixed model?}
}`{=tex} \]

------------------------------------------------------------------------

## Stage C --- Train only RSL

Use the dual-head controller:

### Intervention gate

\[ g=`\sigma`{=tex}(f_g(z)) \]

### Decay magnitude

\[ `\alpha`{=tex}=`\operatorname{positive}`{=tex}(f\_`\alpha`{=tex}(z))
\]

### Raw survival

\[ D\_{`\mathrm{raw}`{=tex}}=e\^{-`\alpha `{=tex}T} \]

### Effective survival

\[ `\boxed{
D_{\mathrm{eff}}
=
(1-g)+gD_{\mathrm{raw}}
}`{=tex} \]

Interpretation:

-   (g`\approx0`{=tex}): abstain;
-   (g`\approx1`{=tex}): intervene;
-   (`\alpha`{=tex}): intervention strength.

RSL may use only inference-visible features.

------------------------------------------------------------------------

# 5. Transition Reward

Start with the balanced objective:

\[ R(W`\to `{=tex}R)=+1 \]

\[ R(R`\to `{=tex}R)=+1 \]

\[ R(R`\to `{=tex}W)=-1 \]

\[ R(W`\to `{=tex}W)=-1. \]

Do not tune reward coefficients using the final held-out test set.

A small intervention cost may be used:

\[ L\_{`\mathrm{intervene}`{=tex}} = `\lambda`{=tex}\_g E\[g\] \]

or a survival-preservation cost:

\[ L\_{`\mathrm{preserve}`{=tex}} = `\lambda`{=tex}\_D
E\[1-D\_{`\mathrm{eff}`{=tex}}\]. \]

Predefine the sweep.

------------------------------------------------------------------------

# 6. Required Competence Sweep

For each frozen base accuracy approximately equal to:

\[ 50%,60%,70%,80%,90%,95% \]

train a fresh RSL controller.

Use at least:

\[ `\boxed{10\text{ deterministic seeds per competence level}}`{=tex} \]

for the primary run.

Prefer 20 seeds for promising regions.

For each competence level report:

\[ `\Delta `{=tex}Acc(p) \]

\[ c(p)=P(W`\to `{=tex}R) \]

\[ d(p)=P(R`\to `{=tex}W) \]

\[ g(p) \]

\[ `\alpha`{=tex}(p) \]

\[ D\_{`\mathrm{raw}`{=tex}}(p) \]

\[ D\_{`\mathrm{eff}`{=tex}}(p). \]

------------------------------------------------------------------------

# 7. Most Important Plot

Produce:

\[ `\boxed{
p\quad\text{vs}\quad\Delta Acc
}`{=tex} \]

This is the main Phase-2 figure.

Also produce separate figures for:

\[ p`\quad`{=tex}`\text{vs}`{=tex}`\quad `{=tex}P(W`\to `{=tex}R) \]

\[ p`\quad`{=tex}`\text{vs}`{=tex}`\quad `{=tex}P(R`\to `{=tex}W) \]

\[ p`\quad`{=tex}`\text{vs}`{=tex}`\quad `{=tex}E\[g\]. \]

Hypothetical behaviors:

### Useful correction regime

At moderate base competence:

\[ `\Delta `{=tex}Acc\>0, `\qquad`{=tex} c`\gg `{=tex}d. \]

### Abstention regime

At high base competence:

\[ g`\to0`{=tex}, `\qquad`{=tex} `\Delta `{=tex}Acc`\to0`{=tex}. \]

### Aggressive failure regime

\[ g`\to1`{=tex}, `\qquad`{=tex} D\_{`\mathrm{eff}`{=tex}}`\to0`{=tex},
`\qquad`{=tex} d`\text{ becomes large}`{=tex}. \]

All three outcomes are scientifically meaningful.

------------------------------------------------------------------------

# 8. Accounting Identity Must Be Verified Everywhere

For every seed and every competence level verify:

\[ `\boxed{
\Delta Acc=(1-p)c-pd
}`{=tex} \]

within numerical/sampling tolerance.

Store:

``` text
delta_measured
delta_accounting
accounting_error
```

Unit-test the identity.

If it fails materially, stop and debug the metric definitions.

------------------------------------------------------------------------

# 9. Transfer Protocol

The competence sweep alone is not sufficient.

For every frozen base checkpoint:

### RSL training distribution

Train RSL on random linear task directions:

\[ q=(a,b) \]

excluding a predefined held-out angular sector.

### Final transfer distribution

Evaluate on the held-out sector that receives:

-   no RSL reward,
-   no hyperparameter tuning,
-   no checkpoint selection,
-   no early-stopping signal.

Measure:

\[ `\boxed{
\Delta_{\mathrm{transfer}}(p)
}`{=tex} \]

separately from seen-task correction.

This produces two curves:

\[ `\Delta`{=tex}\_{`\mathrm{seen}`{=tex}}(p) \]

and

\[ `\Delta`{=tex}\_{`\mathrm{heldout}`{=tex}}(p). \]

The second curve matters more.

------------------------------------------------------------------------

# 10. No-Leakage Rules

At held-out inference RSL MUST NOT access:

-   target (y);
-   correctness indicator;
-   true reward;
-   supervised loss;
-   true gradient;
-   counterfactual loss requiring (y);
-   Oracle mask;
-   transition label.

Allowed inference features include:

-   hidden states;
-   logits;
-   probabilities;
-   entropy;
-   top-1/top-2 margin;
-   attention statistics;
-   layer disagreement;
-   head disagreement;
-   previous recursive state;
-   previous alpha/gate values;
-   task/query representation if normally available to the base model.

The inference API should work as:

``` python
prediction = model(x, task_context, use_rsl=True)
```

with no `target` argument.

------------------------------------------------------------------------

# 11. Error-Detection Analysis

The intervention gate can be interpreted as an implicit error detector.

Test whether:

\[ g \]

is higher when the base model is wrong.

Compute:

\[ E\[g`\mid `{=tex}W\] \]

and

\[ E\[g`\mid `{=tex}R\]. \]

Desired:

\[ `\boxed{
E[g\mid W] > E[g\mid R].
}`{=tex} \]

Also compute error-detection AUROC using (g) as the score if practical.

This is important even if accuracy does not improve.

If (g) predicts base-model errors but the decay actuator fails to
correct them, the detector and actuator should be diagnosed separately.

------------------------------------------------------------------------

# 12. Intervention Precision and Recall

Define intervention using a threshold (g\>`\tau`{=tex}).

Measure:

### Intervention precision

\[ P(W`\mid `{=tex}g\>`\tau`{=tex}) \]

### Error-intervention recall

\[ P(g\>`\tau`{=tex}`\mid `{=tex}W). \]

Sweep (`\tau`{=tex}) only for analysis unless thresholding is part of a
predeclared model.

This tells us whether RSL knows **when the base model is likely wrong**.

------------------------------------------------------------------------

# 13. Required Baselines

For every competence level include:

1.  frozen base model;
2.  fixed decay;
3.  random decay matched for intervention frequency;
4.  random gate matched for (E\[g\]);
5.  shuffled transition-reward control;
6.  original learned SADT;
7.  gradient-pattern imitation RSL;
8.  balanced transition RSL;
9.  dual-head gated balanced RSL;
10. Oracle privileged controller.

The Oracle is an upper bound only.

------------------------------------------------------------------------

# 14. Oracle Headroom

For each base competence level calculate:

\[ `\Delta`{=tex}\_{`\mathrm{Oracle}`{=tex}}(p). \]

If Oracle improvement is near zero, RSL cannot reasonably be expected to
improve the task through this actuator.

Define actuator headroom:

\[ `\boxed{
H(p)=Acc_{\mathrm{Oracle}}-Acc_{\mathrm{base}}.
}`{=tex} \]

Then compare learned RSL improvement with available headroom:

\[ `\boxed{
\eta_{\mathrm{RSL}}
=
\frac{\Delta Acc_{\mathrm{RSL}}}
{\Delta Acc_{\mathrm{Oracle}}}
}`{=tex} \]

when the denominator is positive.

This distinguishes:

-   controller failure;
-   actuator limitation;
-   task already solved.

------------------------------------------------------------------------

# 15. Recursive Experiment Comes Later

Do NOT add recursion until single-pass RSL shows reproducible positive
correction at one or more competence levels.

Once single-pass works, test:

\[ k`\in`{=tex}{1,2,3,5}. \]

Record per pass:

\[ Acc_k,`\quad`{=tex} c_k,`\quad`{=tex} d_k,`\quad`{=tex}
g_k,`\quad`{=tex} D_k. \]

Stop recursion if confidence/intervention criteria indicate abstention.

Do not assume (k=5) is superior to (k=1).

------------------------------------------------------------------------

# 16. Statistical Protocol

For each competence level:

-   =10 seeds;

-   identical evaluation-set generation policy;

-   paired RSL-on/off comparisons;

-   mean;

-   standard deviation;

-   median;

-   95% confidence interval;

-   fraction of seeds with positive delta;

-   all raw seeds saved.

Do not discard failed seeds.

Do not repeatedly inspect the final held-out set while tuning.

Use a validation sector separate from the final test sector if
hyperparameter selection is required.

------------------------------------------------------------------------

# 17. Recommended Task Geometry

Use three angular regions:

### Training region

All allowed directions except validation/test gaps.

### Validation held-out sector

Used for architecture/reward selection.

### Final test held-out sector

Never used until configuration is frozen.

Example only:

``` text
validation gap:  30°–60°
test gap:        285°–345°
```

Exact sectors should be predeclared and saved in configuration.

------------------------------------------------------------------------

# 18. Recommended Files

Create:

``` text
controlled_competence_rsl.py
experiments/controlled_competence_rsl.yaml
experiments/controlled_competence_rsl.md

tests/test_controlled_competence_rsl.py

results/controlled_competence_seed_table.csv
results/controlled_competence_report.json
results/controlled_competence_diagnosis.md

results/competence_vs_delta.png
results/competence_vs_rescue_damage.png
results/competence_vs_gate.png
```

Preserve existing Phase-1 results.

Do not overwrite historical reports.

------------------------------------------------------------------------

# 19. Minimum Unit Tests

Add tests for:

1.  no target argument required during RSL inference;
2.  target perturbation after inference cannot change predictions;
3.  (0`\le `{=tex}g`\le1`{=tex});
4.  (`\alpha`{=tex}`\ge0`{=tex});
5.  (0\<D\_{`\mathrm{raw}`{=tex}}`\le1`{=tex});
6.  (0\<D\_{`\mathrm{eff}`{=tex}}`\le1`{=tex});
7.  (g=0`\Rightarrow `{=tex}D\_{`\mathrm{eff}`{=tex}}=1);
8.  (g=1`\Rightarrow `{=tex}D\_{`\mathrm{eff}`{=tex}}=D\_{`\mathrm{raw}`{=tex}});
9.  accounting identity;
10. frozen base weights remain bitwise/numerically unchanged during RSL
    training;
11. held-out task geometry excluded from RSL training;
12. final test sector never used for model selection.

------------------------------------------------------------------------

# 20. Decision Tree

## Case A --- RSL fails at every competence level

If:

\[ `\Delta `{=tex}Acc(p)`\le0`{=tex} \]

for essentially all (p), while Oracle has headroom:

> The current inference-visible controller features are insufficient to
> choose useful decay interventions.

Investigate representation/features or actuator limitations.

Do not simply increase reward.

------------------------------------------------------------------------

## Case B --- RSL works only for weak models

Example:

\[ p=0.5`\text{–}`{=tex}0.7: `\Delta `{=tex}Acc\>0 \]

but

\[ p`\ge0.8`{=tex}: g`\to0`{=tex}. \]

Interpretation:

> RSL acts as an error-correction layer when error density is high, but
> rationally abstains as base competence increases.

This would be a legitimate and useful result.

------------------------------------------------------------------------

## Case C --- Seen correction works, held-out transfer fails

Interpretation:

> RSL learns task-specific correction policies but not transferable
> correction.

Do not call it general self-correction.

------------------------------------------------------------------------

## Case D --- Held-out transfer works within linear task manifold

Call it:

\[ `\boxed{\text{local task-manifold correction transfer}}`{=tex} \]

Then increase held-out distance and task diversity.

------------------------------------------------------------------------

## Case E --- Transfer survives structurally different task families

Only then begin investigating stronger claims about general learned
self-correction.

Require independent controls and replication.

------------------------------------------------------------------------

# 21. What Not To Do

Do not:

-   cite the old 3-seed +3.9% result as confirmed evidence;
-   tune repeatedly on the final held-out sector;
-   force the gate to intervene merely to avoid no-op behavior;
-   interpret (g`\to0`{=tex}) as a training bug without checking reward
    optimality;
-   increase recursion before single-pass correction works;
-   call error correction "thinking";
-   call interpolation "semantic reasoning";
-   use target-derived features at inference;
-   hide negative seeds;
-   optimize the benchmark until RSL wins.

------------------------------------------------------------------------

# 22. Current Falsifiable Hypothesis

The next experiment tests:

\[ `\boxed{
\exists\,p^\*:
\Delta_{\mathrm{heldout}}(p^\*)>0
}`{=tex} \]

for a frozen base model and an RSL controller trained only on other task
regions.

A stronger version asks whether there is a competence interval:

\[ `\boxed{
p\in[p_1,p_2]
}`{=tex} \]

where RSL reproducibly provides positive correction before transitioning
toward abstention as (p`\to1`{=tex}).

------------------------------------------------------------------------

# 23. Desired Scientific Outcome

We are not trying to prove RSL works.

We are trying to map the system's behavior:

\[ `\boxed{
\text{base competence}
\rightarrow
\text{intervention policy}
\rightarrow
\text{rescue/damage balance}
\rightarrow
\text{transfer}
}`{=tex} \]

The most informative result may be a phase diagram showing regions of:

1.  aggressive harmful intervention;
2.  useful correction;
3.  rational abstention.

That would be substantially more informative than one accuracy number.

------------------------------------------------------------------------

# 24. Final Instruction to the Next AI Agent

First:

\[ `\boxed{\text{audit the }D_{\mathrm{eff}}\text{ metric}}`{=tex} \]

Then:

\[ `\boxed{\text{freeze controlled-accuracy base checkpoints}}`{=tex} \]

Then:

\[ `\boxed{\text{train only RSL}}`{=tex} \]

Then measure:

\[ `\boxed{
\Delta Acc(p),\;
P(W\to R),\;
P(R\to W),\;
g(p),\;
D_{\mathrm{eff}}(p)
}`{=tex} \]

on both seen and genuinely never-rewarded task regions.

Do not add architectural complexity until this experiment tells us
whether a useful correction regime exists.

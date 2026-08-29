# Semantic Alpha Decay Transformer (SADT)

A small research prototype for **selective semantic decay after normal Transformer softmax attention**.

The core hypothesis is:

> A Transformer should not decay every old token. Instead, a learned semantic controller should decide what information deserves to keep influencing the next prediction. Information judged disposable receives a larger learned decay rate \(\alpha\), and its attention contribution is reduced by \(e^{-\alpha T}\).

This repository is intentionally small. The first goal is not SOTA performance. The goal is to test whether a learned semantic decay controller can selectively suppress irrelevant reasoning states while preserving useful ones.

## Current implementation

The repository now contains a controlled multi-seed harness rather than only the
original fixed-layout sanity check. It provides randomized layouts and lengths,
held-out value pairs, explicit architectural ablations, causal gate interventions,
IID/OOD metrics, deterministic seeds, summary statistics, and unit tests.

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python train.py --seeds 1 --steps 100 --modes baseline learned  # quick smoke run
python train.py --workers 5                                       # five-seed suite
```

Results are written to `results/report.json`. The retained sandbox numbers below
describe the original prototype, not the new counterfactual harness.

The isolated-shift diagnosis is in `results/generalization_diagnosis.md`, with
raw five-seed measurements in `results/generalization_report.json`.

The one-step recursive self-output experiment is summarized in
`results/recursive_diagnosis.md`, with raw data in `results/recursive_report.json`.

---

## 1. Core idea

Start with ordinary scaled dot-product attention.

\[
S_{ij} = \frac{Q_i K_j^T}{\sqrt{d}}
\]

Normal Transformer attention is

\[
A_{ij} = \operatorname{softmax}_j(S_{ij})
\]

The proposed semantic engine predicts a non-negative decay rate

\[
\alpha_{ij} \ge 0
\]

for each query-key relationship.

Then define the semantic survival factor

\[
D_{ij}(T) = e^{-\alpha_{ij} T}
\]

and apply it **after normal softmax**:

\[
\boxed{
\widetilde A_{ij}
=
A_{ij} e^{-\alpha_{ij}T}
}
\]

The attention output becomes

\[
\boxed{
H_i
=
\sum_j
\widetilde A_{ij}V_j
}
\]

or

\[
H_i
=
\sum_j
A_{ij}e^{-\alpha_{ij}T}V_j
\]

This prototype intentionally does **not** renormalize \(\widetilde A\).

That means discarded semantic information is allowed to reduce the total attention magnitude instead of being redistributed automatically to other tokens.

A renormalized variant should be tested separately as an ablation.

---

## 2. Meaning of alpha

\(\alpha\) is **not the attention weight**.

It is the learned semantic decay rate.

Small alpha:

\[
\alpha_{ij} \approx 0
\]

gives

\[
e^{-\alpha_{ij}T} \approx 1
\]

so the information survives.

Large alpha:

\[
\alpha_{ij} \gg 0
\]

gives

\[
e^{-\alpha_{ij}T} \rightarrow 0
\]

so the information contributes less and eventually becomes almost irrelevant.

The intended interpretation is:

| Semantic judgement | alpha | survival |
|---|---:|---:|
| important | near 0 | near 1 |
| uncertain | small/moderate | partial |
| irrelevant/wrong | large | near 0 |

---

## 3. What is T?

In this project, \(T\) should **not automatically mean token age**.

It represents training/reasoning exposure: how strongly the learned semantic judgement is allowed to act.

A practical training version is

\[
\tau =
\min\left(1,\frac{T}{T_{\text{warmup}}}\right)
\]

and then

\[
D_{ij}=e^{-\alpha_{ij}\tau}
\]

Why normalize it?

If literal training step \(T=100000\) is inserted directly into

\[
e^{-\alpha T},
\]

even a tiny positive alpha can collapse to zero and gradients can disappear.

So the first prototype uses normalized training exposure \(\tau\in[0,1]\).

Later experiments can test other definitions of \(T\), including number of reasoning verification passes, confidence updates, or bounded recurrent reasoning steps.

---

## 4. How alpha is predicted

The semantic controller should not assign alpha from token identity alone.

The same word can matter in one problem and be useless in another.

Therefore alpha should depend on both the current query representation and the candidate memory representation.

The prototype uses

\[
z_{ij}
=
[q_i,\;k_j,\;q_i\odot k_j]
\]

then

\[
r_{ij}
=
f_\theta(z_{ij})
\]

and

\[
\boxed{
\alpha_{ij}
=
\operatorname{softplus}(r_{ij})
}
\]

Softplus guarantees

\[
\alpha_{ij}\ge0.
\]

This creates a pairwise semantic controller:

\[
(q_i,k_j)
\longrightarrow
\alpha_{ij}
\longrightarrow
e^{-\alpha_{ij}T}.
\]

---

## 5. Why put the gate after softmax?

This is the architecture being tested:

\[
S
\rightarrow
\operatorname{softmax}
\rightarrow
A
\rightarrow
A e^{-\alpha T}
\rightarrow
\widetilde A.
\]

It preserves the normal Transformer attention calculation first.

Then a separate semantic mechanism decides how much of each already-computed attention connection should survive.

This differs from

\[
\operatorname{softmax}(S-\alpha T),
\]

which places semantic decay inside attention-logit competition.

Both are mathematically reasonable, but they are different hypotheses.

This repository starts with the **post-softmax semantic gate** because that is the target architecture.

---

## 6. Optional information-retention cost

If the model receives no cost for keeping everything, it may learn

\[
\alpha \approx 0
\]

everywhere because preserving all information is often the easiest solution.

A small regularizer can create pressure to retain only useful information.

Let

\[
D_{ij}=e^{-\alpha_{ij}T}.
\]

Define

\[
\mathcal L_{\text{keep}}
=
\frac{1}{N}
\sum_{i,j}D_{ij}.
\]

Then train with

\[
\boxed{
\mathcal L
=
\mathcal L_{\text{next-token}}
+
\lambda_{\text{keep}}
\mathcal L_{\text{keep}}
}
\]

where \(\lambda_{\text{keep}}\) should be small.

Interpretation:

- cross entropy says: keep information required to predict correctly;
- retention cost says: do not keep information merely because you can.

This creates a bottleneck that may encourage semantic discrimination.

Do **not** make this penalty too large or the model will simply suppress everything.

---

## 7. Minimal PyTorch implementation

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticDecayAttention(nn.Module):
    def __init__(self, d_model=128, n_heads=4, alpha_hidden=32):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

        self.alpha_net = nn.Sequential(
            nn.Linear(3 * self.d_head, alpha_hidden),
            nn.Tanh(),
            nn.Linear(alpha_hidden, 1),
        )

    def forward(self, x, T=1.0):
        B, L, D = x.shape
        H = self.n_heads
        Dh = self.d_head

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = q.view(B, L, H, Dh).transpose(1, 2)
        k = k.view(B, L, H, Dh).transpose(1, 2)
        v = v.view(B, L, H, Dh).transpose(1, 2)

        # Normal Transformer scores
        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(Dh)

        # Causal mask
        causal = torch.triu(
            torch.ones(L, L, device=x.device, dtype=torch.bool),
            diagonal=1,
        )

        scores = scores.masked_fill(causal, float("-inf"))

        # Ordinary softmax attention
        A = F.softmax(scores, dim=-1)

        # Pairwise semantic features
        qi = q.unsqueeze(3).expand(-1, -1, -1, L, -1)
        kj = k.unsqueeze(2).expand(-1, -1, L, -1, -1)

        semantic_features = torch.cat(
            [qi, kj, qi * kj],
            dim=-1,
        )

        raw_alpha = self.alpha_net(semantic_features).squeeze(-1)

        # alpha >= 0
        alpha = F.softplus(raw_alpha)

        # Semantic survival
        survival = torch.exp(-alpha * T)

        # Respect causal mask
        survival = survival.masked_fill(causal, 0.0)

        # Proposed post-softmax decay
        A_semantic = A * survival

        # No renormalization in main experiment
        y = A_semantic @ v

        y = y.transpose(1, 2).contiguous().view(B, L, D)
        y = self.out(y)

        return y, {
            "attention": A,
            "alpha": alpha,
            "survival": survival,
            "semantic_attention": A_semantic,
        }
```

---

## 8. Renormalized ablation

Test this separately:

```python
A_semantic = A * survival

A_semantic = A_semantic / (
    A_semantic.sum(dim=-1, keepdim=True) + 1e-8
)
```

Compare it against the unnormalized version.

The distinction matters.

### Unnormalized

\[
\sum_j \widetilde A_{ij}\le1
\]

Semantic rejection can reduce total incoming information.

### Renormalized

\[
\sum_j \widetilde A_{ij}=1
\]

Rejected attention is redistributed among surviving states.

Neither should be assumed superior before testing.

---

## 9. Tiny Transformer block

```python
class SADTBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SemanticDecayAttention(
            d_model=d_model,
            n_heads=n_heads,
        )

        self.ln2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x, T):
        a, info = self.attn(self.ln1(x), T=T)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x, info
```

---

## 10. Tiny language model

```python
class TinySADT(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_len=64,
        d_model=128,
        n_heads=4,
        n_layers=2,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)

        self.blocks = nn.ModuleList([
            SADTBlock(d_model, n_heads)
            for _ in range(n_layers)
        ])

        self.final_ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens, T=1.0):
        B, L = tokens.shape

        pos = torch.arange(
            L,
            device=tokens.device,
        )

        x = (
            self.token_embedding(tokens)
            + self.position_embedding(pos)
        )

        diagnostics = []

        for block in self.blocks:
            x, info = block(x, T=T)
            diagnostics.append(info)

        logits = self.lm_head(self.final_ln(x))

        return logits, diagnostics
```

---

## 11. Training loss

```python
logits, diagnostic = model(tokens, T=tau)

next_token_logits = logits[:, :-1]
next_token_targets = tokens[:, 1:]

lm_loss = F.cross_entropy(
    next_token_logits.reshape(-1, vocab_size),
    next_token_targets.reshape(-1),
)

survival = diagnostic[-1]["survival"]

keep_loss = survival.mean()

loss = lm_loss + lambda_keep * keep_loss
```

Recommended starting value:

```python
lambda_keep = 1e-3
```

Then test

```text
0
1e-4
1e-3
1e-2
```

as an ablation.

---

## 12. Training exposure schedule

Start with

```python
tau = min(1.0, step / warmup_steps)
```

Example:

```python
warmup_steps = 1000
```

At the beginning:

\[
T\approx0
\]

so semantic decay is weak.

Later:

\[
T\rightarrow1
\]

and the controller must learn what should survive.

This prevents the alpha mechanism from destroying representations before the base network has learned anything useful.

---

## 13. First toy experiment: selective recall

Use synthetic sequences such as

```text
<BOS> QUERY_A KEY_A red KEY_B blue noise <ASK>
```

Target:

```text
red
```

and

```text
<BOS> QUERY_B KEY_A red KEY_B blue noise <ASK>
```

Target:

```text
blue
```

The model must determine that the relevance of `red` or `blue` depends on the query.

This is important because semantic decay should not simply memorize:

```text
"red is important"
```

It must learn:

```text
"red is important in this context"
```

The test is whether

\[
D_{\text{relevant}}
>
D_{\text{irrelevant}}
\]

while prediction accuracy remains high.

---

## 14. Sandbox sanity result

A small CPU test was run with approximately:

```text
Baseline parameters: 26,768
Semantic-decay parameters: 27,185
Training steps: 800
Batch size: 128
Optimizer: AdamW
Synthetic selective-recall task
```

Observed final toy-task accuracy:

```text
Baseline:       100%
Semantic decay: 100%
```

For the final query position, the learned survival gate averaged approximately:

```text
Relevant value:   0.438
Irrelevant states: 0.000474
```

This shows that, under the toy objective plus a small information-retention cost, the controller learned to preserve the task-relevant state much more strongly than irrelevant states.

This is only a **sanity check**.

It does not prove semantic reasoning or improved language modeling.

---

## 15. Smallest meaningful research model

Do not begin with a billion-parameter model.

Recommended first LM:

```text
Vocabulary:       2k–8k
Context length:   128
d_model:          128
heads:            4
layers:           4
FFN width:        512
Parameters:       roughly 1–5M depending on vocabulary
```

Train first on controlled synthetic reasoning data.

Then move to approximately:

```text
10M–30M parameters
```

for a real text experiment.

A tiny model makes ablations cheap and lets us inspect every alpha matrix.

---

## 16. Training budget

### Stage A — architecture sanity test

```text
25k–1M parameters
500–5,000 steps
batch size 64–256
synthetic data
CPU or small GPU
```

Goal:

```text
Does alpha learn selective survival?
```

### Stage B — tiny LM

```text
1M–5M parameters
20k–100k optimizer steps
short context
small tokenizer
```

Goal:

```text
Does semantic decay improve next-token loss,
reasoning robustness, or memory efficiency?
```

### Stage C — serious experiment

```text
10M–50M parameters
millions to hundreds of millions of tokens
```

Only move here if Stage A/B show something useful.

---

## 17. Required baselines

Do not publish only SADT.

Compare:

1. Normal Transformer.
2. Transformer + random decay.
3. Transformer + fixed exponential token-age decay.
4. Transformer + learned alpha but no retention penalty.
5. Transformer + learned semantic alpha.
6. Semantic alpha with post-softmax gate.
7. Semantic alpha with post-softmax gate + renormalization.
8. Semantic alpha as a pre-softmax logit bias.

This will reveal whether any gain comes from semantic learning or simply from adding regularization.

---

## 18. Measurements

Record:

```text
validation cross entropy
perplexity
task accuracy
average alpha
average survival
survival of relevant tokens
survival of irrelevant tokens
gradient norms of alpha network
attention entropy
semantic-attention entropy
memory retained per query
```

Define a semantic selectivity score:

\[
\boxed{
R_{\text{select}}
=
\frac{
\mathbb E[D_{\text{relevant}}]
}{
\mathbb E[D_{\text{irrelevant}}]+\epsilon
}
}
\]

Large values mean useful states survive much more strongly.

But a large selectivity score is meaningful only if task accuracy stays high.

---

## 19. Important failure modes

### Collapse to keep everything

\[
\alpha\rightarrow0
\]

everywhere.

Possible fix: weak retention penalty.

### Collapse to delete everything

\[
\alpha\rightarrow\infty
\]

Possible fix: reduce retention penalty, warm up decay, or constrain alpha.

### Numerical underflow

Large \(\alpha T\) produces

\[
e^{-\alpha T}\approx0.
\]

Use bounded \(T\) first.

Optional:

```python
alpha = torch.clamp(alpha, max=10.0)
```

### Fake semantic behavior

The controller may learn position, punctuation, or dataset shortcuts instead of meaning.

Therefore evaluate on counterfactual and out-of-distribution examples.

---

## 20. Counterfactual test

Train on:

```text
QUERY_A ... KEY_A red ... KEY_B blue
```

but evaluate with values, positions, and distractors shuffled.

If the controller truly uses context, the important alpha should move with the semantic role, not with a fixed position.

Test:

```text
QUERY_A KEY_B orange noise KEY_A green
```

Expected:

```text
green survives
orange decays
```

even though their positions were changed.

---

## 21. Reasoning-chain experiment

Later, generate chains such as

```text
Fact 1
Fact 2
Hypothesis A
Intermediate calculation
Contradictory evidence
Corrected hypothesis
Answer
```

Measure whether the model learns:

\[
D_{\text{bad hypothesis}}\downarrow
\]

while retaining:

\[
D_{\text{supporting facts}}\approx1.
\]

This is much closer to the original research goal.

---

## 22. Parameter overhead

The alpha controller can be tiny.

For each attention head:

```python
Linear(3 * d_head, alpha_hidden)
Linear(alpha_hidden, 1)
```

With

```text
d_model = 128
heads = 4
d_head = 32
alpha_hidden = 32
```

the controller adds only a small number of parameters compared with the Transformer.

The main cost is the pairwise \(L\times L\) alpha computation, not parameter count.

---

## 23. More efficient alpha variants

The full pairwise controller is

\[
\alpha_{ij}=f(q_i,k_j).
\]

That costs approximately \(O(L^2)\), like attention itself.

A cheaper token-level version is

\[
\alpha_j=f(h_j).
\]

Then

\[
D_j=e^{-\alpha_jT}.
\]

But this is less contextual.

A middle ground is

\[
\alpha_{ij}
=
\operatorname{softplus}
(
a_i+b_j+q_i^TMk_j
).
\]

Test efficiency later.

Start with the expressive pairwise version.

---

## 24. GitHub repository layout

```text
semantic-alpha-decay/
│
├── README.md
├── requirements.txt
├── train.py
├── experiment.py
│
├── sadt/
│   ├── __init__.py
│   ├── attention.py
│   ├── model.py
│   └── losses.py
│
├── data/
│   └── selective_recall.py
│
├── tests/
│   ├── test_attention.py
│   └── test_decay.py
│
├── experiments/
│   ├── baseline.yaml
│   ├── semantic_decay.yaml
│   └── ablations.md
│
└── results/
    └── .gitkeep
```

---

## 25. Instructions for an AI coding agent

Give the coding agent this goal:

```text
Build the repository exactly as described in README.md.

Do not silently replace the proposed post-softmax semantic decay with
pre-softmax logit bias.

The primary experimental equation is:

A_semantic = softmax(QK^T / sqrt(d)) * exp(-alpha * T)

where alpha >= 0 is learned from contextual semantic representations.

Implement both the primary unnormalized version and renormalized/pre-softmax
variants only as explicit ablations.

First reproduce the synthetic selective-recall experiment.

Log:
- validation accuracy
- baseline loss
- SADT loss
- mean relevant survival
- mean irrelevant survival
- semantic selectivity ratio
- alpha histograms

Keep the first model below 5 million parameters.

Add deterministic seeds and unit tests.

Do not claim semantic understanding from the toy task.
Only report measurable behavior.
```

---

## 26. Research hypothesis

The conservative hypothesis is:

\[
\boxed{
\text{A learned post-softmax exponential semantic gate can improve selective information retention.}
}
\]

The stronger hypothesis is:

\[
\boxed{
\text{Training a model to decide what information should survive creates pressure to learn contextual semantic relevance.}
}
\]

The strong hypothesis must be demonstrated experimentally rather than assumed.

---

## 27. What would count as a real positive result?

A useful result is not merely:

```text
training loss decreased
```

A much stronger result would be:

```text
same or better validation loss
+
higher robustness to distractors
+
relevant states receive consistently larger survival
+
irrelevant states are suppressed
+
behavior generalizes when token positions and surface forms change
+
active semantic memory decreases
```

If that happens consistently across seeds and tasks, the architecture becomes genuinely interesting.

---

## 28. First command

```bash
python train.py --seeds 1 --steps 100 --modes baseline learned
```

Start tiny.

If the gate does not learn semantic selectivity on the toy task, do not scale the model.

Fix the mechanism first.

---

## 29. Gradient-Pattern Recursive Self-Learning

### Overview & Hypothesis

Ordinary supervised training computes loss $L_t = \text{CE}(y_t, \hat{y}_t)$ and backpropagation gradients $g_t = \nabla_\theta L_t$ solely to update model weights.

**Gradient-Pattern Recursive Self-Learning (RSL)** tests whether backpropagation corrections can also provide a training signal for a secondary controller:

$$
(h, g, \text{gradient statistics}) \longrightarrow \text{RSL training target} \longrightarrow R_\phi(h, \text{context}) \longrightarrow \hat{\alpha}_{ij} \longrightarrow D_{ij} = e^{-\hat{\alpha}_{ij} T}
$$

The scientific question is:
> *Can a secondary controller observe how backpropagation corrects a Transformer during training, learn recurring patterns of those corrections, and predict useful retention/decay decisions on unseen examples without access to labels or gradients at inference time?*

---

### Methodology & Teacher/Student Separation

The experiment strictly separates training-time gradient observation from inference:

1. **Phase A (Teacher Gradient Signal)**:
   During supervised training on the base Transformer, the teacher extracts exact gradient sensitivity and counterfactual loss interventions:
   $$s_{ij} = - \frac{\partial L}{\partial D_{ij}} = - \left\langle \frac{\partial L}{\partial \widetilde{A}_{ij}}, A_{ij} \right\rangle$$
   $$\Delta L_{ij} = L_{\text{attenuated } j} - L_{\text{normal}}$$
   - If $s_{ij} > 0$ (attenuation increases loss): Token $j$ is **useful** $\implies$ target $D^*_{ij} \to 1.0, \alpha^*_{ij} \to 0$.
   - If $s_{ij} < 0$ (attenuation decreases loss): Token $j$ is **harmful** $\implies$ target $D^*_{ij} \to 0.0, \alpha^*_{ij} \gg 0$.
   - The RSL controller $R_\phi$ is trained on detached inference activations $z_{ij} = [q_i, k_j, q_i \odot k_j, |q_i - k_j|, A_{ij}]$ to predict $D^*_{ij}$.

2. **Phase B (Gradient-Free Inference)**:
   At test time on validation/OOD examples, no ground-truth targets, loss values, or backpropagation passes are provided. The controller predicts $\hat{\alpha}_{ij} = \text{softplus}(-R_\phi(z_{ij}))$ directly from internal activations, applying:
   $$\widetilde{A}_{ij} = A_{ij} e^{-\hat{\alpha}_{ij} T}$$

3. **Multi-Pass Recursive Reasoning ($k \in \{1, 2, 3, 5\}$)**:
   For pass $k$, updated internal states $h^{(k)}$ are fed through the controller:
   $$h^{(k)} \longrightarrow R_\phi \longrightarrow \alpha^{(k)} \longrightarrow D^{(k)} = e^{-\alpha^{(k)} \Delta T} \longrightarrow h^{(k+1)}$$
   Metrics track accuracy, selectivity $S_k = \mathbb{E}[D_{\text{useful}}^{(k)}] - \mathbb{E}[D_{\text{harmful}}^{(k)}]$, transition rates $P(\text{wrong} \to \text{right})$ vs $P(\text{right} \to \text{wrong})$, and wrong persistence (error loops).

---

### 5-Seed Benchmark Results

Comprehensive evaluation over 5 deterministic seeds (800 steps, batch size 128, mean $\pm$ standard deviation):

| Model | IID | Unseen Combinations | Unseen Layout | More Distractors | Reversed Order | Combined Shift |
|---|---|---|---|---|---|---|
| **Transformer Baseline** | .652 ± .050 | .099 ± .010 | .589 ± .052 | .587 ± .024 | .570 ± .022 | .204 ± .047 |
| **Learned SADT ($\mathcal{L}_{\text{keep}}$)** | .671 ± .098 | .116 ± .064 | .651 ± .084 | .593 ± .028 | .561 ± .038 | .221 ± .032 |
| **Fixed Decay ($D=e^{-1}$)** | .639 ± .047 | .099 ± .018 | .589 ± .018 | .549 ± .020 | .558 ± .030 | .237 ± .045 |
| **Blind Recursive ($k=1 \to 5$)** | .611 $\to$ .616 | .093 $\to$ .094 | .608 $\to$ .608 | .600 $\to$ .597 | .569 $\to$ .567 | .170 $\to$ .170 |
| **Gradient RSL ($k=1$)** | .648 ± .094 | .100 ± .025 | .594 ± .036 | .573 ± .025 | .566 ± .024 | .211 ± .072 |
| **Gradient RSL ($k=5$)** | .474 ± .061 | .179 ± .060 | .414 ± .051 | .396 ± .075 | .432 ± .053 | .194 ± .044 |
| **Random Controller ($k=1 \to 5$)** | .555 $\to$ .472 | .245 $\to$ .245 | .499 $\to$ .406 | .481 $\to$ .377 | .509 $\to$ .419 | .265 $\to$ .228 |
| **Shuffled Target Controller ($k=5$)** | .471 ± .052 | .182 ± .065 | .407 ± .047 | .390 ± .092 | .431 ± .049 | .201 ± .055 |
| **Oracle Controller (Upper Bound)** | **.788 ± .035** | **.123 ± .061** | **.690 ± .051** | **.630 ± .024** | **.658 ± .052** | **.263 ± .087** |

---

### Key Findings & Scientific Verdict

1. **Oracle Upper Bound Proves Mechanism Validity**: When true supervised correction targets are provided at test time, post-softmax decay increases accuracy across all distribution shifts (+13.6% IID, +10.1% layout, +8.8% reversed order) and produces positive selectivity $\Delta D > 0$.
2. **Gradient-Free Inference Fails to Generalize OOD**: Without true test labels/gradients, the RSL controller's predictions collapse to near-zero selectivity ($\Delta D \approx 0.0000$), behaving identically to the control baseline trained on scrambled/shuffled targets.
3. **Self-Reinforcing Error Loops Under Recursion**: On structured reasoning splits, recursive passes through ungrounded decay degrade accuracy ($P(\text{right} \to \text{wrong}) \approx 25.6\% - 29.8\%$ vs $P(\text{wrong} \to \text{right}) \approx 11.2\% - 12.4\%$), as errors in pass 1 corrupt the inputs to subsequent passes.

> **Scientific Conclusion**: **Negative / Falsification.**
> Backpropagation provides rich supervised correction information, but a secondary controller observing training-time backpropagation cannot infer those corrections from internal activation patterns on unseen examples without supervision.

---

### Reproduction Commands

```bash
# Run unit tests verifying teacher/student isolation, oracle validity, and shuffled controls
python -m pytest tests/test_gradient_rsl.py

# Run full 5-seed Gradient-Pattern RSL experiment suite
python gradient_rsl.py --seeds 5 --steps 800 --output results/gradient_rsl_report.json
```

See `results/gradient_rsl_diagnosis.md` for the full diagnostic breakdown and `results/gradient_rsl_report.json` for raw seed-level data.

---

## 30. Balanced Transition-Focused Reinforcement Learning (Transition RL)

### Overview & Motivation

Following the empirical findings of Gradient-Pattern RSL and the specifications in [`gradient_rsl_research_cookbook.md`](file:///c:/Users/user/Downloads/time%20pass/gradient_rsl_research_cookbook.md), this experiment transitions from supervised gradient-imitation to **Balanced Transition-Focused Reinforcement Learning** across a multi-task continuous angular rule manifold.

The central hypothesis is:
> *Balanced transition-based reinforcement learning can train an intervention-gated decay controller to rescue errors ($W \to R$) while preserving already-correct states ($R \to R$), transferring a self-correction policy to never-rewarded held-out task sectors without target leakage.*

---

### Dual-Head Architecture & Intervention Gating

Rather than forcing a single decay scalar $\alpha$ to simultaneously determine *whether* and *how strongly* to intervene, the controller uses a **dual-head architecture**:

1. **Intervention Decision Head**:
   $$g_{ij} = \sigma(f_g(q_i, k_j, A_{ij}, \text{uncertainty})) \in [0, 1]$$
2. **Decay Magnitude Head**:
   $$\alpha_{ij} = \text{softplus}(f_\alpha(q_i, k_j, A_{ij}, \text{uncertainty})) \ge 0$$
3. **Net Survival Factor**:
   $$D_{ij} = (1 - g_{ij}) + g_{ij} \cdot e^{-\alpha_{ij} T}$$

- When $g \approx 0 \implies D \approx 1.0$ (leave attention unchanged).
- When $g \approx 1 \implies D \approx e^{-\alpha T}$ (apply selective semantic decay).

Inference-available uncertainty features include max prediction probability $\max P(y)$, margin $P_{(1)} - P_{(2)}$, predictive entropy $H(P)$, and attention entropy. At test time on held-out sectors, **zero targets, loss values, or supervised gradients are accessible**.

---

### Balanced Transition Reward & Accounting Identity

The policy is trained via policy gradient with a balanced transition reward matrix:
$$R(W \to R) = +1.0 \quad (\text{Rescue}), \quad R(R \to R) = +1.0 \quad (\text{Preserve})$$
$$R(R \to W) = -1.0 \quad (\text{Damage}), \quad R(W \to W) = -1.0 \quad (\text{Failure})$$

with a preservation regularizer $\mathcal{L}_{\text{preserve}} = \lambda_p \cdot \mathbb{E}[g(1 - e^{-\alpha T})]$.

The expected accuracy delta is strictly governed by the **Correction Accounting Identity**:
$$\boxed{\Delta \text{Acc} = (1 - p)c - pd}$$
where $p = P(\text{base correct})$, $c = P(W \to R \mid \text{wrong})$ (rescue rate), and $d = P(R \to W \mid \text{correct})$ (damage rate).

---

### 10-Seed Empirical Benchmark Results (800 Steps)

| Metric | Seen Directions (Train Manifold) | Held-Out Sector (Never-Rewarded $[\pi/6, \pi/2]$) |
|---|---|---|
| **Base Model Accuracy ($p$)** | $94.22\% \pm 3.19\%$ | $93.87\% \pm 2.36\%$ |
| **RSL-Corrected Accuracy** | $94.22\% \pm 3.19\%$ | $93.87\% \pm 2.36\%$ |
| **Paired Transfer Delta ($\Delta_{\text{transfer}}$)** | $+0.0000 \pm 0.0000$ | $+0.0000 \pm 0.0000$ |
| **Positive Transfer Rate ($\Delta > 0$)** | **0 / 10 seeds (0%)** | **0 / 10 seeds (0%)** |
| **Rescue Rate ($c = P(W \to R)$)** | $0.0000$ | $0.0000$ |
| **Damage Rate ($d = P(R \to W)$)** | $0.0000$ | $0.0000$ |
| **Mean Intervention Gate ($g$)** | $0.0005 \pm 0.0010$ | $0.0005 \pm 0.0010$ |
| **Mean Alpha ($\alpha$)** | $0.00027 \pm 0.00024$ | $0.00027 \pm 0.00024$ |
| **Mean Net Survival ($D$)** | $0.5833 \pm 0.0000$ | $0.5833 \pm 0.0000$ |

---

### Key Diagnostic Takeaways

1. **Statistical Falsification of Small-Sample Artifacts**: Preliminary 3-seed trials showed a noisy $+3.9\% \pm 4.2\%$ transfer delta. Tested over 10 deterministic seeds, the true transfer effect size is zero ($0 / 10$ seeds positive).
2. **Base Model Dominance & The No-Op Policy Trap**: When base accuracy is high ($p \approx 94\%$), the accounting identity requires $\frac{c}{d} > \frac{0.94}{0.06} \approx 15.7\times$ rescue-to-damage ratio to gain net accuracy. The RL policy gradient converges to an inactive no-op controller ($g \to 0$), as the penalty of corrupting correct states outweighs the opportunity to rescue rare errors.

---

### Reproduction Commands

```bash
# Run unit tests and zero-leakage audit
python -m pytest tests/test_balanced_rsl.py

# Run full 10-seed Balanced Transition RL benchmark
python balanced_rsl.py --seeds 10 --steps 800 --output results/balanced_rsl_report.json --csv-output results/balanced_rsl_seed_table.csv
```

See `results/balanced_rsl_diagnosis.md` for the complete diagnosis and `results/balanced_rsl_seed_table.csv` for per-seed measurements.

---

## 31. Phase 2: Controlled Base Competence & Autonomous Teacher-Free Recursion

### Overview & Metric Audit

Phase 2 resolves the code audit on effective survival and tests two critical hypotheses:
1. **Competence Sweep**: Does RSL learn useful correction rather than abstention when base competence is calibrated across $p \in \{0.50, 0.60, 0.70, 0.80, 0.90, 0.95\}$?
2. **Autonomous Teacher-Free Recursion**: Once the external RL reward teacher is removed, can the learned internal dynamics autonomously produce recursive self-correction ($k \in \{1, 2, 3, 5\}$), or does recursion without supervision amplify errors?

**Metric Audit**: In previous reports, global tensor averaging of $D$ included the upper-triangular causal mask ($21/36 = 0.5833$). Audited metrics isolate valid unmasked positions:
- Mean gate: $g \approx 0.0005$
- Mean effective survival: $D_{\text{eff}} = (1 - g) + g \cdot e^{-\alpha T} = 1.0000$ (exact abstention verification).

---

### Controlled Competence Sweep Results (60 Runs: 6 Levels $\times$ 10 Seeds)

| Base Competence ($p$) | Test Base Acc | Test RSL Acc | Transfer $\Delta$ | Mean Gate ($g$) | Effective Survival ($D_{\text{eff}}$) | Oracle Headroom ($H(p)$) |
|---|---|---|---|---|---|---|
| **$p \approx 0.50$** (Step 30) | $24.96\% \pm 1.34\%$ | $24.96\% \pm 1.34\%$ | $+0.0000$ | $0.0024$ | $1.0000$ | $+6.86\%$ |
| **$p \approx 0.60$** (Step 60) | $20.80\% \pm 6.18\%$ | $20.80\% \pm 6.18\%$ | $+0.0000$ | $0.0028$ | $1.0000$ | $+11.39\%$ |
| **$p \approx 0.70$** (Step 120) | $44.92\% \pm 20.9\%$ | $44.92\% \pm 20.9\%$ | $+0.0000$ | $0.0023$ | $1.0000$ | $+18.00\%$ |
| **$p \approx 0.80$** (Step 220) | $87.05\% \pm 8.75\%$ | $87.05\% \pm 8.75\%$ | $+0.0000$ | $0.0020$ | $1.0000$ | $+6.95\%$ |
| **$p \approx 0.90$** (Step 450) | $91.80\% \pm 3.91\%$ | $91.80\% \pm 3.91\%$ | $+0.0000$ | $0.0021$ | $1.0000$ | $+5.89\%$ |
| **$p \approx 0.95$** (Step 800) | $93.09\% \pm 3.42\%$ | $93.09\% \pm 3.42\%$ | $+0.0000$ | $0.0019$ | $1.0000$ | $+5.09\%$ |

Across all competence levels, single-pass RSL on frozen base models converges to rational abstention ($g \to 0, D_{\text{eff}} \to 1.0$).

---

### Autonomous Teacher-Free Recursion Results ($k=1 \to 5$ Passes, 10 Seeds)

When the external RL reward teacher is removed and internal representations are iteratively refined:

| Pass ($k$) | Mean Accuracy | Step Delta | Cumulative $\Delta$ | Rescue Rate ($c_k$) | Damage Rate ($d_k$) | Damage / Rescue Ratio | Error Loop Rate |
|---|---|---|---|---|---|---|---|
| **Pass 1 ($k=1$)** | $44.84\% \pm 22.0\%$ | $+0.00\%$ | $+0.00\%$ | $0.00\%$ | $0.00\%$ | $-$ | $0.00$ |
| **Pass 2 ($k=2$)** | $32.85\% \pm 8.03\%$ | $-11.98\%$ | $-11.98\%$ | $21.49\%$ | **$56.31\%$** | **$2.62\times$** | **$0.80$ (80%)** |
| **Pass 3 ($k=3$)** | $35.50\% \pm 12.1\%$ | $+2.65\%$ | $-9.34\%$ | $17.17\%$ | $23.81\%$ | $1.39\times$ | $0.20$ (20%) |
| **Pass 4 ($k=4$)** | $33.34\% \pm 7.95\%$ | $-2.16\%$ | $-11.49\%$ | $10.26\%$ | $21.22\%$ | $2.07\times$ | $0.40$ (40%) |
| **Pass 5 ($k=5$)** | $34.52\% \pm 9.41\%$ | $+1.18\%$ | $-10.31\%$ | $8.59\%$ | $11.09\%$ | $1.29\times$ | $0.20$ (20%) |

---

### Scientific Verdict

1. **Training RSL with RL is External Supervision, Not Self-Learning**:
   The RL controller's training relies entirely on external reward signals derived from true ground-truth targets $y$.
2. **Autonomous Recursion Amplifies Errors**:
   Once the external reward teacher is removed, recursive state feedback without ground-truth verification damages correct representations ($d_2 = 56.31\%$) at more than $2.6\times$ the rate it rescues errors ($c_2 = 21.49\%$), collapsing accuracy by $-11.98\%$ and triggering self-reinforcing error loops in 80% of seeds.

---

### Reproduction Commands

```bash
# Run unit tests for competence sweep and autonomous recursion
python -m pytest tests/test_controlled_competence_rsl.py tests/test_autonomous_recursive.py

# Run controlled competence sweep
python controlled_competence_rsl.py --seeds 10 --rsl-steps 400

# Run autonomous teacher-free multi-pass recursion test
python autonomous_recursive_test.py --seeds 10 --k-passes 5 --competence-p 0.70
```

See `results/controlled_competence_diagnosis.md` for the full diagnostic breakdown.




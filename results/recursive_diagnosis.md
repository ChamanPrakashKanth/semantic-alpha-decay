# Recursive output-learning result

Five seeds, 800 steps, unchanged model size. Pass one predicts normally. Its
detached answer token is appended to the same sequence, the query is repeated,
and the same weights produce pass two. Both answers are supervised against the
real target; no pseudo-label is treated as truth.

## Learned-alpha result

| Case | First accuracy | Recursive accuracy | Change | Answer agreement | Wrong -> right | Right -> wrong | Recursive delta_D |
|---|---:|---:|---:|---:|---:|---:|---:|
| IID | .614 | .614 | -.001 | .792 | .104 | .104 | -.0068 |
| Unseen combinations | .087 | .087 | +.000 | .963 | .018 | .018 | +.0002 |
| Unseen layout | .596 | .587 | -.009 | .827 | .082 | .091 | +.0007 |
| More distractors | .603 | .594 | -.009 | .831 | .079 | .089 | +.0103 |
| Reversed order | .578 | .571 | -.007 | .844 | .074 | .081 | -.0051 |
| Combined | .179 | .180 | +.000 | .916 | .037 | .037 | -.0095 |

## Diagnosis

One-step recursive output feedback does not create learning signal that resolves
the failure:

- On unseen combinations, the model copies its first answer 96.3% of the time.
  Corrections and newly introduced errors are equal, so net accuracy is unchanged.
- On IID data, roughly 10.4% of examples are corrected and 10.4% are corrupted.
  The second pass changes answers but has no reliable criterion for improving them.
- Recursive semantic selectivity retains the same shortcut signature: positive
  under extra distractors, negative under reversal and combined shifts.
- Baseline and constrained-alpha models show the same pattern: mostly copying,
  with corrections canceled by corruptions. This is not specific to alpha capacity.

The model is learning a response-to-feedback transformation, but not a verifier.
Its own answer contains no new evidence about whether that answer is correct.
Additional blind recursion would therefore repeat or amplify errors. A useful
next recursive experiment would require an external verification signal or
contradictory evidence, not another ungrounded copy of the model's output.

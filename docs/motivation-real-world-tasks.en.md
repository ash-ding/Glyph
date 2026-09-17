# Motivation — frontier blind-spots and trained specialists

The framing this benchmark is really built for, and the real-world tasks it maps to.

## The framing

A frontier model is a generalist. On any given task, part of the difficulty is
something it is **good at** — structure it can reason about, code it can write, patterns it
can infer — and part is something it is **structurally bad at**: a dense, low-structure mapping
with no rule to reason out and too many entries to hold in context. For the second part the
right move is not "reason harder" or "retrieve more" — it is **train a small specialist** that
absorbs it.

Glyph turns that into a measured question:

> **π ≈ how much of a task falls in the frontier model's blind spot.**
> Can a generalist agent recognize the blind spot, decide when to delegate it to a trained
> specialist, and actually make that specialist help — and at what π does that decision pay off?

### How Glyph encodes it

| Glyph piece | Real-world reading |
|---|---|
| **tables** (frozen random MLP, no rule) | the frontier's structural blind spot — a mapping it can't reason out |
| **skeleton** (describable composition) | what the frontier is already good at — solve in-context |
| **π** = skeleton's share of difficulty | the *complement* of "how much of the task is the blind spot" (low π = mostly blind spot) |
| **train arm** (`build_dataset`/`train`/`student_infer`) | the "recognize weakness → train a specialist" capability itself |
| **weights ceiling** | how far a trained specialist can absorb the blind spot at a given data budget |
| **`unary/binary_coupling`** | how *learnable/structured* the blind spot is — the "easy vs hard modality" knob |

Low π → most of the task is the blind spot → a specialist should pay off big. High π → the
frontier is already good → training a specialist is wasted, just reason it out. The benchmark
measures whether the agent makes that call correctly and whether the specialist actually helps.

## Real tasks it maps to — "frontier bad at it → train a specialist"

### 1. Specific modalities / signal types
The frontier reads raw non-text signals poorly; a small model trained on that modality wins.
- Raw time-series: ECG, EEG, accelerometer / IMU, industrial telemetry, seismic traces, spectrograms.
- Niche audio/speech; protein / DNA sequences; molecular graphs; point clouds.
- Domain imagery where a fine-tuned vision model beats a general VLM: medical scans, satellite,
  microscopy, industrial defect detection.

The agent flow: try itself (describe, reason over the signal) → do only okay → train/fine-tune a
small specialist classifier or regressor for that modality.

### 2. Structure-less precise mappings (closest to Glyph's random table)
A mapping with no rule the model can reason out, but cheap to sample.
- Black-box scoring / ranking functions, proprietary pricing models, calibration curves.
- A simulator's input→output with no closed form.
- Idiosyncratic fuzzy-matching / entity resolution on one dataset's patterns.

The model can't derive it; a small regressor trained on samples nails it.

### 3. High-throughput classification / extraction (distilled specialists)
On a narrow slice, a distilled small model is both *more accurate* than the generalist and far
cheaper at scale.
- Routing, moderation, intent classification, domain-specific NER / extraction.

This is the blind-spot framing crossed with serving economics: the specialist wins on both
quality (narrow focus) and cost (amortized once vs paid per call).

### 4. Agent-as-AutoML / self-improvement (the newest, most defensible)
A long-horizon agent that, mid-task, notices "I keep getting this sub-task wrong — let me train a
model for it," writes the training code, fine-tunes, evaluates, and folds the specialist back in.
This autonomous-ML-engineering capability has almost no clean testbed. Glyph is one: the right
answer is known (via the ceilings), so you can measure whether the agent *decides* correctly and
whether training *helps*.

## The honest caveat — and why it is a controllable axis, not a dead end

In the real cases above, training a specialist wins because the modality **has structure**: a
small model generalizes from limited data to inputs it never saw. Glyph's table is a **random
MLP — the worst case**: the specialist can only memorize seen cells (plus weak digit-wise
extrapolation), so by the letter Glyph *understates* how much a specialist helps on a structured
real modality.

But that gap is exactly what Glyph's `unary/binary_coupling` knob controls: turn it up and the
blind spot becomes more digit-wise-learnable — a specialist starts to generalize instead of just
memorize. So Glyph can model the **whole spectrum from a hard modality (specialist can only
memorize) to an easy one (specialist generalizes)** — the very axis that decides whether "train a
specialist" is worth it, and one you cannot dial on real data.

## What this framing demands of the benchmark

To be credible under this framing, the benchmark must be able to measure two things:

1. **Does the agent correctly recognize the blind spot and decide to train a specialist?**
   (the decision, not just the final score.)
2. **Can a specialist that actually generalizes exist here** — which requires (a) turning up the
   coupling so the blind spot is learnable, and (b) giving the train arm **enough budget to train
   a useful specialist at all.**

The pilot runs are the cautionary evidence: the agent *did* try to train a specialist, and it
failed — but for harness reasons (≈50 gradient steps, no clean cell-level data), not because the
idea is wrong. See `open_questions.md`, "the train arm cannot obtain enough data to train a usable
student." Under this framing that is not a detail — **if the protocol starves the specialist, you
can never measure when delegating to a specialist pays off**, which is the whole point.

## One-line positioning

> A generalist agent, facing a task partly outside its competence, decides when to train a
> specialist to cover the gap — and π is a measured knob for asking "how large must the gap be
> before that decision is worth it." Glyph is a controlled testbed for that decision; the coupling
> knob lets it span hard-to-learn and easy-to-learn blind spots that real data never lets you vary.

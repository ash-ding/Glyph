# Tools

The layer between the frozen data layer and the arms. This file defines **what
the environment can do**; which arm may do which is a separate table at the
bottom, and deliberately so.

Implemented in `src/glyph/agent/schema.py` (declarations and allocation) and
`src/glyph/agent/tools.py` (implementations). 128 tests.

## Why the two are separated

They used to be the same act. `schema.py` built the tool list inside the
container check:

```python
if Container.WEIGHTS in allowed:
    tools += [declare_target, synthesize_data, train, evaluate, inspect_failures]
```

So "this tool exists" and "this arm may use it" were one line of code, and a
tool landed in an arm by where it happened to be typed. `evaluate` was
weights-only for exactly that reason — nobody decided it.

The arms are supposed to differ **only in which container they may spend on**.
Declaring tools in one table and allocating them in another makes that invariant
readable, and a test now asserts it.

## Discipline: a tool is fully implemented or it does not exist

No parameter may be accepted and ignored.

The cost of breaking this is not "a missing feature". `synthesize_data` declared
`description`, `emphasis` and `include_reasoning` and read none of them. The
agent filled `emphasis: "operator-focused"` on every call of one run and wrote
increasingly detailed descriptions — by the third it had derived the structural
operators' permutation rules explicitly — believing it was designing a
curriculum. It received the same rows at three lengths. It was operating a
control that was not connected, and the trace recorded the intent as though it
had an effect.

`test_every_declared_parameter_is_read_by_the_implementation` now walks the
declarations and checks each parameter against the implementation's signature,
so this cannot return silently.

---

## The v1 tool set

Nine tools. Everything here is implemented; anything not needed yet is absent
rather than stubbed.

### Acquisition

```
query(exprs: list[str], why: str) -> {results, malformed, queries_so_far}
```
The only way to learn semantics. Malformed expressions are billed like any
other — probing the grammar for free would make the public syntax spec worth
less than it is meant to be. `why` is recorded, not acted on, and says so.

### Building a training set

```
build_dataset(source: "queries"|"demos"|"mixture", n: int) -> {dataset_id, size, drawn_from}
```
Repeats purchased rows up to `n`: the agent cannot manufacture labels it has not
bought, so `n` is an epoch count wearing a costume, and the description says so.
Curriculum control is **absent**, not stubbed — see deferred.

### Producing an artifact

```
set_context(text: str)                          -> {artifact_id, chars}
write_code(src: str)                            -> {artifact_id, bytes}
train(dataset_id: str, epochs: int, lr: float)  -> {artifact_id, examples, final_loss}
```

All three register an artifact and return its id, which is what makes the
evaluation tools uniform. One artifact is exactly one production call rather
than a running snapshot, so `evaluate(artifact_id)` is unambiguous about what it
measured.

**`train` is always a full fine-tune.** `HParams` carried the flag but the
schema never exposed it and `lora_rank` had a minimum of 4, so the arm could
only do LoRA — while the published weights-arm ceiling (0.498 at 10% of the
table seen) was measured with full fine-tuning. The arm was being compared
against a line it could not reach. LoRA is deferred.

### Self-assessment

```
evaluate(artifact_id: str, n: int) -> {dev_accuracy, n}
inspect(artifact_id: str, k: int)  -> {n_inspected, n_wrong, examples}
```

Both take **any** artifact and route through `seal.answer_with`.

`inspect` evaluates `k` items rather than running the whole dev split and
truncating the output, which is what the previous version did.

### Declaration

```
declare_target(role: Role, rationale: str) -> {ok, role}
```

Its description says plainly that it is **declarative**: all seven roles produce
identical training, so "the agent chose `world_model`" is a fact about how it
described itself. Making the roles operationally distinct is deferred.

### Termination

```
seal(artifact_id: str, summary: str) -> {sealed, digest}
```

Takes an artifact rather than an entry plus a checkpoint id, so **the thing
sealed is the thing that was evaluated** — an agent can no longer measure one
object and hand over another.

A forced seal by the harness always succeeds. An agent whose training all failed
still owes a comparable data point: the base student with no prompt and no
adapter is a legitimate, and very bad, artifact. Refusing turns a poor result
into a missing one, and the arms most likely to hit that are the ones whose
preparation costs most.

---

## One answering path, shared

```python
seal.answer_with(artifact, base_model, ledger, exprs) -> list[str]
```

Used by the tool layer at dev time and by all three arm runners at test time.

This is what made universal self-assessment possible at all: `evaluate` was
checkpoint-only by construction, so ungating it would have handed A2 (a string)
and A4 (source code) a tool that returns an error. But the second benefit
matters more. Two implementations could drift — different answer cleaning,
different caching — and the agent would then be steering on a number produced
differently from the one it is finally graded by. That class of bug is now
impossible rather than unlikely, and the calibration measurement below depends
on it.

---

## Allocation

Declared here, not inside the tool definitions.

| | `query` | `build_dataset` | `set_context` | `write_code` | `train` | `evaluate` | `inspect` | `declare_target` | `seal` |
|---|---|---|---|---|---|---|---|---|---|
| **A2** context | ✓ | — | ✓ | — | — | ✓ | ✓ | — | ✓ |
| **A4** code | ✓ | — | — | ✓ | — | ✓ | ✓ | — | ✓ |
| **A6** weights | ✓ | ✓ | — | — | ✓ | ✓ | ✓ | ✓ | ✓ |
| **A7** free choice | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

**Self-assessment is universal** (#13, settled). It is not a property of a
container: someone writing a prompt can try it, someone writing code can run it,
someone training a model can hold out a validation set. Issuing it to one arm
measured the harness rather than the container — A2 wrote its prompt blind and
sealed while A6 ran a full train/evaluate/inspect loop, and the dev signal is
known to shape behaviour: once it stopped reading zero, one agent went from
buying 187 facts to 565.

A0′ has no agent loop and takes no tools: the harness buys evidence and the
frontier answers directly.

**A7 is one row, not a module.** Under this structure the free-choice arm is an
allocation, which is most of why the separation was worth doing.

---

## What the dev split is, and what it is not

`DEV_FRACTION = 0.15` of the agent's own query log. Handing over a free labelled
dev set would quietly refund the query budget, so the agent pays for its own
feedback.

**It does not estimate test, and that is now measured rather than hidden**
(#14, settled as "leave the environment alone"). The agent buys single-level
probes because isolating one table entry is how you read a table, and it is
right to — but almost nothing on the test set looks like that:

| | single-level share |
|---|---|
| what one run purchased | 91.5% |
| the test set | 12.5% |

That run reached **dev 0.400 against test 0.035**. The loop is worse than the
gap: once dev stopped reading zero, the agent bought more of the same probes,
trained on them, watched dev rise, and continued — a clearer signal made it
optimise harder in the wrong direction.

The argument for leaving it alone: an agent with a narrow probing strategy
*should* get a misleading self-estimate, and whether it knows how well it is
doing is part of what the experiment asks. Selling a matched-distribution dev
set would hand the agent the shape of the exam, and inferring that shape is
itself a capability worth observing.

So `ScoreReport.calibration` carries the dev accuracy, the test accuracy, the
gap, and the depth histograms of what was purchased against what was scored —
the gap arrives with the distributions that explain it. Dev is scored through
the same `answer_fn` as the test set, so both numbers describe one artifact.

This matters more after #13 than before: `evaluate` is now universal, so all
three arms share the same compass and it points the same wrong way for each.

---

## Deferred

**LoRA.** All training is full fine-tuning, to keep one fewer variable between
the arm and the ceiling it is measured against. At matched learning rate LoRA
scored 0.715 against full's 0.797 on binary, and rank had no measurable effect
between 4 and 128. Revisit if the question becomes "how cheaply can the
capability be installed" rather than "can it be installed".

**Curriculum control.** `filter_data` and a working `emphasis` are what H3's
"the biggest lever is data curation" needs to be testable at all (#12). Absent
from v1 because a declared-and-inert tool is worse than a missing one.

**Operationally distinct roles.** Making `declare_target`'s seven roles produce
different training — a weighted loss for proposal, scores rather than answers
for value, a search loop for world model — is a larger change and belongs after
the arms are comparable (#16).

**Combining artifacts.** A7 can produce a context *and* an adapter, but each
production call registers its own artifact and `seal` takes one. `SealedArtifact`
already supports carrying both, so this is a `combine` tool when A7 needs it, not
a redesign.

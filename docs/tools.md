# Tools

The layer between the frozen data layer and the arms. This file defines **what
the environment can do**; which arm may do which is a separate table at the
bottom, and deliberately so.

## Why the two are separated

They are currently the same decision. `schema.py` builds the tool list inside
the container check:

```python
if Container.WEIGHTS in allowed:
    tools += [declare_target, synthesize_data, train, evaluate, inspect_failures]
if Container.CODE in allowed:
    tools.append(write_code)
if Container.CONTEXT in allowed:
    tools.append(set_context)
```

So "this tool exists" and "this arm may use it" are one line of code, and a tool
lands in an arm by where it happens to be written. `evaluate` is weights-only
for exactly that reason — nobody decided it, it was typed inside that block.

The arms are supposed to differ **only in which container they may spend on**.
Separating declaration from allocation makes that invariant something you can
read in one table, and eventually test.

## Discipline: a tool is fully implemented or it does not exist

No parameter may be accepted and ignored.

The cost of breaking this is not "a missing feature". `synthesize_data` declares
`description`, `emphasis` (`uniform | tail | operator-focused`) and
`include_reasoning`; **none of the three is ever read**. The agent filled
`emphasis: "operator-focused"` on every call of one run and wrote increasingly
detailed descriptions — by the third it had derived the structural operators'
permutation rules explicitly — believing it was designing a curriculum. It
received the same rows at three lengths.

So the agent operated a control that was not connected, and the trace recorded
the intent as though it had an effect. Any later analysis of "what curriculum
did the agent choose" would have been reading noise.

---

## Audit: what the nine current tools actually do

| tool | declared | actually read | state |
|---|---|---|---|
| `query_oracle` | `exprs`, `why` | both; `why` is ledger metadata and the schema says so | honest |
| `declare_target` | `role`, `rationale` | `role` stored and traced | **all seven roles train identically** |
| `synthesize_data` | `description`, `n`, `source`, `emphasis`, `include_reasoning` | **only `n` and `source`** | three inert parameters |
| `train` | `dataset_id`, `lora_rank`, `epochs`, `lr` | all four | **cannot request full fine-tuning** |
| `evaluate` | `checkpoint_id`, `n` | both | **checkpoint-only by construction** |
| `inspect_failures` | `checkpoint_id`, `k` | both | re-runs the whole dev split; `k` only truncates output |
| `write_code` | `src`, `check_on` | both | self-evaluation built into the tool |
| `set_context` | `text` | yes | **no self-evaluation at all** |
| `seal` | `entry`, `checkpoint_id`, `summary`, `forced` | all four | honest |

Two of these are structural rather than cosmetic.

### Defect A — the weights arm could not fine-tune fully

`HParams` carries `full_finetune: bool`, but `train`'s schema never exposed it
and `lora_rank` has a minimum of 4. The agent could only ever request LoRA.

That matters because the published weights-arm ceiling — 0.498 at 10% of the
table seen — was measured with **full** fine-tuning, and at matched learning
rate full beats LoRA (0.797 against 0.715 on binary at 4000 steps). The
reference line was one the arm could not reach.

**Settled: all training is full fine-tuning.** The ceiling and the arm now use
the same method, so the reference is valid without remeasuring. LoRA becomes a
deferred question — see the bottom of this file.

### Defect B — `evaluate` is checkpoint-only, so ungating it is not enough

```python
def _t_evaluate(self, checkpoint_id, n):
    path = self.checkpoints.get(checkpoint_id)
    if not path:
        return {"error": f"no checkpoint {checkpoint_id!r}"}
```

A2 holds a string; A4 holds source code. Neither has a checkpoint id to pass, so
moving this tool out of the container block would hand them a tool that returns
an error. The tool was written to evaluate *an adapter*, not *what I made*.

**The capability already exists one layer up.** `seal.evaluate()` takes an
`answer_fn` and does not care what produced it; each arm runner supplies one:

```python
a2   Student(base, context=...).answer(exprs)
a4   run_solver(program, exprs)
a6   Student(base, adapter_path=...).answer(exprs)
```

So the fix is to factor that into one function and use it from both layers:

```python
def answer_with(artifact, base_model, ledger, exprs) -> list[str]:
    if artifact.entry == "program":
        return run_solver(artifact.program, exprs, ledger=ledger).answers
    student = Student(base_model, adapter_path=artifact.adapter_path,
                      context=artifact.context)
    return student.answer(exprs, ledger=ledger).answers
```

- arm runners call it on the **sealed** artifact
- the tool layer calls it on a **provisional** artifact built from current state

The second benefit is the one that matters more than the first: **dev-time and
test-time scoring become the same code path**. Two separate implementations
could drift — different answer cleaning, different caching — and the agent would
then be steering on a number produced differently from the one it is finally
graded by. That class of bug becomes impossible rather than unlikely.

Consequences to expect: A2 starts consuming GPU seconds, since evaluating a
prompt means running the student with it. That is real, not an artifact.

---

## The v1 tool set

Minimal on purpose. Everything here is implemented; anything not needed yet is
absent rather than stubbed.

### Acquisition

```
query(exprs: list[str], why: str) -> {results, malformed, queries_so_far}
```
The only way to learn semantics. Malformed expressions are billed like any
other — probing the grammar for free would make the public syntax spec worth
less than it is meant to be. `why` is recorded, not acted on.

### Building a training set

```
build_dataset(source: "queries"|"demos"|"mixture", n: int) -> {dataset_id, size, drawn_from}
```
Replaces `synthesize_data` with its three inert parameters removed. It repeats
purchased rows rather than inventing any: the agent cannot manufacture labels it
has not bought, and pretending otherwise would be leakage. `n` therefore acts as
an epoch count in disguise, which the tool description should say plainly.

Curriculum control (`filter_data`, emphasis by operator or by tail coverage) is
**not** in v1. It goes in when it is implemented, not before — see deferred.

### Producing an artifact

```
set_context(text: str)            -> {artifact_id, chars}
write_code(src: str)              -> {artifact_id, bytes}
train(dataset_id: str, epochs: int, lr: float) -> {artifact_id, examples, final_loss}
```

All three return an `artifact_id`, which is what makes the evaluation tools
uniform. `train` is always a full fine-tune; `lora_rank` is gone.

`write_code`'s `check_on` is removed — self-assessment belongs to `evaluate`,
not folded into the production tool. One job per tool.

### Self-assessment

```
evaluate(artifact_id: str, n: int) -> {dev_accuracy, n}
inspect(artifact_id: str, k: int)  -> {n_inspected, n_wrong, examples}
```

Both accept **any** artifact and route through `answer_with`. `inspect`
evaluates `k` items rather than the whole dev split and truncating.

Whether every arm gets these is an allocation question, below — but note that
self-assessment is not a property of a container. Someone writing a prompt can
try it, someone writing code can run it, someone training a model can hold out
a validation set. Issuing it to one arm measures the harness, not the container.

### Declaration

```
declare_target(role: Role, rationale: str) -> {ok, role}
```

Kept, and its description must say plainly that it is **declarative**: all seven
roles produce identical training. "The agent chose `world_model`" is a fact
about how it described itself. Making the roles operationally distinct is a
separate question.

### Termination

```
seal(entry: "program"|"model", artifact_id: str, summary: str) -> {sealed, digest, sizes}
```

Ends the prepare phase. A forced seal by the harness must always succeed: an
agent whose training all failed still owes a comparable data point, and the base
student with nothing attached is a legitimate — and very bad — artifact.

---

## Allocation

Declared here, not inside the tool definitions.

| | `query` | `build_dataset` | `set_context` | `write_code` | `train` | `evaluate` | `inspect` | `declare_target` | `seal` |
|---|---|---|---|---|---|---|---|---|---|
| **A2** context | ✓ | — | ✓ | — | — | ? | ? | — | ✓ |
| **A4** code | ✓ | — | — | ✓ | — | ? | ? | — | ✓ |
| **A6** weights | ✓ | ✓ | — | — | ✓ | ✓ | ✓ | ✓ | ✓ |
| **A7** free choice | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

`?` marks the live allocation question (issue #13): whether A2 and A4 get
self-assessment. Recommendation is yes, on the argument above — but it is a
decision about the protocol, so it is marked rather than assumed.

A0′ has no agent loop and takes no tools: the harness buys evidence and the
frontier answers directly.

**A7 is one row, not a new module.** Under this structure the free-choice arm is
an allocation, which is most of why the separation is worth doing.

---

## Deferred

**LoRA.** All training is full fine-tuning for now, to keep one fewer variable
between the arm and the ceiling it is measured against. What LoRA would change:
at matched learning rate it scored 0.715 against full's 0.797 on binary, and
rank had no measurable effect between 4 and 128. Revisit if the question becomes
"how cheaply can the capability be installed" rather than "can it be installed".

**Curriculum control.** `filter_data` and a working `emphasis` are what H3's
"the biggest lever is data curation" needs in order to be testable at all. Not
in v1 because a tool that is declared and inert is worse than one that is
absent.

**Operationally distinct roles.** Making `declare_target`'s seven roles produce
different training — a weighted loss for proposal, scores rather than answers
for value, a search loop for world model — is a larger change and belongs after
the arms are comparable.

# Tools

The layer between the frozen data layer and the agent. This file defines **what
the environment can do**, and **who may do it when** — the availability matrix
is a separate table, deliberately so.

Protocol v2 collapses v1's explore/prepare/test phases into two:

* **practice** — the agent works a *visible* validation set. It may spend
  queries (capped at `Q`), submit validation answers up to `submit_cap` times
  for an aggregate score, and (train arm only) build data and train a student.
* **final** — the held-out test is revealed. Every practice tool is gone; the
  agent commits its answers exactly once with `final_answer` and the run ends.

Two arms differ in **one** thing: the `train` arm has a trainable Qwen3-1.7B
student (and its three tools); the `no_train` arm has only the frontier model.
Everything else — the phases, the caps, the feedback discipline — is identical,
so the arms are compared at equal stop conditions.

Declared in `src/glyph/v2/mcp.py` (`_TOOL_SPECS`, the SDK schema), allocated in
`src/glyph/v2/session.py` (`_MATRIX`, the `(arm, phase)` gate), implemented in
`src/glyph/v2/tools.py` (`t_*` handlers).

## How the tools reach the agent

The agent runs under the **Claude Agent SDK** harness, so it also has the CLI's
own `Bash` and file tools — but the whole CLI is sealed inside a `bwrap`
sandbox that hides the source, GPUs, and credentials, and all model traffic is
routed through a host-side metering gateway (model-pinned, per-request billed).
The eight Glyph tools are registered as **in-process MCP tools** (`create_sdk_mcp_server`),
so the agent sees them as `mcp__glyph__<name>`. Each is a thin async shim around
the matching `t_<name>(session, **args)` handler, which does all the state work
(charging, `(arm, phase)` gating, counters) and emits a trace event.

## Discipline: a tool is fully implemented or it does not exist

No parameter may be accepted and ignored. In v1, `synthesize_data` declared
`emphasis`/`include_reasoning` and read none of them; the agent filled them on
every call believing it was shaping a curriculum, and received identical rows.
A control that is not connected but records intent as though it had effect is
worse than a missing feature. Every v2 handler reads every argument it declares.

## Discipline: declaration is separate from allocation

"This tool exists" (`_TOOL_SPECS` in `mcp.py`) and "this arm may use it in this
phase" (`_MATRIX` in `session.py`) are two tables, not one line of code. The
arms are supposed to differ **only** in whether the student exists, and phases
only in what is exposed; keeping allocation in its own table makes that
invariant readable and a pure function testable.

```python
tool_available_for(arm, phase, name) -> bool   # (arm, phase) in _MATRIX[name]
```

---

## The v2 tool set

Eight tools. Everything here is implemented; anything not needed is absent
rather than stubbed.

### Practice — learning and self-assessment (both arms)

```
query(exprs: list[str], why: str)
    -> {results:[{expr, out} | {expr, refused} | {expr, error}], q_used, <remaining>}
```
The only way to learn semantics. **Query oracle**: an expression that would read
a validation- or test-set item is *refused* (`query_violation(expr, policy)`)
and **not charged**; a malformed expression **is** charged (charge-first, so
probing the grammar for free cannot cheapen the public syntax spec); a query
past the cap returns `q_exhausted`. `why` is recorded, not acted on. Answered
queries are logged to `task/queries.jsonl`.

```
submit(path: str) -> {submission, overall, by_depth, <remaining>}
```
Score an answer file against the **visible validation set**. Feedback is
**aggregate only** — `overall` and a `by_depth` breakdown, never per-item and
never per-split. An *illegal* file (wrong ids, wrong answer type, path escape)
is rejected and **does not consume** a submission. A legal submit counts toward
`submit_cap`; reaching the cap flags the run to switch to the final phase.

```
check_answers(path: str, set: "validation"|"test") -> {ok, violations, examples, <remaining>}
```
A free dry-run legality check — no scoring, no counter touched. `set="validation"`
works in either phase; `set="test"` only in the final phase. This is how the
agent confirms an answer file is well-formed before spending a `submit` or its
one `final_answer`.

```
finish_practice(reason: str) -> {ok, <remaining>}
```
Voluntarily end practice before the caps are hit; flags the switch to final.

### Practice — the student (train arm only)

```
build_dataset(path: str)                              -> {dataset_id, ..., <remaining>}
train(dataset_id: str, epochs: int, lr: float)        -> {checkpoint, ..., <remaining>}
```
`build_dataset` assembles a student training set from **purchased queries** — the
agent cannot manufacture labels it has not bought. `train` fine-tunes the
Qwen3-1.7B student; GPU time is metered into the ledger like any other spend, so
training competes with querying under one budget line.

### Both phases — the student (train arm only)

```
student_infer(checkpoint: str, input_path: str, output_path: str, prefix_path: str|None)
    -> {..., <remaining>}
```
Run a trained student checkpoint over an input file. Available in **practice**
(to self-assess a checkpoint on validation) and in **final** (to produce the
held-out-test answers the agent then commits).

### Final — termination (both arms)

```
final_answer(path: str) -> {committed: true, digest, <remaining>}
```
Commit the held-out-test answers and end the run. An *illegal* file is **not
committed** and may be corrected and re-submitted; a legal one is committed once
and is the thing scored. The held-out test keeps its `iid`/`comp`/`depth`
splits, but the agent never sees a split-level or per-item test score — only the
final report does.

Every handler echoes a `<remaining>` block (queries left, submissions left,
budget left) so the agent can pace itself without a separate status call.

---

## Availability matrix

Declared in `session.py._MATRIX`, not inside the tool definitions.

| | no_train · practice | no_train · final | train · practice | train · final |
|---|---|---|---|---|
| `query`          | ✓ | — | ✓ | — |
| `submit`         | ✓ | — | ✓ | — |
| `check_answers`  | ✓ | ✓ | ✓ | ✓ |
| `finish_practice`| ✓ | — | ✓ | — |
| `build_dataset`  | — | — | ✓ | — |
| `train`          | — | — | ✓ | — |
| `student_infer`  | — | — | ✓ | ✓ |
| `final_answer`   | — | ✓ | — | ✓ |

A call outside its cell returns a gate error naming the phase, and is not
charged.

---

## What the validation set is, and what it is not

The visible validation set is drawn from the **iid** split only (generated after
the held-out test so the two never share items). Practice feedback is
intentionally coarse — overall + by-depth aggregates — so the agent gets a real
but blunt signal. It **does not estimate test**: an agent that probes narrowly
*should* get a validation number that flatters it, and whether it can infer the
shape of the held-out distribution is part of what the experiment asks. The
final report carries validation-vs-test so the gap arrives with the trajectory
that explains it, but that comparison is never handed to the agent mid-run.

---

## Deferred

**`query_ood_policy` = open (D1).** Whether a practice `query` may probe `comp`
or `depth` items (not just `iid`) is a knob (`QUERY_OOD_POLICIES`), currently set
to refuse anything outside the validation-eligible region. Opening it is a
deferred comparison, not a redesign.

**The Q sweep.** `Q` (the query cap) is the coverage knob that positions an arm
on the effort axis; sweeping it to compare arms at matched coverage is planned,
not yet run.

# Tools

The layer between the frozen data layer and the agent. This file defines **what
the environment can do**, and **who may do it when** — the availability matrix
is a separate table, deliberately so.

Protocol v2 collapses v1's explore/prepare/test phases into two:

* **practice** — the agent works a *visible* validation set. It may spend
  queries (capped at `Q`), submit validation answers up to `submit_cap` times
  for an aggregate score, and (train arm only) build data and train a student.
* **final** — the held-out test is revealed. Every practice tool is gone; the
  agent commits its answers exactly once with `submit_final_answer` and the run ends.

Two arms differ in **one** thing: the `train` arm has a trainable Qwen3-0.6B
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
submit_validation_answer(path: str) -> {submission, overall, by_depth, <remaining>}
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
works in either phase; `set="test"` only in the final phase, and checks every
row of `final/test.jsonl`, probes included — the same rows `submit_final_answer`
requires. This is how the
agent confirms an answer file is well-formed before spending a `submit_validation_answer` or its
one `submit_final_answer`.

```
finish_practice(reason: str) -> {ok, <remaining>}
```
Voluntarily end practice before the caps are hit; flags the switch to final.

### Practice — the student (train arm only)

```
build_dataset(path: str)                              -> {dataset_id, ..., <remaining>}
train_model(dataset_id: str, epochs: int, lr: float, student_id: str) -> {checkpoint_id, student_id, continued_from, ..., <remaining>}
```
`build_dataset` turns an agent-written JSONL of `{"expr", "answer"}` rows into a
training set. Rows that do not parse are dropped; every kept row is labelled by
provenance (`purchased`, `demo`, `validation` or `other`, checked against
`task/queries.jsonl`, the demos and the validation set). Provenance is recorded,
not enforced: an `other` row trains like any other. `train_model` fine-tunes a
Qwen3-0.6B student (`--student-model`); GPU time is metered into the ledger like any other spend, so
training competes with querying under one budget line. `student_id` selects the
lineage and is minted by the AGENT in a fixed format — exactly 8 lowercase hex
characters (`^[0-9a-f]{8}$`, e.g. `a1b2c3d4`); anything else is rejected with
a structured error. A new id starts a fresh student from the base model, an
existing id continues training that student from its latest checkpoint, and
students coexist. Each `train_model` call returns a `checkpoint_id` in the fixed
format `<student_id>_ck_xxxxxxxx` (owner prefix + 8 hex hashed from the
lineage position, so names are stable against other students' interleaving), which must be
echoed back verbatim. `infer_model`'s `checkpoint` accepts `"base"`, a
`<student_id>_ck_xxxxxxxx` checkpoint_id, or an 8-hex student_id (that student's latest
checkpoint) — the three forms are disjoint by construction, so resolution is
unambiguous; malformed or unknown references return a structured error.

### Both phases — the student (train arm only)

```
infer_model(checkpoint: str, input_path: str, output_path: str, prefix_path: str|None)
    -> {..., <remaining>}
```
Run a trained student checkpoint over an input file. Available in **practice**
(to self-assess a checkpoint on validation) and in **final** (to produce the
held-out-test answers the agent then commits).

### Final — termination (both arms)

```
submit_final_answer(path: str) -> {committed: true, digest, <remaining>}
```
Commit the held-out-test answers and end the run. An *illegal* file is **not
committed** and may be corrected and re-submitted; a legal one is committed once
and is the thing scored. Legality covers every row of `final/test.jsonl`,
including the `probe_*` rows (the per-op probe set) — a file missing probe
answers is illegal. Probes are scored separately, never in `overall`. The held-out test keeps its `iid`/`comp`/`depth`
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
| `submit_validation_answer` | ✓ | — | ✓ | — |
| `check_answers`  | ✓ | ✓ | ✓ | ✓ |
| `finish_practice`| ✓ | — | ✓ | — |
| `build_dataset`  | — | — | ✓ | — |
| `train_model`     | — | — | ✓ | — |
| `infer_model`     | — | — | ✓ | ✓ |
| `submit_final_answer` | — | ✓ | — | ✓ |

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

## The free protocol

A second protocol condition (`--protocol free`; spec
`docs/superpowers/specs/2026-10-15-free-protocol-design.md`). The three
student tools are not registered at all, so the model never sees them; the
agent instead has a root shell in an open Ubuntu container
(`src/glyph/v2/free_sandbox.py`) with conda, network, web search and
subagents. The `train` arm's container has one GPU, the `no_train` arm's none.
A third arm, `silent`, has a GPU like `train`, but its system prompt and
README have no Environment section, so the agent learns what the machine
offers only by looking. `tools/free_env_probes.py` reports what a run
checked (GPU, CPU, memory, disk, OS, Python environment, network), when,
and when a shell command first showed it a GPU.

| | practice | final |
|---|---|---|
| `query` | ✓ | — |
| `submit_validation_answer` | ✓ | — |
| `finish_practice` | ✓ | — |
| `submit_final_answer` | — | ✓ |

The same in every arm (`session.py._FREE_MATRIX`). What changes from v2:

- No `check_answers`. With no submission cap, `submit_validation_answer`
  reports the same violations at no cost, and `submit_final_answer` rejects an
  illegal file without committing it. If the final phase ends without a
  commit, the run is scored on the most recently written legal answer file
  under the work root, outside `task/`, written since the final began
  (`free_harness.latest_legal_answer_file`; `final_commit` =
  `auto_latest_legal`).

- `query` also refuses any expression whose evaluation looks up a
  binary-operator entry that a validation item needs
  (`GlyphInstance.reserved_binary`; refusal reason `needs_validation_entry`,
  not counted). The check evaluates the expression, so a fold that reaches a
  reserved entry is refused as well. Without it an agent can buy the few
  thousand entries validation needs and score 1.0 on validation while learning
  nothing about the test. Test entries are not reserved: refusing them would
  reveal which entries the hidden test needs. Unary entries are not reserved,
  so the unary tables can still be bought whole.
- Each container runs pinned to its own block of `--cpus` CPUs inside one
  NUMA node (`free_sandbox.CpuLease`, a flock under
  `<scratch-root>/cpusets/`; `taskset`, because the user manager has no cpuset
  controller). `nproc`, `os.sched_getaffinity`, OpenBLAS and joblib then report
  the quota instead of the host's 160 CPUs. `/proc/cpuinfo`, Python's
  `os.cpu_count()` and `free` still report the host. `--no-cpu-pin` turns
  pinning off.
- `--no-bare-atomics` (option; off by default) makes `query` answer a bare
  atomic application, `u*(value)` or `b*(value, value)`, with
  `{"error": "malformed"}`, as the parser did before such applications
  existed; unlike other malformed input it is not counted. Table cells are then reachable only through
  structural expressions, for example a map `s0(u0, [x, y])` or a two-element
  fold `s1([x, y], b0)`, so an agent that wants cell-level data has to decode
  the skeleton first and read cells back out of structural results. The README
  says nothing about it and `syntax.md` drops the lines on bare applications,
  so the agent sees the language as it was before 2026-09-30, when the grammar
  had no bare atomics; the malformed error is the only signal. The test set still
  holds bare `probe_` rows, which the agent then meets without a description.
  Run names carry a `_nobare` suffix; the report records `bare_atomics` and
  `refused_bare`.
- Each binary probe operator in the report also carries `swapped.acc`: the
  100 bare rows `b(x, y)` scored against `b(y, x)`. An agent that met the
  operator only inside folds cannot learn which order the bare application
  uses, so under `--no-bare-atomics` the higher of `overall` and `swapped.acc`
  is the orientation-free score.
- No cap on queries, validation submissions or turns, and no idle rule. Each
  phase is bounded by wall-clock only (`T_p`, `T_f`); every result's
  remaining block is `{seconds_remaining}`, and `task/timer.sh` prints the
  same.
- At the end of each phase the harness interrupts the agent, kills every
  process in the container except the CLI, and lets the turn drain for 60 s.
- The gateway listens on a per-run TCP port and refuses any request without
  the run's path token (the containers share the host network).
- The text the agent sees (`src/glyph/v2/free_prompts.py`) states facts only;
  outside its two rules it contains none of the words train, model,
  download, student, suggest or consider, which a test enforces.

`tools/run_free_sweep.py` runs the sweep in waves, one per `T_p`.

---

## Deferred

**`query_ood_policy` = open (D1).** Whether a practice `query` may probe `comp`
or `depth` items (not just `iid`) is a knob (`QUERY_OOD_POLICIES`), currently set
to refuse anything outside the validation-eligible region. Opening it is a
deferred comparison, not a redesign.

**The Q sweep.** `Q` (the query cap) is the coverage knob that positions an arm
on the effort axis; sweeping it to compare arms at matched coverage is planned,
not yet run.

# Glyph protocol v2 — practice on a visible validation set, answer a held-out test once

- **Status:** draft for review
- **Date:** 2026-09-14
- **Replaces:** the prepare → seal protocol (arms `a0p` / `a2` / `a4` / `a6`), the
  hand-written agent loop in `agent/orchestrator.py`, and the v1 tool layer in
  `docs/tools.md`. The data layer (`glyph/data/`) is extended, not replaced.
- **Decided in:** a design session on 2026-09-14. Two throwaway measurements back the
  parameter choices; they are summarised in §12 and were not committed.

---

## 1. Why the protocol changes

The v1 protocol gave the agent a prepare phase, had it seal one artifact, and scored
that artifact on 10⁴ items the agent never saw. Three things about it were weak:

1. **The agent's feedback did not resemble the exam.** Dev was carved from its own
   purchases, 91.5% single-level probes against a test set that is 12.5% single-level;
   one run reached dev 0.400 against test 0.035 (#14).
2. **Three containers produced three incomparable artifact shapes**, and the arm
   comparison never got past n = 200 with invalidated instances (#23).
3. **The hand-written loop fell short of the jobs it was written for**:
   - tool results were truncated at 8,000 characters;
   - the agent was told its budget bought ~1.67 M queries;
   - prompt caching was never set;
   - its replay cache could not hit past the first turn.

v2 keeps what worked — the hidden interpreter, the skeleton/table split, π, the
ceilings — and changes the protocol around it. It borrows the standard
train/validation/test shape. The agent buys labels with queries, practises against a
visible validation set with a small number of aggregate-score submissions, and then
answers a held-out test exactly once with no further access to the oracle.

The claim this supports: **given the same interaction budget, does an agent that can
train a small student generalise better to held-out items (in distribution, to unseen
operator compositions, and to greater depth) than the same agent without one?**

## 2. The protocol in one page

```
PRACTICE PHASE      validation questions visible; held-out test not visible
  query            ≤ Q expressions; refused expressions are not charged
  submit(path)     ≤ 20 legal submissions to validation → aggregate scores only
  [train arm]      build_dataset · train · student_infer
  ends on the first of: 20th legal submission · finish_practice · T_p turns · USD safety line

FINAL PHASE         held-out test questions revealed
  query, submit, finish_practice, build_dataset, train     removed
  [train arm]      student_infer (weights frozen)
  final_answer(path)   exactly one legal commit, no score returned
  ends on: legal final_answer · T_f turns · USD safety line (→ auto-commit rule, §4.4)

SCORING             committed answers vs held-out test: iid / comp / depth / tail, with ceilings
```

Running out of Q does **not** end practice: `query` refuses and the run continues.

## 3. Arms

There are two arms. They run on the same instances, with the same harness, sandbox,
caps, system prompt and built-in tools.

| arm | what the agent has beyond the common set |
|---|---|
| `no_train` | nothing: the frontier agent with Bash and file tools in the sandbox |
| `train` | a Qwen3-1.7B student it can fine-tune and run: `build_dataset`, `train`, `student_infer` |

- The `train` arm can do everything the `no_train` arm can, and may ignore the student.
  The comparison is therefore about whether training pays under equal caps, since
  training spends turns and cost. Reports state it that way.
- The old context arm (A2) is subsumed: `student_infer` accepts a prompt prefix.
- The old code arm (A4) is subsumed: both arms can write and run programs in the sandbox.
- The old A0′ is subsumed by `no_train`, which is A0′ with self-chosen evidence and
  validation feedback. That is the control #20 listed as missing.

## 4. Phases and caps

### 4.1 Practice phase

- The agent starts with:
  - the system prompt (§8.3);
  - an opening message pointing at `/work/task/README.md`;
  - read-only files: syntax spec, 30 demos, the validation questions, and a live copy of
    its query log.
- It may query, run code, write files, submit to validation and, in the `train` arm,
  build datasets, train and run the student.

### 4.2 Transition to the final phase

Practice ends on the **first** of:

1. the 20th legal `submit`;
2. `finish_practice(reason)`, which is irreversible;
3. the practice turn cap `T_p`;
4. the USD safety line (§11).

On a trigger the harness:
1. calls `client.interrupt()`, so no further practice tool call runs in the same turn;
2. sets the phase to `final`;
3. writes `test.jsonl` into `/work/task/final/`;
4. sends the final-phase opening message as a new user turn in the same session.

### 4.3 Final phase

- `query`, `submit`, `finish_practice`, `build_dataset` and `train` are unavailable.
- In the `train` arm, `student_infer` remains available and every checkpoint is frozen.
- Both arms may use `check_answers` and must call `final_answer(path)` once with a legal
  file.
- The receipt reports the commit. It does not report a score. The session then ends.

### 4.4 Final-phase timeout

If `T_f` or the USD line is reached before a legal `final_answer`:

- if the agent ever ran `check_answers(path, "test")` and that check passed, the
  harness commits **the last path that passed** in its current state;
- otherwise the run scores 0 on the held-out test and the report flags
  `final_commit = "none"`.

A run always yields a number rather than a hole in the grid, following the v1
forced-seal rule.

### 4.5 Turn counting

- A turn is an assistant message that contains at least one tool call.
- **Every such turn counts** toward `T_p` (practice) or `T_f` (final).
- Answers are submitted as files, so writing them costs a few tokens and needs no
  exemption.
- The harness counts turns from the SDK message stream.

**Idle turns.** If the agent ends its turn without a tool call:
- The harness sends a fixed continuation message, one of:
  - practice: `"Continue, or call finish_practice if you are done practising."`
  - final: `"Continue, or call final_answer."`
- The idle turn counts toward the phase's cap.
- **Three consecutive idle turns end the phase:**
  - in practice this acts as `finish_practice` with reason `idle`;
  - in the final phase it acts as the §4.4 timeout.

### 4.6 What the agent is told

The README and system prompt state every cap: Q, 20 submissions, `T_p`, `T_f`, and
the fact that a final phase follows with no oracle. Every `glyph` tool result also
carries the remaining Q, submissions, and turns for the current phase.

## 5. Data layer changes (`glyph/data/`)

The boundary rule stands: nothing in `data/` imports from outside it
(`tests/test_data_boundary.py`). No experiment-fixed constant changes, so π, the
ceilings and #19/#20 remain valid.

### 5.1 Validation split

- **New config field:** `n_val` (default 5000).
- **Sampling:** the same spec as `iid`:
  - budget `demo_max_depth`;
  - `forbid = held_pairs`;
  - `min_depth = 1`;
  - deduplicated against demos and all test items.
- **Order:** drawn **after** all existing splits, on the same `np.random.Generator`
  stream. For every existing seed, `demos`, `test` and `held_pairs` stay bit-identical;
  a regression test hashes `pi_mid/1001`.
- **Failure:** `GenerationFailed("val", ...)` when the split cannot be filled, with the
  existing diagnosis.
- **Access:** `inst.val` holds `TestItem`s with `split = "val"`. They are not part of
  `inst.test`.

### 5.2 Query policy

A pure function in `data/` (e.g. `instance.query_violation(expr_src, policy)`) returns
a reason or `None`.

- **Canonicalisation:** `parse` then `render`, so spacing and formatting variants
  match.
- **Refusal reasons:**

| reason | policy |
|---|---|
| `is_validation_item` | always |
| `is_test_item` | always |
| `contains_held_out_pair` | `strict` only |
| `deeper_than_validation` (depth > `demo_max_depth`) | `strict` only |

- **Policy values:** `query_ood_policy ∈ {"strict", "open"}`; the main setting is
  `strict`. `open` is deferred comparison D1 (§13).
- **Malformed expressions** are not refusals. They fail at parse/check and are charged,
  as in v1, because probing the grammar should not be free.

### 5.3 Per-run coverage covariates

Both use the harness's authoritative query log:

- `val_lookup_solvable`: the fraction of validation items whose every needed table entry
  appears in queried results (including items needing none). It is the score a perfect
  retriever with a perfect skeleton could get on validation.
- `test_covered`: the same quantity on the held-out test.

Both reuse `needs_u` / `needs_b`. Those record entries *touched*, so both are slightly
conservative (#4 residue).

## 6. Tools

### 6.1 Availability

All custom tools come from one in-process MCP server named `glyph`, so the model sees
`mcp__glyph__<name>`.

| tool | practice · `train` | practice · `no_train` | final · `train` | final · `no_train` |
|---|:-:|:-:|:-:|:-:|
| `query` | ✓ | ✓ | | |
| `submit` | ✓ | ✓ | | |
| `check_answers` | ✓ | ✓ | ✓ | ✓ |
| `finish_practice` | ✓ | ✓ | | |
| `build_dataset` | ✓ | | | |
| `train` | ✓ | | | |
| `student_infer` | ✓ | | ✓ | |
| `final_answer` | | | ✓ | ✓ |
| built-in `Bash`, `Read`, `Write`, `Edit`, `Glob`, `Grep` | ✓ | ✓ | ✓ | ✓ |

- **Built-ins that are off:** `WebFetch`, `WebSearch`, `Task` (subagents), `TodoWrite`,
  `NotebookEdit`, `Skill`.
- **Enforcement:** the handler checks the phase and arm on every call and returns a
  refusal as data. This is authoritative. Whether the unavailable tools can also be
  removed from the request's tool list is pre-implementation spike S3 (§15).
- **Test:** the table above is asserted as a whole by one test, as `tests/test_agent.py`
  does for v1.

### 6.2 Legal answer files

`submit`, `check_answers` and `final_answer` share one checker. A file is **legal**
when all of the following hold:

1. It is UTF-8 JSONL. Each line is an object `{"id": <str>, "answer": <str>}` with no
   other keys.
2. Every id of the target set appears **exactly once**, and there are no unknown ids.
3. Every `answer` is well-formed under `grammar.parse_value` / list rendering: a single
   value `v_x_y_z`, or a non-empty list `[v_..., v_...]`.
4. The answer's type (value vs list) equals the expression's `result_type`. That type
   follows from the outermost operator's public signature, so the check leaks nothing.
   **List length is never checked**, because it depends on hidden semantics.
5. The path resolves inside `/work` and is not under `/work/task`.

**An illegal file never consumes a submission and never commits.** The tool returns the
list of violations (counts plus up to 5 examples with line numbers). All 20 submissions
are therefore legal and complete.

### 6.3 Tool specifications

**`query(exprs: list[str], why: str)`** — practice only
- Accepts at most 256 expressions per call, processed in order. Each result is one of:
  - `{"expr", "out"}`
  - `{"expr", "refused": <reason>}` — not charged
  - `{"expr", "error": "malformed"}` — charged
  - `{"expr", "error": "q_exhausted"}` — not charged, once Q hits 0 mid-batch
- Returns the per-expression results, `q_used`, `q_remaining`, `submissions_remaining`
  and `turns_remaining`.
- Successful results are appended to `/work/task/queries.jsonl`, which is read-only to
  the agent, and to the authoritative host-side log.
- `why` is recorded in the trace and does not affect execution.

**`submit(path: str)`** — practice only
- Legality check against the validation set (§6.2). An illegal file returns the
  violations and is not counted.
- A legal file returns:
  - `submission` (1–20)
  - `submissions_remaining`
  - `overall`
  - `by_depth: {"1": acc, "2": acc}`
  - `q_remaining`, `turns_remaining`
- Accuracy is exact match after `strip()`, as in held-out scoring. **No per-item
  information is ever returned.**
- The file is copied to the host run directory, and harness-side per-item scores are
  kept for reporting only.
- The 20th legal submission triggers the phase switch (§4.2).

**`check_answers(path: str, set: "validation" | "test")`**
- Runs the §6.2 checker and returns legality and violations. Never touches correctness.
- Unlimited calls.
- `set = "test"` is only valid in the final phase.
- In the final phase, a passing check records the path for the timeout rule (§4.4).

**`finish_practice(reason: str)`** — practice only
- Ends practice immediately (§4.2). The reason is recorded.

**`build_dataset(path: str)`** — `train` arm, practice only
- Reads agent-authored JSONL `{"expr", "answer"}`.
- Checks that every `expr` parses and passes `check`, and every `answer` is well-formed.
  **Labels are not verified**: synthesised, filtered or pseudo-labelled rows are allowed.
- Returns `dataset_id`, the row count, and a provenance breakdown: rows equal to a
  purchased query result, rows from demos, rows whose `expr` is a validation question,
  and other rows.
- The file is copied to the host run directory.

**`train(dataset_id: str, epochs: int, lr: float)`** — `train` arm, practice only
- Full fine-tune of Qwen3-1.7B, using the frozen algorithm in `train/sft.py`, on one GPU.
- Wall-clock caps: 30 GPU-minutes per call, 2 GPU-hours cumulative in practice. A call
  that reaches a cap stops, saves its checkpoint, and returns `stopped_at_cap: true`.
- Returns `checkpoint_id`, `final_loss`, `gpu_seconds`, `gpu_seconds_remaining`.
- Checkpoints live host-side and are never visible in the sandbox.

**`student_infer(checkpoint: str, input_path: str, output_path: str, prefix_path: str | None)`** — `train` arm, both phases
- `checkpoint` is a `ckpt_*` id or `"base"`.
- Reads JSONL `{"id", "expr"}` from `input_path` and writes JSONL `{"id", "answer"}` to
  `output_path` with the vLLM `Student` (greedy, prefix caching on, `max_new_tokens = 96`).
- `prefix_path` optionally supplies a prompt prefix.
- Returns the row count, the truncated count and `gpu_seconds`.
- Outputs are copied host-side for the provenance measure (§10.3).
- In the final phase checkpoints are frozen; no `train` exists.

**`final_answer(path: str)`** — final phase only
- Legality check against the held-out test (§6.2). An illegal file returns the
  violations and does not commit.
- A legal file returns `{"committed": true, "digest": ...}`. No score. The session ends.

## 7. Workspace

```
/work/                         read-write, the agent's cwd
├── task/                      read-only bind mount, updated host-side
│   ├── README.md              rules, caps, tools, file formats
│   ├── syntax.md              the public syntax spec
│   ├── demos.jsonl            30 × {"expr", "answer"}
│   ├── validation.jsonl       5000 × {"id": "val_00000", "expr"}   — no answers
│   ├── queries.jsonl          appended by the harness: {"expr", "out"}
│   └── final/                 empty in practice; test.jsonl appears at the phase switch
└── …                          the agent's scripts, notes, answer files
```

`test.jsonl` is shuffled and uses neutral ids (`test_00000`, …). Neither the ids nor the
order reveal the split.

## 8. Harness

### 8.1 Components

```
host (sees repo, instance, GPUs, credentials)
┌─ harness process (Python) ───────────────────────────────────────────┐
│  GlyphInstance · Ledger · TraceWriter · counters · phase             │
│  ClaudeSDKClient + in-process MCP server "glyph"                     │
│  model gateway: holds GCP credentials, forwards the pinned model      │
│  student: sft.train / infer.Student on one GPU                        │
└──────┬ stdio (SDK control protocol) ─────────┬ unix socket ───────────┘
       ▼                                        ▼
┌─ bwrap sandbox (the SDK's cli_path points at the wrapper) ───────────┐
│  Claude Code CLI · Bash · Read/Write/Edit/Glob/Grep                   │
│  sees: /usr (ro), the CLI binary (ro), /work (rw), /work/task (ro),   │
│        /agent_home (rw, HOME)                                         │
│  does not see: /home, the Glyph repo, ~/data, gcloud credentials,     │
│        /dev/nvidia*                                                   │
│  network: own namespace; only 127.0.0.1:8787 bridged to the gateway   │
└───────────────────────────────────────────────────────────────────────┘
```

MCP tools are callable only by the model through the CLI, never from Bash. The Q cap,
the submission cap and the phase rules cannot be bypassed from code running in the
sandbox.

### 8.2 SDK configuration

Pinned to `claude-agent-sdk==0.2.152` and its bundled CLI. Options:

| option | value |
|---|---|
| `cli_path` | the sandbox wrapper (§8.4) |
| `cwd` | `/work` |
| `model` | `claude-opus-4-8` (pending spike S1 on the current Vertex project) |
| `effort` / `thinking` | `high` / adaptive, fixed across arms, as in v1 |
| `system_prompt` | the Glyph system prompt (§8.3), replacing Claude Code's |
| `setting_sources` | `[]`, set explicitly (the SDK default `None` leaves it to the CLI) |
| `tools` | `["Bash", "Read", "Write", "Edit", "Glob", "Grep"]` |
| `allowed_tools` | the same plus `mcp__glyph__*` |
| `disallowed_tools` | `WebFetch`, `WebSearch`, `Task`, `TodoWrite`, `NotebookEdit`, `Skill` |
| `mcp_servers` | `{"glyph": <in-process server>}` |
| `hooks` | `PreToolUse` (phase-gate message, turn bookkeeping), `PostToolUse` (trace built-in tool I/O) |
| `max_budget_usd` | unset; the gateway enforces the USD line (§9) |

Claude Code's automatic prompt caching and auto-compaction stay on unless spike S5 shows
compaction must be controlled (`DISABLE_AUTO_COMPACT` and
`CLAUDE_AUTOCOMPACT_PCT_OVERRIDE` exist in the pinned CLI). Every compaction boundary is
traced.

### 8.3 System prompt

It carries over the v1 principle from `agent/prompts.py`: it states the protocol, the
caps and the arm's tools, and **does not hint at what the right strategy is**. The two
arms receive the same text except for one paragraph that describes the student tools.
The prompt names `/work/task/README.md` as the full rules.

### 8.4 Sandbox wrapper

The wrapper launches the CLI as follows:

```bash
exec taskset -c "$CPUS" prlimit --as="$MEM_LIMIT" --nproc=512 -- \
  bwrap --unshare-all --die-with-parent --new-session --clearenv \
    --ro-bind /usr /usr --symlink usr/lib64 /lib64 --symlink usr/lib /lib --symlink usr/bin /bin \
    --proc /proc --dev /dev --tmpfs /tmp \
    --ro-bind "$SDK_BUNDLED_CLI" /opt/claude \
    --ro-bind "$BRIDGE_PY" /opt/bridge.py \
    --bind "$RUN/work" /work --ro-bind "$RUN/task" /work/task \
    --bind "$RUN/agent_home" /agent_home \
    --ro-bind "$RUN/gateway.sock" /run/gw.sock \
    --setenv HOME /agent_home \
    --setenv CLAUDE_CODE_USE_VERTEX 1 \
    --setenv ANTHROPIC_VERTEX_BASE_URL http://127.0.0.1:8787 \
    --setenv CLAUDE_CODE_SKIP_VERTEX_AUTH 1 \
    --setenv DISABLE_TELEMETRY 1 --setenv CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC 1 \
    --chdir /work \
    /usr/bin/python3 /opt/bridge.py -- /opt/claude "$@"
```

- **Clean environment.** `--clearenv` removes every inherited variable, including
  `GOOGLE_APPLICATION_CREDENTIALS`; only the variables set above exist inside.
- **Runtime.** The system `python3` (standard library only, no numpy or torch) and
  coreutils. No GPU device nodes. CPU cores are pinned with `taskset`, and memory and
  process count are limited with `prlimit`.
- **Bash timeouts.** A single command's wall clock is bounded by Claude Code's Bash
  timeout.
- **Bridge.** `bridge.py` forwards `127.0.0.1:8787` to `/run/gw.sock`, then execs the CLI.
- **Verified on lumen1** (2026-09-14, bwrap 0.11.2, simplified invocation): inside the
  sandbox `/home` does not exist, `~/.bashrc` is unreadable, `/dev/nvidia0` is absent,
  outbound TCP returns *Network is unreachable*, and `python3` runs.

### 8.5 Model gateway

- Listens on the run's unix socket and holds the GCP credentials.
- Forwards only requests for the pinned model and rejects all others. How the CLI's
  background requests behave under this rule is part of spike S1.
- Records per request: model, input tokens, cache-read tokens, cache-write tokens,
  output tokens, latency, and the full request body (system prompt and tools array).
  **These records are the authoritative model cost**; the SDK's `total_cost_usd` is an
  estimate and is not used.
- Retries 429 and 5xx with backoff. Only successful responses are billed.
- Any request that reaches it is metered, including one issued from Bash, so model use
  cannot escape the ledger.

### 8.6 Trace

One host-side JSONL stream per run, following v1's rule that analysis reads traces and
never parses prose. It records:

- the SDK message stream: assistant text, thinking, `tool_use`, compaction boundaries;
- `PostToolUse` records for built-in tools (Bash command and output, file paths read or
  written);
- every `glyph` tool call with its arguments and full result;
- phase events and trigger reasons;
- a reference to each gateway request record.

Copies of every submitted file, dataset file and `student_infer` output are stored in
the run directory.

### 8.7 Failure handling

- **CLI exit mid-run:** resume the session once from `/agent_home`. If that fails, the
  run is marked `failed` with its reason and reported, never silently dropped (v1
  `worker.py` rule).
- **Grid failures:** an instance with a failed arm is reported as such. Surviving arms
  are not averaged over as if nothing happened.

## 9. Accounting

The ledger records, per run and per phase:

- model cost in USD (from the gateway, §8.5), split into input, cache read, cache write
  and output;
- GPU seconds for `train` and for `student_infer`;
- sandbox CPU seconds;
- Q used, submissions used, turns used.

The USD safety line is checked by the gateway after each response. Crossing it is a
phase trigger (§4.2, §4.4), never an abort.

## 10. Scoring and reporting

### 10.1 Primary

The committed `final_answer` is scored by exact match on the held-out test.

- **Raw scores:** overall, `iid`, `comp`, `depth`.
- **`tail`:** items whose needed entries were not all queried.
- **Ceilings:** skeleton and table ceilings on the scored items, reusing `ceilings()`.
- **`headroom`** per split is **the primary figure**, with raw scores alongside
  (settled in #4).
- **Instance block:** measured π, seed, preset, `uses_binary_tables`.

### 10.2 Secondary

- The 20 validation feedbacks as the agent saw them, and harness-side per-item scores
  for the same submissions.
- The best and last validation scores.
- **The generalisation gap:** last validation score − held-out `iid` score. Both come
  from the same distribution, so the gap measures how much of validation performance
  was fitted rather than generalised.

### 10.3 Covariates

- `q_used`, `submissions_used`, `turns_practice`, `turns_final`, `finished_practice_early`
- `compactions`, `usd_by_kind`, `gpu_seconds`, `sandbox_cpu_seconds`
- `val_lookup_solvable`, `test_covered` (§5.3)
- `train` arm: `final_from_student`, the fraction of committed answers identical to
  any `student_infer` output for the same id
- `final_commit ∈ {agent, auto_checked_path, none}`, `run_status`

### 10.4 Comparison

- Both arms run on the same instances: `pi_mid` and `pi_low`, instance seeds 1001–1005.
- Reports give per-instance paired differences and the paired SE.
- Round 1 uses one run seed per (arm, instance). Run-seed replication for the
  instance/run variance split (#25) follows round 1.

## 11. Parameters

| parameter | value | basis |
|---|---|---|
| Q | 1000 | §12: keeps validation from being lookup-saturated at V = 5000 and leaks < 3 points to the test |
| V (`n_val`) | 5000 | §12: at V ≤ 2000, Q = 1000 makes 83–84% of validation lookup-solvable |
| held-out test | 10,000 (iid 6500 / comp 2300 / depth 1200) | unchanged |
| legal submissions | 20 | design decision |
| `T_p` / `T_f` | 100 / 30 turns | provisional; calibrated by the pilot (§14) |
| `train` cap | 30 GPU-min per call; 2 GPU-h practice total; 1 GPU | prevents one call from dominating a run |
| USD safety line | $300 per run | provisional; calibrated by the pilot |
| `query_ood_policy` | `strict` | main setting; `open` is D1 |
| presets × seeds | `pi_mid`, `pi_low` × 1001–1005 | `pi_high` is retrieval-saturated (§12) |
| model / effort | `claude-opus-4-8` / `high` | as v1; availability on the current Vertex project pending S1 |

For `pi_low` instances whose test set never invokes a binary table (3 of 5 seeds in the
measurement below): keep them, flag `uses_binary_tables = false` in every report, and
log the finding in `open_questions.md`.

## 12. Evidence behind the parameters

These were throwaway scripts on lumen1 (`/tmp/glyph_spike/`), not committed. The buyer is
an upper bound: it knows the true skeleton for free and retrieves perfectly. An entry
counts as revealed only if its input was a literal in the query and its output is
visible in the result.

**Per query:** 3–4 unary entries and at most 1 binary entry.

**Measurement 1 — a visible 10⁴-item test, buyer targets it directly.** Fraction of
items whose needed entries are all bought (median of 5 seeds):

| Q | pi_low | pi_mid | pi_high |
|---:|---:|---:|---:|
| 0 | 0.139 | 0.249 | 0.647 |
| 1000 | 0.373 | 0.468 | 0.872 |
| 2000 | 0.600 | 0.628 | 0.951 |
| 4000 | 1.000 | 0.874 | 1.000 |

This is why the held-out test is not shown during practice.

**Measurement 2 — visible validation of V items, hidden test of 5000, buyer targets
validation.**

| preset | V | Q | validation lookup-solvable | hidden test covered | vs. no queries |
|---|---:|---:|---:|---:|---:|
| pi_mid | 500 | 1000 | 1.000 | 0.250 | +0.002 |
| pi_mid | 2000 | 1000 | 0.826 | 0.259 | +0.011 |
| pi_mid | 5000 | 1000 | 0.593 | 0.260 | +0.012 |
| pi_mid | 5000 | 2000 | 0.810 | 0.299 | +0.051 |
| pi_low | 2000 | 1000 | 0.839 | 0.170 | +0.029 |
| pi_low | 5000 | 1000 | 0.513 | 0.169 | +0.028 |
| pi_low | 5000 | 2000 | 0.861 | 0.288 | +0.147 |

- **The hidden test is protected by compounding, not by disjoint entries.** An item needs
  2.4–5.6 lookups, so partial entry coverage rarely completes an item.
- **The binding risk is on the validation side.** A small validation set becomes
  lookup-solvable, and its feedback then measures retrieval.

## 13. Deferred comparisons and open items

- **D1 — `query_ood_policy = "open"`.** Allow queries with held-out pairs or greater
  depth, and measure whether the agent seeks them out and what that does to `comp` and
  `depth`. Implemented as a config switch from the start.
- **Q sweep** at {500, 2000} with V fixed.
- **Run-seed replication** (#25).
- **H1 wording (#18)** to be restated for v2.
- **`open_questions.md` updates:**
  - rewrite or close #9, #11, #12, #13, #14, #15, #16, #17 and #27 under v2;
  - add the `pi_low` binary-table finding;
  - add D1 and the Q sweep.

## 14. Migration

- **Remove:** `arms/`, `agent/orchestrator.py`, `agent/tools.py`, `agent/schema.py`,
  `agent/prompts.py` (replaced), `worker.py` (replaced by a v2 scheduler with the same
  paired, instance-major rule), and the tests that only cover them.
- **Keep and adapt:**
  - `seal.py`: ceilings, headroom and the scoring functions; `SealedArtifact` and
    `answer_with` go.
  - `budget.py`: USD by kind plus GPU seconds.
  - `trace.py`: the writer; `ResponseCache` goes.
  - `train/sft.py` and `train/infer.py`: unchanged algorithm; caps added at the call site.
  - `sandbox.py`: superseded by the bwrap wrapper and removed.
- **Keep unchanged:** the data layer apart from §5, `scripts/table_ceiling.py`,
  `scripts/run_a0p.py` (historical reproduction of #20), and all other `scripts/`.
- **Docs:** a `progress.md` entry for the protocol change; README, CLAUDE.md and
  `docs/tools.md` rewritten for v2; the project page later.
- **Pilot before the grid:** one full-size run per arm on `pi_mid/1001` to calibrate
  `T_p`, `T_f` and the USD line.

## 15. Pre-implementation spikes

Each spike is answered by a short throwaway run, read from the gateway log where
possible. Implementation proceeds only once S1–S4 pass. S5 informs a setting.

| id | question | pass condition |
|---|---|---|
| S1 | Does the CLI inside bwrap reach Vertex through the gateway with `CLAUDE_CODE_SKIP_VERTEX_AUTH=1` on the current project (`lightwell-devel`), is `claude-opus-4-8` available there, and what happens to background small-model requests? | a tool-using turn completes; every request is logged; background requests are either absent, routed to the pinned model, or fail harmlessly |
| S2 | After `system_prompt` replacement and `setting_sources=[]`, what does the request actually contain? | no `~/.claude` or project CLAUDE.md content; the system prompt is ours plus tool schemas only |
| S3 | Can resuming with a different tool list remove tools from the request? | the gateway log shows the phase-1 tools absent after the switch (else: handler gating only, documented) |
| S4 | Is `interrupt()` then `query()` a reliable phase switch in one session? | no practice tool executes after the trigger; the final opening message is the next user turn |
| S5 | When does auto-compaction trigger in a 100-turn practice phase, and should it be controlled? | a decision recorded with the observed trigger point |

## 16. Tests

1. **Data layer (CPU):**
   - validation split determinism;
   - existing demos/test bit-identical (hash of `pi_mid/1001`);
   - validation satisfies iid constraints and is disjoint from demos and test;
   - every `query_violation` reason, including canonicalisation;
   - `test.jsonl` shuffled with no split information.
2. **Answer legality:**
   - each rule in §6.2 violated in isolation;
   - list length never checked;
   - illegal files consume no submission and never commit.
3. **Tool layer:**
   - the §6.1 availability table asserted as a whole;
   - refused queries not charged, malformed charged, mid-batch Q exhaustion;
   - feedback contains aggregates only;
   - `build_dataset` provenance;
   - `train` caps;
   - the final timeout rule (§4.4).
4. **Isolation (host, no API; kept as a permanent regression):** run the wrapper with a
   stub in place of the CLI and assert:
   - no read access to `/home`, the repo or gcloud credentials;
   - no `/dev/nvidia*`;
   - no network except the bridge;
   - an empty environment apart from the set variables;
   - `/work/task` not writable.
5. **Gateway (fake upstream):**
   - model allowlist;
   - usage accounting for all four token kinds;
   - retries not double-billed;
   - USD line triggers the phase-switch signal.
6. **End to end (API, marked slow):** the `smoke` preset with Q = 20, V = 50, test = 100,
   3 submissions, `T_p` = 10, `T_f` = 5, one run per arm.

---

## Appendix A — choices made while writing this spec

These were not discussed explicitly. Each is the least surprising reading of what was
decided; override any of them at review.

1. Remaining Q, submissions and turns are echoed in every `glyph` tool result (§4.6).
2. A `train` call that reaches its GPU cap stops and saves rather than being refused (§6.3).
3. Reaching the USD line in the final phase follows the `T_f` timeout rule (§4.4).
4. Validation feedback accuracy is exact match after `strip()`, as for held-out scoring (§6.3).
5. `check_answers(set="test")` is rejected during practice (§6.3).
6. `queries.jsonl` records only successful results; refusals and malformed expressions
   appear in the tool result and the trace (§6.3).
7. `worker.py` and `sandbox.py` are removed rather than kept alongside v2 (§14).
8. Idle turns count toward the cap; three consecutive idle turns end the phase (§4.5).
9. `check_answers(set="validation")` stays available in the final phase; it is harmless
   and needs no special case (§6.1).

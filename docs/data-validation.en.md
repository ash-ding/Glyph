# Data Validation — how arms and reference oracles are measured

> Phase 2 of the pipeline. Once an instance exists (see **Data Generation**), we measure two
> different kinds of thing against it:
>
> 1. **CA-agent arms** — a frontier agent actually attempts the task, in a `train` or `no_train`
>    condition, inside a metered sandbox.
> 2. **Reference oracles** — a battery of fixed "ceilings" that say *how far a given kind of
>    knowledge can get you*, so any agent score has a frame of reference.
>
> A note up front: there is **no statistical p-value** anywhere in this system. The instance-level
> quantity is **π** (the skeleton's share of difficulty, see Data Generation §7); every arm and
> oracle reports accuracy-style metrics (`overall`, `by_split`, `tail`, `headroom`). Code is on
> `lumen1` under `~/code/Glyph-v2`; citations are `file:line`.

## 1. The metrics

Every oracle (and each arm) reports the same shape:

- **overall** — mean exact-match accuracy over the scored items.
- **by_split** — the same, broken out by test split: `iid`, `comp`, `depth` (Data Generation §6).
- **tail** — accuracy restricted to *tail* items (items whose required table cells were never
  seen/bought). `None` when there are no tail items.
- **headroom** — a normalized position between the skeleton ceiling and perfect
  (`seal.py:15-33`):

  ```
  headroom(score, ceiling) = (score - ceiling) / (1 - ceiling)      # None if ceiling >= 1
  ```

  `0.0` = "no better than knowing every structural rule and not one table entry"; `1.0` = perfect;
  **negative is meaningful and not clipped** (you did *worse* than the pure-skeleton baseline).

Scoring is **exact string match** after `.strip()` for both arms and oracles (`answers.py:163`,
`instance.py:395-404`). The lenient digit-match scorer is used *only* for the π measurement, never
here (`measure.py:44-46`).

## 2. What gets scored — the paired subset vs the full test

Two different scored populations, on purpose:

- **Reference oracles** score a **per-instance paired subset** of **500 items**:
  `paired_subset(inst, n=500, seed=777)` (`subset.py:20-35`). It is **stratified** (each split's
  quota ∝ its share of the test set) and **seed-fixed**, so *every oracle on that instance is
  scored on exactly the same 500 items* — the comparison is "paired," controlling for item
  difficulty. Each instance draws its **own** 500 from its **own** test items; the subset is not
  shared across instances. (A fixed 500 keeps the expensive oracles — GPU/API — affordable while
  staying representative.)
- **CA-agent arms** are scored on the **full** `inst.test` — all 10,000 items across the three
  splits (`report.py:36`). The larger population is what gives the ~0.5% standard error needed to
  separate arms at the 1–2% level.

## 3. The reference oracle battery

Five oracles. Three are cheap (CPU, always computed and committed); two are deferred (GPU/API).

### 3.1 skeleton ceiling (cheap)

**Setting:** the *true* skeleton composed over `IdentityTables` (`u(x)=x`, `b(x,y)=x`)
(`instance.py:350-382`). "Every structural rule, not one table entry." It measures how far the
**structure/reasoning half alone** gets you. High skeleton ceiling ⇒ a skeleton-dominated
(high-π) instance. It is also the baseline that `headroom` is measured against.

### 3.2 table ceiling (cheap)

**Setting:** the *true* tables composed over a **trivial skeleton** (`semantics.py:168-186`:
`UL`→plain map, `KL`→take-k, else identity). Deliberately *not* a no-op skeleton — a no-op would
also silently ablate the table (never invoke the atomic ops) and bias the measurement. It measures
how far the **table/memorization half alone** gets you. High table ceiling ⇒ a table-dominated
(low-π) instance.

### 3.3 perfect (cheap)

**Setting:** true skeleton + true tables ⇒ `overall = 1.0` by construction (`subset.py:63-68`).
A sanity check that the instance is solvable and the eval harness is wired correctly.

### 3.4 weights ceiling — `[GPU]`, deferred

**Question it answers:** how far does putting the table into *weights* (training) get you, at a
given coverage?

**Setting** (`weights_ceiling.py`): fine-tune a small student model
(`Qwen/Qwen3-1.7B`, `lr=1e-4`, `steps=6000`, `batch=128`, `weights_ceiling.py:156`) on a
`seen_frac` slice of the table. The "seen" set is chosen by a hash so it is *not* an arithmetic
progression (which would be an artificially easy region): `is_seen(key) = (key·2654435761) %
100000 < frac·100000`, with `seen_u(i)` keyed on the input value and `seen_b(i,j)` on `i·7919+j`
(`weights_ceiling.py:41-50`).

**Data granularity — atomic cells.** Each training example is a single table lookup:
`"u0 <val> ="` → `"<output>"` (and `"b0 <val> <val> ="` for binary) (`weights_ceiling.py:53-58`).
One example per seen cell, across all 3 unary + 2 binary operators. The student memorizes atomic
cells; digit-structure in the value embeddings lets it extrapolate to unseen cells.

**Scoring — table-only.** `score_ceiling` runs the **true** skeleton over the *student's* learned
tables (`weights_ceiling.py:258-282`). So composition is given exactly; the oracle isolates "how
well can the student learn the table from a `seen_frac` slice." It also reports `entries_needed`
(distinct unary/binary cells the items require).

### 3.5 A0′ ceiling — `[API]`, deferred

**Question it answers:** how far does putting the table into *context* (retrieval/in-context
evidence) get you, at a given coverage?

**Setting** (`a0prime.py`): `buy_evidence(inst, entries_target, probe_frac=0.7)` buys facts by
issuing queries in two phases (`a0prime.py:36-102`):

- **probe** — shallow, one operator (`depth_stop_prob=0.9`, budget 1), so each answer isolates a
  single table entry — the way an agent reads the table directly.
- **in-domain** — drawn at `demo_max_depth`, the way test items are, so the skeleton is inferable
  from the same distribution.

The stopping condition counts **distinct unary entries revealed** (`len(query_log.unary)`), not a
raw fact count. `entries_target = frac · n_values` (i.e. against a single unary table's worth of
cells).

**Data granularity — full expressions.** The `facts` handed to the model are the
`(rendered expression, answer)` pairs actually bought — probes are near-atomic by design, but
in-domain facts are *full composed expressions with their final answer*. The cell count is only
the coverage/stopping statistic; the context is **not** a cell-by-cell table dump.

**Scoring — end-to-end.** `run_a0prime(inst, items, evidence, answer_fn)` puts the evidence in the
frontier's context and asks it to answer the **test items** directly (`a0prime.py:146-198`). So
the frontier must do *both* table inference *and* skeleton composition in-context. (The real
frontier call is a later `[API]` step; the module itself never makes a network call — `answer_fn`
is an injectable seam.) `retrieval_split` (`a0prime.py:105-120`) partitions items into **covered**
(all needed cells were revealed) vs **uncovered**, for the retrieval-vs-extrapolation analysis.

### 3.6 The granularity contrast (important)

The two "partial coverage" oracles are **not** symmetric — and neither is the train arm (§4.3):

| | knowledge lives in | data granularity | composition |
|---|---|---|---|
| weights ceiling | model weights | **atomic cells** (`op val = out`) | given (true skeleton over student tables) |
| A0′ ceiling | context | **full expressions** (+ probes) | done by the frontier, in-context |

When comparing "context vs weights at the same coverage," these differences (granularity;
table-only vs end-to-end; and the `entries_target` denominator — A0′ counts distinct unary across
ops ≈ `frac·4913`, while `seen_u` is per-op ≈ `frac·3·4913`, a ~3× gap) must be reconciled first.
These are the open A0′-standardization design points; they are why the `[API]` runs are gated.

### 3.7 Where the results live

`tools/run_reference.py` writes `docs/benchmark/reference_ceilings.json`, keyed by instance id.
`CHEAP_ORACLES = (skeleton, table, perfect)` run by default; `weights`/`a0prime` only when
`--only` requests them (`run_reference.py:37, 193`). Cheap oracles store
`{overall, by_split, tail, headroom}`; `weights`/`a0prime` store the same **keyed by
`seen_frac`** (default scan `[0.02, 0.05, 0.10]`, `run_reference.py:196`), and evidence is
cumulative across increasing fractions.

## 4. The CA-agent arms

### 4.1 RunConfig

A run is configured by `RunConfig` (`harness.py:197-216`):

| field | default | meaning |
|---|---|---|
| `arm` | *(required)* | `"train"` or `"no_train"` |
| `preset` | `"pi_mid"` | which preset to generate from (ignored if `instance_id` set) |
| `instance_seed` | `1001` | seed for `generate()` |
| `instance_id` | `None` | if set, use a frozen manifest instance (fingerprint-verified) |
| `model` | `"claude-opus-4-8"` | the frontier model (pinned at the gateway) |
| `student_model` | `"Qwen/Qwen3-1.7B"` | train-arm local student (distinct from `model`) |
| `q_cap` | `1000` | total query budget |
| `submit_cap` | `20` | validation submissions before forced switch to final |
| `tp` | `100` | practice-phase turn cap |
| `tf` | `30` | final-phase turn cap |
| `usd_line` | `300.0` | cumulative-cost safety line |
| `n_val` | `5000` | validation split size |
| `max_turns` | `200` | SDK agent-loop backstop (must sit above `tp + tf`) |
| `effort` | `"high"` | pinned for comparability |

### 4.2 `train` vs `no_train`

The **only** difference is three extra tools, available in `train` only (`session.py:16-41`):

- `build_dataset` — turn purchased/demo/validation query pairs into a training JSONL.
- `train` — full-fine-tune the local `student_model` on a dataset (GPU-gated, under a
  cumulative GPU-seconds budget; default caps 1800 s/call, 7200 s total).
- `student_infer` — run the resulting checkpoint to answer table lookups.

Both arms can `query` the oracle, `submit`/`check_answers` against validation, and `final_answer`.
`no_train` has no student at all. `harness.run()` builds a `StudentPool` only when
`arm == "train"` (`harness.py:330-331`, `student.py`).

### 4.3 What the train arm's student is actually trained on — expressions, not cells

This is a subtle but important point. The `build_dataset` tool reads the agent-written JSONL
(`{"expr":…, "answer":…}` lines) and packs `Example(prompt="<expr> =", answer=" <answer>")`
(`student.py:162-206`). **The train-arm student is trained on full expression → answer pairs**,
whatever expressions the agent chooses (demos, purchased probes, or self-generated) — there is
**no** atomic-cell ingestion path.

This is a *different, coarser* granularity than the **weights ceiling** oracle (§3.4), which
enumerates atomic cells. So the train arm and the weights ceiling do **not** train the student on
the same kind of data, and are not directly comparable as "the same knowledge in weights."

*(Observed in the pilot `pi_mid`/seed 1001 train run: the agent's largest trained dataset was 160
examples = 30 demos + 130 self-generated expressions; it also hedged into a hand-written symbolic
solver for the skeleton. Its `overall` was 0.4197 vs 0.402 for `no_train` — training barely moved
it, on this near-crossover π=0.44 instance.)*

### 4.4 Scoring a run

`build_report` (`report.py:33-122`) scores the **committed final answers** against the **full**
`inst.test` (all 10k items) by exact match, and reports:

- `overall`, `by_split` (per test split), `by_depth` (per expression-tree depth);
- `tail` — the same restricted to tail items;
- `headroom` per split and for tail, vs `inst.ceilings(items)`;
- an `instance` block (`seed, preset, pi, n_structural, atomic_ratio, uses_binary_tables`);
- a `validation` block (submission history, best/last, `gap_last_vs_test_iid`);
- `covariates` (`q_used`, submissions, turns, `val_lookup_solvable`, `test_covered`,
  `final_from_student`, `final_commit`, `run_status`).

If nothing legal is committed, the score is all-zero (`_zero_score`).

### 4.5 Practice vs final phases

- **practice** (`drive_practice`, `harness.py:111-144`) — the agent explores: it can `query`,
  `submit` against validation, `check_answers`, and (train arm) build/train/infer a student. It
  ends on the `submit_cap`-th submit, an agent-called `finish_practice`, the `tp` turn cap, the
  `usd_line`, or 3 idle turns.
- **final** (`drive_final`, `harness.py:146-191`) — the agent commits answers via `final_answer`.
  If it ends without one but has a passing `check_answers(set="test")` path, that path is
  auto-committed (`final_commit="auto_checked_path"`); otherwise the score stays zero.

### 4.6 Metering gateway and the query oracle

A `Gateway` (`gateway.py`) sits between the sandboxed Claude Code CLI and Vertex AI: it forwards
only the **pinned** model's requests and parses each response's `usage` into a `Ledger`, so every
successful request is billed exactly once. The query oracle `t_query` (`tools.py:79-106`) enforces
the budgets: a query may be *refused* by the OOD policy (not charged) or hit `q_exhausted` (not
charged), but is otherwise **charged first** — even malformed expressions cost a query. `q_cap`
(1000) bounds queries, `submit_cap` (20) bounds validation submits, and `usd_line` ($300) is a
cumulative-cost safety line checked every turn.

## 5. Reading it in the viewer

The run viewer overlays, for each run, the reference battery of the frozen instance it used
(`skeleton`/`table`/`perfect`, plus `weights`/`a0prime` per `seen_frac` once populated), showing
`overall` and `tail` per oracle, plus `by_split` (`iid · comp · depth`) and `headroom`. The
Instances view lets you browse the 15 frozen instances by band → id and see the same battery.

---

*Companion: **Data Generation** — how an instance is built.*

# Bare atomics, multi-student training, and per-op probes — design

Date: 2026-09-29. Status: spec for review.

## Motivation

Three linked changes, all downstream of one grammar decision:

1. **Bare atomic expressions become legal** (`u0(v_a_b_c)`, `b1(v_x, v_y)`), so
   the agent can buy clean table cells during practice. Today atomic operators
   are reachable only through a structural operator, which is why the train arm
   is structurally starved (open_questions, starvation item, option (c)): it
   can only build expression-level, entangled training data, while the weights
   oracle trains on clean atomic cells. Per-atomic-op accuracy also becomes a
   measurable *subproblem* of the whole DSL.
2. **Multi-student training.** `train` gains a `student_id`: a new id starts a
   fresh student from the base model, an existing id continues training from
   that student's latest checkpoint. The agent can grow a pool of specialists
   (e.g. one per atomic operator) — whether it *chooses* to is part of what the
   benchmark measures, so the prompt describes the mechanics and never the
   strategy.
3. **Per-op reference ceilings.** The reference side gains per-operator
   specialist students and a per-op probe evaluation, so the agent's per-op
   performance has a trained-in-isolation yardstick.

## Settled scope decisions

- **Query-layer only.** Legality changes affect parsing/checking of *inputs*
  (queries, datasets). Generation (`_sample`, `_make_demos/_make_test/_make_val`)
  is untouched: the 15 frozen instances, their fingerprints, measured π, and
  the cheap ceilings all remain valid. The integrity test proves it.
- **1 cell = 1 query; Q stays 1000.** Q remains the coverage knob; no bulk
  channel. (#15 stays open.)
- **No config flag** for bare-atomic legality: it is unconditionally legal.
  Generation never emits it, so old instances regenerate identically.
- **Student-pool middle option:** id semantics + continue-training, shared
  global GPU cap. No per-student budget accounting, no student-count cap.
- **Probes are mandatory at Final** (enforced by answer-file legality), scored
  separately, never part of `overall`.

## A. Grammar: the `AtomApp` node

New AST node `AtomApp(op, args)`:

- `op` is an enabled atomic operator (`u*` arity 1, `b*` arity 2).
- **Args are value literals only.** No nesting (`u0(u1(x))` stays illegal), no
  list args, no structural sub-expressions. Rationale: the purpose is buying
  clean cells; nested atomic chains are a new expression family that never
  appears in any test item and would widen the public spec for nothing.
- `result_type = "VAL"`; `depth(AtomApp) = 1` (an application), safely under
  any `demo_max_depth`, so `query_violation`'s depth check never rejects it.
- `op_pairs` ignores it (it contains no structural nesting), so held-pair
  refusal is unaffected.

Touched: `data/grammar.py` (AST, parser, printer, checker, `syntax_spec`
text), `data/interp.py` (eval = one table lookup, **recorded in `LookupLog`**
— tail derivation, `_purchased_set`, and seen/unseen accounting then work
unchanged). `test_syntax_spec_leaks_no_semantics` must stay green: the spec
names the form and arities, nothing about meaning.

`build_dataset` reuses the same `parse`+`check`, so atomic examples become
valid training rows with no student-side change ("one surface form for
everyone" is preserved).

## B. Student pool: `student_id` + continue-training

Current behavior (verified in `v2/student.py`): every `train` call fine-tunes
from `base_model` and produces a fresh global checkpoint `ckN`; checkpoints
coexist; `student_infer` picks one. So multiple specialists are *expressible*
today — what is missing is continuation and discoverability.

Changes:

- `train(dataset_id, epochs, lr, student_id)`: unknown id → init from
  `base_model`; known id → init from that student's **latest** checkpoint.
  `StudentPool` keeps `{student_id: [ck…]}` lineages; checkpoints stay global
  (`ckN`) and remain valid `student_infer` targets. Result gains `student_id`.
- `train/sft.py::train` gains `init_checkpoint: str | None` (load weights from
  a checkpoint dir instead of the base model). Tokenizer always from base.
- `student_infer` additionally accepts a `student_id` meaning "its latest
  checkpoint"; the explicit `checkpoint` argument is unchanged.
- GPU accounting unchanged: one shared cap, training metered, inference not
  (status quo; the inference-unmetered issue stays a separate open question).
- `v2/tools.py`, `v2/mcp.py` schemas updated. **Prompt describes mechanics
  only** — "new id trains from base; existing id continues; students coexist"
  — never strategy. A code comment marks this as load-bearing for the
  measurement (the delegate-or-not decision is the quantity under test).

## C. Per-op probe set

New generation-side helper, e.g. `probe_set(inst, n_per_op=100)`:

- For each enabled atomic op, `n_per_op` bare-atomic items (unary: uniform
  values; binary: uniform pairs), rendered in public syntax, answers from the
  true tables.
- Deterministic from a **dedicated RNG stream** (instance seed + fixed salt,
  following `paired_subset(seed=777)`'s precedent). **Not** part of
  `inst.test`/`inst.val`, **not** hashed by the fingerprint → frozen instances
  stay frozen.
- Disjoint from test/val by construction (probes are bare-atomic; test/val
  items are structural).

Protocol wiring:

- **Final phase:** probe items are revealed alongside the 10k test.
  `final_answer` legality requires answers to all probe items; probe scores are
  reported in a separate `by_op` block and are **excluded from `overall`**.
- **Practice is not filtered:** querying a cell that happens to be a probe item
  is allowed (refusing would leak which cells are probes). Instead the report
  splits each op's probe accuracy into `seen`/`unseen` via the run's
  `LookupLog` — same epistemic stance as `tail` and
  retrieval-vs-extrapolation. The `unseen` column is the clean per-op
  generalization read.
- Report/`ScoreReport` and the viewer gain the `by_op` block.

## D. Reference side

1. **Align the weights oracle's surface form to the public syntax.**
   `weights_ceiling.py` currently trains on a private format
   (`prompt_unary = "u0 v_a_b_c ="` — not parseable in the public grammar).
   With AtomApp legal it switches to the public form (`u0(v_a_b_c) =`), which
   is what CLAUDE.md rule 7 intends. Consequence, recorded in progress.md: the
   published 0.498 was measured under the old format; weights numbers must be
   re-measured before comparison (those runs were deferred [GPU] anyway).
2. **Per-op specialist ceilings.** `train_student` (or a sibling) gains an
   `ops=[...]` restriction: train one student per atomic op on that op's cells
   only (same `seen_frac` hashing), score on the probe set → a per-op weights
   ceiling directly comparable to the agent's `by_op` block and to a
   multi-specialist train arm. `tools/run_reference.py` grows `--only
   weights-per-op` (or similar).
3. **Student base model becomes a first-class, end-to-end parameter** (CLI /
   `RunConfig` / reference tools), default unchanged `Qwen/Qwen3-1.7B`.
   Candidate small student: **Qwen3-0.6B** (smallest Qwen3; there is no 0.4B).
   **Gate:** per CLAUDE.md, self-check #5 (capacity) is the one whose failure
   forces a design change — a 0.6B student is admissible only after
   `scripts/capacity_check.py` passes for it under the cap5 protocol
   (4000 steps, batch 128, held-out 1-in-10; single GPU, 11–15 min). [GPU],
   user-gated.

## Testing

- Grammar: AtomApp parse/render/check round-trip; illegal forms (nesting, list
  args, bad arity, disabled ops); `syntax_spec` leak test still green.
- Interp: AtomApp eval correct + logged in `LookupLog`.
- Query: bare-atomic queries billed like any query; probe items not refused.
- **Frozen integrity test passes unchanged** — the proof that generation moved
  not at all.
- Student pool: lineage semantics (new id from base, known id from latest ck)
  against the fake backend; `init_checkpoint` plumbing in `sft.py` (slow/GPU
  test).
- Probes: determinism (same instance → same probes), public-syntax rendering,
  legality enforcement in `final_answer`, `seen/unseen` split correctness.
- `pytest -m "not slow"` green before every merge.

## Docs

`progress.md` entry (including the oracle-format break and its consequence);
`open_questions.md` starvation item updated (option (c): decided, opened);
`docs/tools.md` (new tool schemas); syntax spec; `data-generation.*` and
`data-validation.*` bilingual updates.

## Rollout

Independent PRs, in dependency order, each user-approved:

1. **A** — grammar `AtomApp` (+ interp + spec text).
2. **C** — probe set + final-phase wiring + report/viewer `by_op`.
3. **B** — student pool `student_id` + continue-training + prompt mechanics.
4. **D** — reference: public-syntax oracle format, per-op specialists, model
   parameter. (Actual GPU runs — 0.6B capacity check, weights re-measurement,
   per-op ceilings — are separate, user-gated executions, not part of the PR.)

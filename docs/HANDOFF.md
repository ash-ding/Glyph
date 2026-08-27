# Handoff

Read this first if you are picking this project up in a fresh session.

## What this is

A research program on **weight-space delegation**: when a frontier agent faces
a task it cannot solve alone, when does it pay to put capability into
*context*, into *code*, or into a small model's *weights* — and does the agent
itself know which? This repo is **Glyph**, the benchmark that makes the
question measurable (task T1 of the Phase 1 plan).

The framing that matters and is easy to get wrong: the teacher is **not
incapable**. It starts out not *possessing* the knowledge, buys it from the
environment during a prepare phase, and the student ends up better at
*deploying* it. Purchase and amortisation, not compression. Difficulty must be
of a kind training can absorb — informational or throughput-bound, never
capability-bound, because a capability-hard task defeats the smaller student too.

## Where the state lives

| | where |
|---|---|
| Code | this repo, `main` |
| Program roadmap (4 phases) | https://claude.ai/code/artifact/72f3adaf-01fa-453e-bbdf-0ff41e0a6f80 |
| Phase 1 experiment plan | https://claude.ai/code/artifact/ae321f3c-d618-4bdd-8b7f-e8e05e91db00 |
| Glyph benchmark spec | https://claude.ai/code/artifact/410cb90a-e9fd-4d00-b1c4-a2ebefdfb28a |

The three artifacts carry the reasoning; `README.md` and `CLAUDE.md` carry
what a contributor needs day to day. Numbering key (`W`/`D`/`E`/`A`/`H`/`π`…)
is in the Phase 1 plan §13.

## Status

Built and verified: benchmark core (`grammar` `semantics` `tables` `interp`
`instance` `measure`) plus `budget` (the ledger) and `trace`. 36 tests green.

| self-check | state |
|---|---|
| #1 round-trip | pass |
| #2 oracle exactness | pass — 1.000 on every preset |
| #3 π ordering | pass — 0.25 / 0.43 / 0.69 across presets |
| #4 hiddenness | **needs a frontier model** |
| #5 capacity | **needs a GPU** — the only check whose failure forces a design change |
| #6 tokenizer probe | script ready, needs a real tokenizer |

Not built: `train/sft.py`, `train/infer.py`, `sandbox/runner.py`, and the arm
runners (A2 context / A4 code / A6 weights). Deliberately — they cannot be
verified without a GPU and models.

## Decisions already settled

- **D1 → B**: no `floor` split. Test set fully solvable, ceiling 100%. The
  "ceiling calibration" argument for keeping it was **wrong** and was retracted:
  without `floor` the ceiling is already a clean 100%. Its only real value was
  as a free leakage detector, and E8 covers that.
- **D2 → C+B**: binary operators are digit-wise-dominant with a weak coupling
  term. Pure digit-wise is learnable but *enumerable* (867 entries ≈ 10k tokens),
  which hands the win to the code arm. **Unverified** until #5 reports unary and
  binary separately.
- **D8 → A**: value surface form spells out digits. Exact notation is decided by
  running #6, not by argument.

Still open (none block work): D3 digit base, D4 exact vs graded scoring for
arms, D5 noise ε (leaning: off in Phase 1), D6 whether queries return
intermediate steps, D7 demo count.

## Two findings from building it

**π's measurement was broken under exact match.** Any single wrong operator or
lookup ruins the whole answer, so both `L_skel` and `L_table` saturated near
1.0 and every preset measured π ≈ 0.5 — the knob did nothing. π now uses graded
credit (by digit, by list position); **arms are still scored by exact match**.

**The tables are verifiably structured.** Perturbing one input digit leaves the
output unchanged 28% of the time against a 0.02% chance baseline. That
correlation is the only reason anything extrapolates to unqueried entries, and
it is now a test.

## Next steps

1. Run **#6** (`python scripts/probe_tokenizer.py Qwen/Qwen3-1.7B`), set
   `GlyphConfig.value_form` from the result.
2. Run **#5** in parallel — it needs no benchmark, and it is the only check
   that can invalidate the student choice. Report unary and binary separately
   and sweep `binary_coupling` (that is D2's verification).
3. Then the training stack and the A2/A4/A6 runners, i.e. the minimum closed
   loop: agent queries → seal → evaluate → one score. W2's E0 pilot needs it.

W2 settles two gates: does a crossover exist at all, and — via **A0′**, the
information-saturated frontier — is the weights advantage economic or
learning-algorithmic. A0′ decides the paper's claim strength, so do not skip it.

## Things that only exist in the conversation

Mostly folded into the artifacts already. The exceptions worth knowing:

- Generation is **not** a bottleneck: 10k items take 1.5–11 s. `GlyphConfig.scaled(n)`
  exists for fast test feedback, not out of necessity; real runs use the full 10k.
- The `pi_low` preset has high between-instance variance (π ranged 0.02–0.49
  across four seeds). This is the between-instance variance the plan already
  accounts for with ≥5 instances and paired comparison — and it is why phase
  diagram axes must use `measured_pi()`, never a preset name.
- A candidate repo name for the wider harness (arms, analysis, orchestrator)
  was `substrate`; nothing depends on it.

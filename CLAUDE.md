# CLAUDE.md

## What this is

Glyph — a hidden-semantics DSL execution benchmark. It is task **T1** of the
Phase 1 experiment plan for a research program on weight-space delegation.
Read `README.md` first; it carries the design rationale.

## Rules that exist for a reason

1. **The ledger is the only way to advance the clock.** Anything that consumes
   compute goes through `budget.Ledger.charge`. One bypass and the crossover
   figure is meaningless.
2. **The public syntax must leak no semantics.** Operator names stay opaque;
   `tests/test_grammar.py::test_syntax_spec_leaks_no_semantics` enforces it.
3. **π's numerator is `L_skel`.** See README. Flipping it fails silently.
4. **The two π baselines are frozen.** Changing `trivial_skeleton` or
   `IdentityTables` invalidates every previously measured π.
5. **The trivial skeleton must still invoke the tables.** Textbook defaults
   (`map`, left-fold), never "return the list unchanged."
6. **Analysis reads traces, never natural language.** Agent decisions go
   through typed schemas into JSONL.
7. **One surface form for everyone.** The agent and the student see the same
   value notation; converting between them would either leak the digit
   structure or quietly favour one arm.

## Settled decisions

- **D1 → B**: no `floor` split. Test set fully solvable, ceiling 100%.
- **D2 → C+B**: binary operators are digit-wise-dominant with a weak coupling
  term (`binary_coupling`). Pure digit-wise is learnable but also *enumerable*,
  which hands the win to the code arm. **Unverified** until the capacity check
  reports unary and binary separately.
- **D8 → A**: value surface form spells out digits. Exact notation comes from
  `scripts/probe_tokenizer.py` (self-check #6), not from argument.

## Before touching the arms

Self-checks in order **6 → 5 → 1 → 2 → 3 → 4** (`README.md`). #5, the capacity
check, is the only one whose failure forces a design change rather than a bug
fix — run it first and in parallel, it needs no benchmark.

## Conventions

- Python 3.11+, numpy only in the core. No torch/transformers below `arms/`.
- `pytest -q` must be green before every commit; `-m slow` covers π sweeps.
- Use `GlyphConfig.scaled(n)` for fast iteration; real runs use the full 10k.

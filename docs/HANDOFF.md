# Handoff

Read this first if you are picking this project up in a fresh session.

## What this is

A research program on **weight-space delegation**: when a frontier agent faces a
task it cannot solve alone, does it pay to train a small model on what it learns
from the environment, or to just keep reasoning with the frontier — and does the
agent itself know which? This repo is **Glyph**, the benchmark that makes the
question measurable (task T1 of the Phase 1 plan).

The framing that matters and is easy to get wrong: the teacher is **not
incapable**. It starts out not *possessing* the knowledge, buys it from the
environment during the practice phase, and (in the train arm) a student ends up
better at *deploying* it. Purchase and amortisation, not compression. Difficulty
must be of a kind training can absorb — informational or throughput-bound, never
capability-bound, because a capability-hard task defeats the smaller student too.

## The protocol (v2)

A run has two phases and compares two arms at equal stop conditions:

- **practice** — a *visible* validation set, a metered query oracle capped at
  `Q`, up to 20 aggregate-scored validation submissions, and (train arm only)
  student build/train/infer;
- **final** — the hidden held-out test is revealed, all practice tools are gone,
  and the agent commits once with `final_answer`.
- **arms** — `train` (frontier + a trainable Qwen3-1.7B student) vs `no_train`
  (frontier only), differing only in whether the student exists.

The agent runs under the **Claude Agent SDK**; the eight Glyph tools are
in-process MCP tools; the CLI is sealed in a `bwrap` sandbox; all model traffic
goes through a host-side metering gateway (pin, meter, credential injection).
The design is in `docs/superpowers/specs/2026-09-14-protocol-v2-design.md`; the
tool set and the `(arm, phase)` matrix are in `docs/tools.md`.

## Where the state lives

| | where |
|---|---|
| Code | this repo, `main` (v2 merged in PR #31) |
| v2 design spec | `docs/superpowers/specs/2026-09-14-protocol-v2-design.md` |
| v2 implementation plan | `docs/superpowers/plans/2026-09-14-protocol-v2.md` |
| Program roadmap (4 phases) | https://claude.ai/code/artifact/72f3adaf-01fa-453e-bbdf-0ff41e0a6f80 |
| Phase 1 experiment plan | https://claude.ai/code/artifact/ae321f3c-d618-4bdd-8b7f-e8e05e91db00 |
| Glyph benchmark spec (v1-era) | https://claude.ai/code/artifact/410cb90a-e9fd-4d00-b1c4-a2ebefdfb28a |

`README.md` and `CLAUDE.md` carry what a contributor needs day to day; the
progress log and open-questions list are the living record (below).

## Status

**Built and verified:** the benchmark core (`data/`: grammar, semantics, tables,
interp, instance, measure) — frozen 2026-08-31 — plus the full **protocol v2**
harness (`v2/`: session, tools, mcp, harness, gateway, sandbox/bridge, student,
ledger, answers, report, workspace, prompts, trace, cli), the training stack
(`train/sft.py`, `train/infer.py`), and scoring (`seal.py`: headroom,
score_answers). `pytest -m "not slow"` = **194 tests green**; the `slow` marker
covers the π sweeps, the GPU tests, and the real-API end-to-end run.

The harness is validated end to end: staged feasibility spikes proved the
SDK → bwrap → gateway → Vertex chain (no host-config leak, model-pin, `/v1`
re-insertion), and a real **`train`-arm run completes and is scored** (a smoke
run committed a held-out answer at `overall = 0.53`; the student SFT path
`build_dataset → train Qwen3-1.7B → infer` runs on GPU).

**Data-layer self-checks:**

| self-check | state |
|---|---|
| #1 round-trip | pass |
| #2 oracle exactness | pass — 1.000 on every preset |
| #3 π ordering | pass — 0.25 / 0.43 / 0.69 across presets |
| #4 hiddenness | measured — A0′ 0.258 vs weights 0.498 (see README) |
| #5 capacity | measured on GPU — weights arm reaches 0.498 at 10% table seen |
| #6 tokenizer probe | run — `value_form` set from it |

**Removed:** the v1 explore/prepare/seal protocol — arms A2/A4/A6/A7,
`seal`'s SealedArtifact/evaluate/calibration, `budget.py`, `worker.py`, the v1
sandbox and CLI. v2 replaces it.

## Decisions already settled

- **D1 → B**: no `floor` split. Test set fully solvable, ceiling 100%; E8 covers
  the leakage-detection role `floor` would have served.
- **D2 → C+B**: binary operators are digit-wise-dominant with a weak coupling
  term. Pure digit-wise is learnable but *enumerable*, which hands the win to
  code; the coupling term prevents that.
- **D8 → A**: value surface form spells out digits, decided by running #6.
- **query_ood_policy → strict** (v2 D1): a practice `query` may not probe
  `comp`/`depth` items. Opening it is a deferred comparison, not a redesign.

## Two findings from building the data layer

**π's measurement was broken under exact match.** Any single wrong operator or
lookup ruins the whole answer, so both `L_skel` and `L_table` saturated near 1.0
and every preset measured π ≈ 0.5 — the knob did nothing. π now uses graded
credit (by digit, by list position); **the test set is still scored by exact
match**.

**The tables are verifiably structured.** Perturbing one input digit leaves the
output unchanged 28% of the time against a 0.02% chance baseline. That
correlation is the only reason anything extrapolates to unqueried entries, and
it is now a test.

## Next steps

1. **Pilot calibration** (Task 20): one full-size `pi_mid/1001` run per arm to
   set `T_p` / `T_f` / the USD line for the grid; record in
   `docs/superpowers/plans/pilot-notes.md`.
2. **The Q sweep**: `Q` is the coverage axis. Sweep it to compare `train` vs
   `no_train` at matched coverage (open item, see `docs/open_questions.md`).
3. **The grid**: multiple seeds × both arms, paired on the instance, at the
   calibrated caps.

The paper's claim strength turns on **A0′** — the information-saturated frontier
— which decides whether the weights advantage is economic or
learning-algorithmic. Do not skip it.

## Things that only exist in the conversation

- Generation is **not** a bottleneck: 10k items take 1.5–11 s. `GlyphConfig.scaled(n)`
  exists for fast test feedback; real runs use the full 10k.
- The `pi_low` preset has high between-instance variance (π ranged 0.02–0.49
  across four seeds) — which is why phase-diagram axes must use `measured_pi()`,
  never a preset name. A related open item: some `pi_low` seeds produce a test
  set needing no binary-table entries (`uses_binary_tables` False).
- lumen `ANTHROPIC_VERTEX_PROJECT_ID=lightwell-devel`; model `claude-opus-4-8`,
  `claude-agent-sdk==0.2.152` + bundled CLI, bwrap 0.11.2, effort `high` —
  pinned for comparability.

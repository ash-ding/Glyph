# Frozen Instance Set + Per-Instance Reference Ceilings — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a frozen, stratified 3×5 benchmark of instances (selected by measured π) and a per-instance reference battery (skeleton/table ceiling, weights training-ceiling, A0′ retrieval), then surface those references in the run viewer — separating data generation/characterization from evaluation.

**Architecture:** A new `src/glyph/reference/` package (frozen-set loader + fingerprint, the paired-subset + CPU ceilings, the ported weights-ceiling and A0′ oracles) plus `tools/` scripts that produce committed JSON artifacts under `docs/benchmark/`. The run viewer reads those artifacts to show each run's instance in context. No change to the protocol, tools, or scoring codec.

**Tech Stack:** Python 3.11 (numpy-only data layer; torch/transformers for the weights ceiling; the v2 metering gateway for A0′), static HTML/JS viewer.

**Spec:** `docs/superpowers/specs/2026-09-16-frozen-instances-and-reference-ceilings-design.md`

## Global Constraints

- **Data-layer boundary:** nothing under `src/glyph/data/` may import outside `data/`. The new code lives in `src/glyph/reference/` (which may import `glyph.data`, `glyph.train`, `glyph.v2`).
- **Determinism:** generation is bit-deterministic from the seed; the frozen set stores seeds + fingerprints, never data. The fingerprint extends `tests/test_backcompat.py`'s (demos+test+held_pairs) to also cover `val`.
- **Reference ≠ agent:** these oracles are reference/ceiling runs (no agent loop, no metering budget). The weights-ceiling hyperparameters are fixed and distinct from what an agent picks in the `train` arm.
- **Committed artifacts** live under `docs/benchmark/`: `candidates.json`, `frozen_instances.json`, `reference_ceilings.json`.
- **Fixed values:** value space `|V| = 17³ = 4913`; per instance 3 unary + 2 binary operators; scored on a **stratified 500-item** subset (fixed sampling seed 777) for paired comparison.
- **π bands (defaults, tunable, recorded in the manifest):** low `[0.20,0.35)`, mid `[0.40,0.55)`, high `[0.60,0.80)`, drawn from `pi_low`/`pi_mid`/`pi_high` respectively; **N=5 per band → 15**.
- **Commit trailer:** end every commit with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **Cost flags:** Tasks marked **[GPU]** need a free GPU; **[API $]** spend real API budget — do not run their execution step without explicit go-ahead; their *code + CPU tests* can be built and reviewed freely.

---

## File Structure

- `src/glyph/reference/__init__.py`
- `src/glyph/reference/frozen.py` — `fingerprint(inst)`, `load_frozen(path)`, band helpers.
- `src/glyph/reference/subset.py` — `paired_subset(inst, n, seed=777)` and CPU `ceilings_on(inst, items)`.
- `src/glyph/reference/weights_ceiling.py` — port of `scripts/table_ceiling.py` (v2, torch lazy).
- `src/glyph/reference/a0prime.py` — port of `scripts/run_a0p.py` (v2, via the gateway).
- `tools/scan_instances.py`, `tools/select_instances.py`, `tools/run_reference.py`
- `docs/benchmark/candidates.json`, `frozen_instances.json`, `reference_ceilings.json`
- `tests/reference/test_frozen.py`, `test_subset.py`, `test_select.py`, `test_weights_ceiling.py`, `test_a0prime.py`, `test_run_reference.py`, `tests/test_frozen_instances.py` (integrity)
- `glyph-viewer/index.html` — reference overlay.

---

### Task 1: Fingerprint + frozen-set loader

**Files:**
- Create: `src/glyph/reference/__init__.py`, `src/glyph/reference/frozen.py`
- Test: `tests/reference/test_frozen.py`

**Interfaces:**
- Produces: `fingerprint(inst) -> str` (sha256 hex over demos + test[split,expr,answer] + sorted held_pairs + **val[expr,answer]**); `load_frozen(path="docs/benchmark/frozen_instances.json") -> list[dict]` (each `{id, band, preset, seed, measured_pi, fingerprint_sha256}`); `band_of(pi, cutoffs) -> str | None`.

- [ ] **Step 1: Failing test** — `test_fingerprint_matches_backcompat_plus_val`: for `generate(1001, PRESETS["smoke"])`, the demos+test+held part of `fingerprint` equals the existing `test_backcompat` baseline prefix, and `fingerprint` is stable across two calls; `test_band_of` maps 0.27→"low", 0.5→"mid", 0.37→None (gap).
- [ ] **Step 2: Run to confirm it fails** (module missing).
- [ ] **Step 3: Implement** `frozen.py`: reuse the backcompat fingerprint construction and append `V|expr|answer` rows for `inst.val`; `load_frozen` reads JSON; `band_of` returns the band whose `[lo,hi)` contains `pi`, else None. `DEFAULT_CUTOFFS = {"low":(0.20,0.35),"mid":(0.40,0.55),"high":(0.60,0.80)}`.
- [ ] **Step 4: Run tests — green.**
- [ ] **Step 5: Commit.**

---

### Task 2: Paired subset + CPU ceilings

**Files:**
- Create: `src/glyph/reference/subset.py`
- Test: `tests/reference/test_subset.py`

**Interfaces:**
- Consumes: `glyph.data` (generate, Interpreter, IdentityTables), `inst.ceilings`, `inst.is_tail`.
- Produces: `paired_subset(inst, n=500, seed=777) -> list[TestItem]` (stratified proportional per split, fixed seed — copy the standalone function from `scripts/run_a0p.py:paired_subset`, which has no `glyph.arms` dependency); `ceilings_on(inst, items) -> dict` returning `{skeleton,table,perfect}` each `{overall, by_split, tail, headroom}` on `items`.

- [ ] **Step 1: Failing test** — `test_paired_subset_stratified_and_fixed`: `len == 500` (for a full preset) or `<=len(test)`; same seed → identical ids; proportions per split within ±1 of `n*share`. `test_cpu_ceilings_ordered`: `skeleton.overall < table.overall <= 1.0`, `perfect.overall == 1.0`, every `headroom` in `[−ε,1]`, `skeleton.tail == 0.0`.
- [ ] **Step 2: Run to confirm fail.**
- [ ] **Step 3: Implement** using `inst.ceilings(items)` for skeleton/table and constructing perfect via the true interpreter; `headroom` from `glyph.seal.headroom`.
- [ ] **Step 4: Green.**
- [ ] **Step 5: Commit.**

---

### Task 3: Candidate scan tool

**Files:**
- Create: `tools/scan_instances.py`
- Test: `tests/reference/test_scan.py`

**Interfaces:**
- Produces: `scan(presets, n_seeds, start_seed) -> list[dict]` (per candidate: `seed, preset, pi, pi_components, uses_binary_tables, fingerprint, skeleton_ceiling_overall`); writes `docs/benchmark/candidates.json`; prints a per-preset measured-π histogram.

- [ ] **Step 1: Failing test** — `test_scan_smoke_small`: `scan(["smoke"], n_seeds=3, start_seed=1001)` returns 3 dicts, each with the required keys, `pi` in `[0,1]`, `fingerprint` 64-hex; deterministic across two calls.
- [ ] **Step 2: Run to confirm fail.**
- [ ] **Step 3: Implement** the scan + JSON write + a text histogram (10 bins over π). CLI: `--presets`, `--n-seeds` (default 60), `--start-seed`, `--out`.
- [ ] **Step 4: Green.**
- [ ] **Step 5: Commit.**

---

### Task 4: Selection tool → manifest

**Files:**
- Create: `tools/select_instances.py`
- Test: `tests/reference/test_select.py`

**Interfaces:**
- Consumes: `candidates.json`, `frozen.band_of`.
- Produces: `select(candidates, cutoffs, n_per_band) -> list[dict]` (filters `uses_binary_tables False`; keeps in-band; picks N spread across the band by π); writes `docs/benchmark/frozen_instances.json` with `band_cutoffs`, `n_per_band`, and the chosen instances.

- [ ] **Step 1: Failing test** — `test_select_from_synthetic`: given a synthetic candidate list spanning π 0..1 (some with `uses_binary_tables False`, some in gaps), `select(..., n_per_band=2)` returns exactly 2 per band, all in-band, none degenerate, and the two per band are not both at the same π.
- [ ] **Step 2: Run to confirm fail.**
- [ ] **Step 3: Implement** the filter + per-band spread selection (sort in-band candidates by π, pick evenly spaced N). Manifest write.
- [ ] **Step 4: Green.**
- [ ] **Step 5: Commit.**

---

### Task 5: Produce & commit the frozen manifest  [CPU, execution step]

**Files:**
- Create (data): `docs/benchmark/candidates.json`, `docs/benchmark/frozen_instances.json`
- Create: `tests/test_frozen_instances.py` (integrity gate)

- [ ] **Step 1: Integrity test first** — `test_manifest_reproduces`: for every entry in `frozen_instances.json`, `generate(seed, PRESETS[preset])` reproduces `fingerprint_sha256`, its measured π is inside its band's cutoffs, and `uses_binary_tables` is True. `test_manifest_shape`: 3 bands present, `n_per_band` entries each.
- [ ] **Step 2: Run the scan** across the three presets (default 60 seeds each), commit `candidates.json`, and print the histogram for the record.
- [ ] **Step 3: Select** N=5 per band (adjust cutoffs only if the histogram clearly requires it — record any change), write and commit `frozen_instances.json`.
- [ ] **Step 4: Run the integrity test — green.**
- [ ] **Step 5: Commit** the manifests + test.

---

### Task 6: Weights-ceiling reference (port `table_ceiling.py`)  [GPU]

**Files:**
- Create: `src/glyph/reference/weights_ceiling.py`
- Test: `tests/reference/test_weights_ceiling.py`

**Interfaces:**
- Consumes: `glyph.data` (generate, tables, grammar names), `glyph.train.sft` optional; torch imported lazily.
- Produces: `seen_u(i, frac)`, `seen_b(i, j, frac)` (hash-based, verbatim from the v1 script); `train_student(inst, seen_frac, *, model, lr=1e-4, steps=6000, batch=128) -> model` [GPU]; `score_ceiling(inst, model, tok, items) -> dict` (build `StudentTables` from the model's answers to every entry the `items` need, run the true skeleton over them, return `{overall, by_split, tail, headroom, entries_needed}`).

- [ ] **Step 1: Failing test** (CPU only) — `test_seen_frac_is_hash_stable` (deterministic, ~`frac` fraction over 4913); `test_score_ceiling_with_perfect_fake_student`: a fake student whose `StudentTables` returns the *true* table gives `overall == table_ceiling.overall` on the subset (i.e. the scoring wiring is correct without any GPU).
- [ ] **Step 2: Run to confirm fail.**
- [ ] **Step 3: Implement** by porting `scripts/table_ceiling.py`'s `stream/StudentTables/generate_answers/score` into the module, replacing `from run_a0p import paired_subset` with `glyph.reference.subset.paired_subset`; keep torch/transformers imports inside `train_student`.
- [ ] **Step 4: Run the CPU test — green.** Mark the real GPU training path with a `slow` marker (a separate, opt-in `test_train_smoke` that trains 50 steps and asserts loss decreases).
- [ ] **Step 5: Commit.**

---

### Task 7: A0′ reference (port `run_a0p.py`)  [API $]

**Files:**
- Create: `src/glyph/reference/a0prime.py`
- Test: `tests/reference/test_a0prime.py`

**Interfaces:**
- Consumes: `glyph.data`, the v2 metering gateway for the frontier call.
- Produces: `buy_evidence(inst, entries_target, probe_frac=0.7, seed) -> (facts, revealed)` (purchase probes + in-domain through the real oracle until ~`entries_target` unary entries are revealed); `retrieval_split(inst, items, revealed) -> {covered_n, covered, uncovered_n, uncovered}` (split `items` by `is_tail` against `revealed` — the 126/374 analysis, pure/CPU); `run_a0prime(inst, items, evidence, *, model, chunk=50) -> dict` [API $] (frontier answers with evidence in context via the gateway; `{overall, by_split, tail, headroom}`).

- [ ] **Step 1: Failing test** (CPU only) — `test_buy_evidence_reveals_entries` (returns real facts and a non-empty `revealed` set, all facts legal); `test_retrieval_split_partitions`: `covered_n + uncovered_n == len(items)`, `covered` entries all satisfy `needs ⊆ revealed`, `uncovered` do not.
- [ ] **Step 2: Run to confirm fail.**
- [ ] **Step 3: Implement** by porting the `buy`/`paired_subset` logic (drop the `glyph.arms` import; the answering call goes through a small gateway client rather than v1's `a0p_saturated.run`). Parametrize evidence by `entries_target` so A0′ and the weights ceiling can be compared at matched coverage.
- [ ] **Step 4: Run the CPU test — green.** The real frontier call is an opt-in `slow` test.
- [ ] **Step 5: Commit.**

---

### Task 8: Reference-battery runner → `reference_ceilings.json`

**Files:**
- Create: `tools/run_reference.py`
- Test: `tests/reference/test_run_reference.py`

**Interfaces:**
- Consumes: `frozen.load_frozen`, `subset`, `weights_ceiling`, `a0prime`.
- Produces: `merge_reference(path, instance_id, oracle, payload)` (idempotent per-oracle merge into `reference_ceilings.json`); CLI `run_reference.py --instance <id|all> --only <skeleton,table,weights,a0prime> [--seen-frac ...]`.

- [ ] **Step 1: Failing test** — `test_merge_is_per_oracle_idempotent`: merging `weights` for `low_1` then `a0prime` for `low_1` leaves both, re-merging `weights` overwrites only `weights`; unrelated instances untouched.
- [ ] **Step 2: Run to confirm fail.**
- [ ] **Step 3: Implement** the merge + CLI that dispatches to each oracle (cheap ones run inline; `--only` gates the GPU/API ones).
- [ ] **Step 4: Green.**
- [ ] **Step 5: Commit.**

---

### Task 9: Run the cheap ceilings on the frozen set  [CPU, execution step]

- [ ] **Step 1:** `python tools/run_reference.py --instance all --only skeleton,table,perfect` over the 15 frozen instances; commit the resulting `reference_ceilings.json` (cheap CPU numbers for all 15). The GPU (`weights`) and API (`a0prime`) columns are filled later, on explicit go-ahead, per the spec's cost notes.
- [ ] **Step 2: Commit** `reference_ceilings.json`.

---

### Task 10: Viewer — show the instance's reference battery beside the run

**Files:**
- Modify: `glyph-viewer/index.html`

**Interfaces:**
- Consumes: a run's `config.seed`/`preset`; `docs/benchmark/reference_ceilings.json` + `frozen_instances.json` (served alongside the runs, or copied into the viewer data dir by `collect_runs`).

- [ ] **Step 1:** teach `tools/collect_runs.py` (and `serve_viewer.py`) to also copy `docs/benchmark/{frozen_instances,reference_ceilings}.json` into the viewer data dir. Test: `test_collect_copies_benchmark` — after collect, both files are present in the out dir.
- [ ] **Step 2:** in the viewer, when a run is opened, look up its instance (by seed/preset) in the benchmark files and render a **Reference** card next to the stat cards: the instance's band + measured π + `uses_binary_tables`, and the reference battery rows (skeleton / table / perfect / weights@seen_frac / a0prime) with `overall` and `tail`, so the run's own `overall`/`tail` sit against them. Degrade gracefully when a run's instance is not in the frozen set (show "not a frozen instance").
- [ ] **Step 3:** manual visual check via the served viewer; then commit.

---

## Notes for the executor

- Tasks 1–4, 8, 10-step-1 are cheap and fully unit-tested. Tasks 5 and 9 are CPU execution steps that produce committed data. Tasks 6 and 7 build code + CPU tests freely, but their **[GPU]/[API $] execution** (real training / real frontier calls to fill `reference_ceilings.json`) waits for explicit go-ahead — flag and pause there.
- The v1 sources to port are `scripts/table_ceiling.py` (Task 6) and `scripts/run_a0p.py` (Task 7); both currently import the deleted `glyph.arms` and must be lifted into `src/glyph/reference/` without that dependency.

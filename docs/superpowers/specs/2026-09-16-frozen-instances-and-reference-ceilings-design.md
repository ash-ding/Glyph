# Frozen instance set + per-instance reference ceilings — design

**Status:** draft for review
**Date:** 2026-09-16
**Depends on:** protocol v2 (`docs/superpowers/specs/2026-09-14-protocol-v2-design.md`)

## 1. Goal

Separate **data generation / instance characterization** (deterministic, done
once, frozen) from **evaluation** (agent runs). Concretely:

1. Freeze a **stratified benchmark of 3×5 = 15 instances** — five per measured-π
   band (low / mid / high) — selected by *measured* π and committed as a
   manifest, so every future trial reuses exactly the same instances.
2. For each frozen instance, run a **reference battery** of ideal-strategy
   oracles that bracket the achievable score — in particular a **training upper
   bound** (a fully-trained student in the table slot) and a **retrieval upper
   bound** (the information-saturated frontier). Commit the results as a
   per-instance **reference table**.
3. Read every agent run (`train` / `no_train`) **against that reference table**,
   so a raw score becomes an interpretable fraction of what each ideal strategy
   achieves on that specific instance.

This makes the benchmark **reusable** (fixed instances), **fair** (paired on the
instance), and **interpretable** (scores relative to per-instance ceilings).

## 2. Motivation

- **π drifts from the preset.** A preset (`pi_low/mid/high`) only *biases* π; the
  *measured* π of a given seed varies — seven of twenty `pi_mid` seeds measure
  inside `pi_high`'s range. So instances must be selected by **measured π**, never
  by preset name (the codebase already mandates `measured_pi()` for phase-diagram
  axes). A one-off "5 seeds per preset" would mislabel bands.
- **Generation is bit-deterministic** (verified 2026-09-16: demos, held-out test,
  validation, skeleton, tables, and measured π are byte-identical across separate
  processes and different `PYTHONHASHSEED`). So a frozen set needs to store only
  **seeds**, not data — the data regenerates identically from the seed. The only
  non-deterministic artifact is the *iteration order* of the `held_pairs` set,
  which does not leak into any generated data.
- **The reference machinery already exists in spirit** — `measured_pi()`,
  `inst.ceilings()`, `headroom`, and the v1 scripts `scripts/run_a0p.py` (A0′) and
  `scripts/table_ceiling.py` (weights ceiling) — but the two scripts import the
  deleted `glyph.arms` package and no longer run. This spec ports their design
  into the v2 world as a clean reference module.

## 3. Definitions (from the data layer)

- **Value space:** `|V| = base^n_digits = 17³ = 4913` values, rendered `v_x_y_z`.
- **Operators per instance:** `n_structural = 5` skeleton operators (5 of `s0..s7`),
  `n_unary = 3` unary operators (`u0,u1,u2`), `n_binary = 2` binary (`b0,b1`).
- **Table entry:** a single cell — a unary entry is `(u_k, i) → out` (4913 per
  unary op; 3×4913 = 14 739 unary cells total); a binary entry is `(b_k, i, j) →
  out` (~4913² per binary op).
- **What an item needs:** `TestItem.needs_u` / `needs_b` — the exact table cells a
  test item's answer depends on.
- **tail (per oracle):** items whose needed entries were **not** all supplied to
  that oracle (`is_tail` = `needs − revealed ≠ ∅`). On the tail the skeleton
  ceiling is 0, so tail accuracy is pure table knowledge.
- **Metrics** (all exact-match on the scored subset):
  - `overall` — accuracy on all scored items;
  - `by_split` — accuracy on `iid` / `comp` / `depth`;
  - `tail` — accuracy on the tail subset;
  - `headroom = (score − skeleton_ceiling) / (1 − skeleton_ceiling)` — the
    fraction of the room above the code-only floor that a strategy captures.
- **Scored subset:** a **stratified 500-item sample** of the 10 000-item test
  set (proportional per split, fixed sampling seed), so every oracle and arm is
  scored on the **same** items (paired; ~halves the SE on a difference).

## 4. Part A — the frozen instance set

### 4.1 π bands

Three bands with **guard gaps**, so each instance is unambiguously in-band
(defaults; tunable in one config constant):

| band | measured-π range | drawn from preset |
|---|---|---|
| **low**  | `[0.20, 0.35)` | `pi_low`  |
| **mid**  | `[0.40, 0.55)` | `pi_mid`  |
| **high** | `[0.60, 0.80)` | `pi_high` |

Seeds landing in a gap `(0.35,0.40)` or `(0.55,0.60)`, or outside `[0.20,0.80)`,
are discarded. Each band is drawn from its matching preset so that config knobs
other than π (e.g. `unary_coupling`) stay consistent **within** a band.

### 4.2 Selection algorithm

For each band, scan seeds `1001, 1002, …` from the band's preset; for each seed:

1. `generate(seed, preset)` and compute `measured_pi()["pi"]` (CPU, cheap —
   measured on a fixed 1500-item internal sample).
2. **Filter out** any seed that is degenerate or out-of-band:
   - measured π outside the band's range (the drift filter);
   - `instance.uses_binary_tables == False` (the deferred `pi_low`-without-binary
     corner — a test set needing no binary entries is excluded);
   - generation raised the short-test guard (an instance whose test set could not
     be filled).
3. Keep the first **5** surviving seeds per band.

### 4.3 The manifest

Commit `docs/benchmark/frozen_instances.json` (git-tracked, human-readable):

```json
{
  "created": "2026-09-16",
  "band_cutoffs": {"low": [0.20,0.35], "mid": [0.40,0.55], "high": [0.60,0.80]},
  "instances": [
    {"id": "low_1", "band": "low", "preset": "pi_low", "seed": 1003,
     "measured_pi": 0.271, "uses_binary_tables": true,
     "fingerprint_sha256": "…"}
  ]
}
```

- `fingerprint_sha256` is the same fingerprint `tests/test_backcompat.py` uses
  (demos + test + held_pairs), **extended to also cover `val`**, so a numpy /
  environment change that would silently perturb an instance is caught by a test.
- No data is stored — the seed regenerates it. A `frozen-instances` test asserts
  every manifest fingerprint still reproduces (integrity gate).

### 4.4 Tooling

- `tools/select_instances.py` — runs the scan/filter, writes the manifest,
  prints the chosen 15 with their measured π.
- `glyph.data` helper `load_frozen()` → the list of `(id, band, preset, seed)`
  so eval code and the harness can iterate the fixed set.

## 5. Part B — the per-instance reference battery

For each of the 15 frozen instances, compute the oracles below on the **same
stratified 500-item subset**, and record `overall / by_split / tail / headroom`
plus coverage (`entries_seen`, `entries_needed`). All are **reference runs**, not
agent runs — no agent loop, no metering budget.

### 5.1 The oracles

| oracle | what it knows | cost | source |
|---|---|---|---|
| **skeleton ceiling** | true skeleton + identity tables (no table entry) | CPU | `inst.ceilings()` |
| **table ceiling** | trivial skeleton + true tables | CPU | `inst.ceilings()` |
| **perfect** | true skeleton + true tables (= 1.000) | CPU | trivial |
| **weights ceiling** | true skeleton + a **student-learned** table | GPU | port of `table_ceiling.py` |
| **A0′ (retrieval)** | frontier + revealed entries in context | API $ | port of `run_a0p.py` |

### 5.2 weights ceiling (training upper bound) — the settings, made explicit

Ported from `table_ceiling.py`, unchanged in substance:

- **One** Qwen3-1.7B, **full fine-tune**, learning **all 5 operators' tables** at
  once (not one model per table). Training examples are single lookups:
  `u0 v_i =` → ` v_out`, `b0 v_i v_j =` → ` v_out`, mixed 50/50 unary/binary over
  every operator.
- **Coverage knob `seen_frac`** — a hash-based subset of table cells the student
  may train on. Reference sweeps `seen_frac ∈ {0.02, 0.05, 0.10}` (a real agent
  buys a few percent; 10% is the headline number, ≈ 491 of a unary op's 4913).
- **Hyperparameters (fixed for the reference):** lr `1e-4`, `6000` steps, batch
  `128`, bf16, grad-clip 1.0, seed `20260831`. *(These are the reference
  ceiling's hyperparameters — deliberately fixed and tuned — and are distinct
  from whatever an agent picks inside the `train` arm.)*
- **Scoring:** query the trained model for every entry the 500 items need, build a
  `StudentTables`, run the **true skeleton over the student tables**, exact-match.
  → "the most a weights arm could score if only the table remained a problem."

### 5.3 A0′ (retrieval upper bound) — settings and standardization

Ported from `run_a0p.py`, with two design fixes:

- **Evidence:** entries revealed by purchasing through the real oracle — 70%
  single-operator **probes** (each isolates one cell) + 30% **in-domain**
  expressions (so the skeleton is inferable). Re-sent in the frontier's context
  each 50-item call, unlimited thinking, exact-match.
- **Fix 1 — align the coverage knob.** Expose evidence size as `entries_seen`
  and run A0′ at the **same coverage points** as the weights sweep
  (`{0.02,0.05,0.10}` of the unary table), so "context vs weights" is compared at
  matched entries-seen rather than at one arbitrary 2000-fact budget.
- **Fix 2 — name the caveat as a control.** A0′'s evidence is drawn by the
  harness, not chosen adaptively. Keep the fixed-evidence A0′ as the reproducible
  ceiling, and record an **adaptive-buyer** variant (evidence = an actual agent's
  query log) as a deferred control (see Open questions).
- Routed through the **v2 metering gateway** (model-pinned, billed) rather than
  v1's direct call.

### 5.4 The reference table

Commit `docs/benchmark/reference_ceilings.json`, keyed by instance id:

```json
{
  "low_1": {
    "scored_items": 500,
    "skeleton_ceiling": {"overall": 0.31, "tail": 0.0, "by_split": {…}},
    "table_ceiling":    {"overall": …},
    "weights_ceiling":  {"0.02": {"overall": …, "tail": …}, "0.05": {…}, "0.10": {…}},
    "a0prime":          {"0.02": {"overall": …, "tail": …}, "0.05": {…}, "0.10": {…}},
    "retrieval_vs_extrapolation": {"covered_n": 126, "covered_acc": 0.976,
                                   "uncovered_n": 374, "uncovered_acc": 0.016}
  }
}
```

The CPU oracles (skeleton/table ceiling) are cheap enough to recompute; the GPU
(weights) and API (A0′) oracles are expensive, so their numbers are committed and
reused. A `--only` flag lets a costly oracle be filled in later without rerunning
the cheap ones.

## 6. Part C — evaluating agents against the references

- Agent runs (`train` / `no_train`) use the frozen set: `--seed` comes from the
  manifest; the grid iterates all 15.
- `build_report` gains a `reference` block: for the run's instance, attach the
  committed reference row and derive interpretable ratios, e.g.
  `overall / weights_ceiling@0.10`, `tail / weights_tail`, and the agent's
  position between `skeleton_ceiling` (0) and `weights_ceiling` (1).
- The run viewer overlays the reference band on the run's stat cards (the ceiling
  values and where the agent's score falls).

## 7. Non-goals / out of scope

- Choosing final numeric band cutoffs by hand-tuning — 4.1 gives defaults; a
  first scan may motivate small adjustments, recorded in the manifest.
- Multi-seed *run* replication (instance vs run variance, #25) — orthogonal.
- Changing the protocol, the tools, or the scoring codec.

## 8. Open questions

1. **A0′ adaptive-buyer control** — is the fixed-evidence ceiling enough, or do we
   also run A0′ on a real agent's query log per instance? (deferred)
2. **Band cutoffs & guard gaps** — the 4.1 defaults are provisional; the first
   scan reports the measured-π distribution per preset to confirm them.
3. **Reference cost** — A0′ per instance × 3 coverage points × 15 instances is a
   real API spend; do we run A0′ on all 15 or a representative subset per band?
4. **Where the weights ceiling's `seen` set and an agent's purchased set differ** —
   the reference uses a hash-based `seen_frac`; an agent buys adaptively. The
   comparison is "best case at matched coverage", not "same entries".

## 9. Implementation order (high level; the plan will expand this)

1. `select_instances.py` + `load_frozen()` + the manifest + the integrity test
   (covers Part A; CPU-only, cheap, unblocks everything).
2. Port `table_ceiling.py` → a v2 `reference.weights_ceiling` module; run the
   `seen_frac` sweep on the 15 instances (GPU).
3. Port `run_a0p.py` → a v2 `reference.a0prime` module via the gateway; run the
   coverage sweep (API).
4. Assemble `reference_ceilings.json`; add the `reference` block to `build_report`
   and the viewer overlay.

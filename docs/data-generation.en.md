# Data Generation — how a Glyph instance is built

> Phase 1 of the pipeline. Given a `(seed, preset)` pair, `generate()` deterministically
> produces one self-contained task ("instance"): a hidden program (skeleton + tables), a set
> of in-context demonstrations, a sealed test set, and a validation set. This document explains
> the value space, the configuration knobs, the three presets, the generation flow, and how the
> instance's difficulty (π) is measured.
>
> Code is on `lumen1` under `~/code/Glyph-v2`. Citations are `file:line`.

## 1. What an instance is

An instance is a hidden function `P = skeleton ∘ tables` over a finite value space, plus the
data needed to probe and grade an agent against it:

- **skeleton** — a small, *describable* program (a composition of structural operators) sampled
  from a finite grammar.
- **tables** — per-operator *indescribable* lookup functions (frozen random MLPs).
- **demos** — 30 solved `(expression, answer)` pairs the agent sees in context.
- **test** — a large sealed set (default 10,000) split into `iid` / `comp` / `depth`.
- **val** — a validation set (default 5,000) the agent can score against in aggregate during a run.

The whole point of the split into *skeleton* and *tables* is that the two halves are learned
differently: the skeleton is code/reasoning you can infer and write down; the tables are
memorization you can only get by seeing examples (see the companion **Data Validation** doc).

## 2. Value space and rendering

Every value is one point in a finite space `V` of size

```
|V| = base ** n_digits = 17 ** 3 = 4913     (config.py:47-48, 160-162)
```

A value index is encoded as `n_digits` little-endian digits, each in `[0, base)`
(`grammar.py:60-67`). The **surface form** shown to the agent is controlled by
`cfg.value_form` (default `"letter_sep"`), which renders each digit as a letter `a..q`
(17 symbols) — e.g. value → `v_k_e_e` (`grammar.py:69-81`). The `flat` form (`v1234`) is
deliberately rejected because it erases the digit structure the tables rely on
(`config.py:12-38`).

Why 3 digits over base 17: it gives 4913 distinct values while keeping a value to a short,
fixed token count, and — crucially — gives every value an internal **digit structure** that the
tables can generalize over (see §3).

## 3. Operators and tables

**Structural operators** (the skeleton's building blocks): 8 are defined, `s0..s7`
(`grammar.py:28-38`). Each has a *shape* deciding whether it returns a value or a list
(`UL`/`LB`/`L`/`KL`). `enabled_ops(cfg)` takes the first `n_structural` of them; `s0` (map-like)
and `s1` (fold-like) come first because they are the two that actually *consume* the atomic
table operators (`grammar.py:42-47`).

**Atomic operators** (the tables): `n_unary = 3` (`u0, u1, u2`) and `n_binary = 2` (`b0, b1`)
by default (`grammar.py:49-54`).

**Table sizes** (`tables.py:6-8`) — a table is a function, not a stored array:

| kind | signature | entries if you tabulated it | per instance |
|---|---|---|---|
| unary | `V → V` | ~4913 | × 3 ops ≈ 14,739 |
| binary | `V × V → V` | ~24 million (`4913²`) | × 2 ops ≈ 48 million |

The tables are **frozen random MLPs** — `FrozenMLP`, two-layer tanh, random weights, never
trained (`tables.py:38-56`). Two properties hold at once (`tables.py:10-24`):

- **structured** — digit embeddings are shared across all values, so values with similar digits
  get similar outputs. This is the *only* reason a learner can extrapolate to entries it never
  saw.
- **indescribable** — the weights are random, so "writing the function down" means transcribing
  the weight matrices. You cannot state a rule; you can only learn from examples.

Binary tables are digit-wise-dominant plus a weak global `mix` term weighted by
`binary_coupling` (default 0.25); pure digit-wise (`coupling = 0`) would have only
`n_digits · base²` distinct parts — small enough to enumerate — so the coupling is what keeps
binary genuinely hard (`tables.py:20-24, 89-100`). `IdentityTables` (`u(x)=x`, `b(x,y)=x`) is
the degenerate baseline used by the skeleton ceiling (`tables.py:261-274`).

## 4. Configuration — `GlyphConfig`

`GlyphConfig` is a frozen dataclass (`config.py:44-186`). Every field:

| Field | Default | Meaning |
|---|---|---|
| `base` | 17 | digits per position |
| `n_digits` | 3 | number of digit positions (`base**n_digits = n_values`) |
| `d_digit` | 16 | embedding dims per digit |
| `value_form` | `"letter_sep"` | how a value is rendered (§2) |
| `n_structural` | 5 | how many of `s0..s7` are enabled |
| `max_transform_depth` | 2 | how deep a `<transform> then <transform>` may nest |
| `guard_prob` | 0.5 | chance a structural op gets an `if/then/else` guard |
| `n_unary` | 3 | number of unary atomic operators |
| `n_binary` | 2 | number of binary atomic operators |
| `mlp_width` | 64 | hidden width of each `FrozenMLP` |
| `mlp_temp` | 1.0 | pre-activation scale (higher = less smooth) |
| `binary_coupling` | 0.25 | 0 = purely digit-wise; larger = harder binary tables |
| `unary_coupling` | 0.25 | weak global coupling on unary; `None` = single joint MLP |
| `decode` | `"whiten"` | how a real-valued MLP output becomes a legal symbol |
| `atomic_ratio` | 0.5 | **primary knob for sweeping π** (§5) |
| `binary_freq` | 0.35 | how often a sampled root wants a value (fold) vs a list |
| `max_expr_depth` | 4 | max expression nesting depth |
| `demo_max_depth` | 2 | depth cap for demos / `iid` / `comp` |
| `depth_stop_prob` | 0.15 | chance a recursion level stops early |
| `list_len_range` | (2, 4) | legal list-literal length |
| `n_demos` | 30 | in-context demo pairs |
| `n_iid` | 6500 | size of the `iid` test split |
| `n_comp` | 2300 | size of the `comp` test split |
| `n_depth` | 1200 | size of the `depth` test split |
| `n_val` | 5000 | validation split size |

Derived (`config.py:160-170`): `n_values = base**n_digits`, `n_test = n_iid+n_comp+n_depth`
(= 10,000 by default). `with_(**kw)` returns a modified copy; `scaled(n_test)` resizes the test
set keeping the 65/23/12 split proportions.

There is **no** `held_pairs` or op-count field on the config — held pairs are computed per
instance at generation time (§6), not set as a knob.

## 5. Presets and the π knob

Three presets are shipped (`config.py:198-279`); unlisted fields take the dataclass defaults.

| preset | `atomic_ratio` | `n_structural` | `max_transform_depth` | `guard_prob` | `max_expr_depth` | `demo_max_depth` |
|---|---|---|---|---|---|---|
| `pi_low` | **0.85** | 5 | 0 | 0.0 | 3 | 2 |
| `pi_mid` | 0.5 | 5 | 2 | 0.5 | 4 (default) | 2 (default) |
| `pi_high` | **0.15** | 8 | 3 | 0.9 | 5 | 3 |

**`atomic_ratio` is the primary continuous knob for sweeping π** (`config.py:128`). It biases
expression sampling toward operators that *consume the table* versus purely structural ones
(`instance.py:85-90`):

- **low `atomic_ratio` → high π** (`pi_high`, 0.15): expressions rarely reach for the table, so
  difficulty lives in the skeleton. All 8 structural ops, deep nesting, guards.
- **high `atomic_ratio` → low π** (`pi_low`, 0.85): expressions constantly touch the table, so
  difficulty lives in the tables. Near-trivial structural ops, no guards, no composition.

The secondary knobs move monotonically with `atomic_ratio` (skeleton complexity shrinks from
`pi_high` → `pi_low`), reinforcing the same axis.

**Important — nominal vs measured π.** The preset *name* carries the *intended* π; the real
value is whatever `measured_pi()` reports for that specific `(seed, preset)` and it **drifts**
(a `pi_mid` seed can measure into the `pi_high` band). Phase-diagram axes and the frozen
benchmark use the **measured** value, never the preset name (`config.py:190-192`). This is why
the 15 frozen instances were selected by measured π, not by preset (see the benchmark manifest).

## 6. Generation flow

`generate(seed, cfg)` just constructs `GlyphInstance(cfg, seed)` (`instance.py:499-500`). The
constructor runs these steps in order, all off **one** RNG (`instance.py:216-236`):

1. `rng = np.random.default_rng(seed)` — created **once**, then threaded through every step.
2. `skeleton = sample_skeleton(cfg, rng)` — one `StructSem` per enabled structural op, drawn
   from the finite transform grammar (`semantics.py:134-165`). Sampling from a *finite* grammar
   is what guarantees the skeleton is always describable in finitely many words.
3. `tables = Tables(cfg, rng)` — the frozen MLPs (§3).
4. `P = Interpreter(cfg, skeleton, tables)` — the reference interpreter, `P = skeleton ∘ tables`.
5. `held_pairs = _draw_held_pairs(cfg, rng)` — see below.
6. `demos = _make_demos(rng)` — 30 pairs at `demo_max_depth`, held pairs forbidden.
7. `test = _make_test(rng)` — the three splits (below).
8. `val = _make_val(rng)` — 5,000 items, drawn like demos but disjoint from demos+test.

Because a single `default_rng(seed)` is threaded sequentially through all of this (no other RNG,
no global random, no torch), the same `(seed, cfg)` reproduces the same instance **byte for
byte** — the basis for the fingerprint and for `load_instance` (§8).

### Test splits (`instance.py:432-435`)

| split | size | depth budget | held pairs | extra |
|---|---|---|---|---|
| `iid` | 6500 | `demo_max_depth` | forbidden | ordinary in-distribution |
| `comp` | 2300 | `demo_max_depth` | **required** | tests a held-out operator *composition* |
| `depth` | 1200 | `max_expr_depth` | forbidden | `min_depth = demo_max_depth + 1` — strictly **deeper** than any demo |

- **iid** — same constraints as the demos; ordinary items.
- **comp** — same shallow depth, but each item is *required* to contain a held-out (outer,inner)
  operator pair — does a specific composition generalize?
- **depth** — allowed to nest all the way to `max_expr_depth`, and forced strictly deeper than
  anything demonstrated — does generalization to unseen depth hold?

Each `TestItem` records `needs_u` / `needs_b`: exactly which `(op, i)` / `(op, i, j)` table
cells the *true* answer required, captured from the logged evaluation of the gold expression
(`instance.py:454, 264-268`).

### Held pairs (`instance.py:159-210`)

`held_pairs` is a set of (outer, inner) structural-op pairs excluded from the demo/iid/depth
distribution and *required* in `comp`. Only grammar-realizable pairs are candidates; the target
share is held to an exact ~30–33% by largest-remainder allocation, rather than drifting under a
naive Cartesian draw.

### Tail (`instance.py:339-348`)

`is_tail(t)` = "did this run never buy a table entry this item needs?" — it is computed against
the **live** query log of a *run*, not fixed at generation time, because "never seen" is only
knowable after the agent has done its probing.

## 7. π measurement

π quantifies **where the difficulty lives** — the skeleton's share of total difficulty.
`measure.py:1-18`:

```
a_skel = accuracy(true skeleton + identity tables)   # only the table is missing
a_tab  = accuracy(trivial skeleton + true tables)    # only the skeleton is missing
L_table = 1 - a_skel                                 # loss from not knowing the table
L_skel  = 1 - a_tab                                  # loss from not knowing the skeleton
π = L_skel / (L_skel + L_table)                      # the SKELETON's share
```

- **π → 1**: almost all loss traces to not knowing the *skeleton* (the table is easy/near
  trivial) → a "code/reasoning" instance.
- **π → 0**: almost all loss traces to not knowing the *table* → a "memorization/training"
  instance.

`measure_pi(inst, sample=1500)` (`measure.py:75-90`) measures on `inst.test[:1500]` (which at
full size is entirely `iid`) and returns `pi_components`:
`{full, a_skel, a_tab, L_table, L_skel, pi, n}`.

The π measurement (only) is scored with a lenient **digit-match** (fraction of digit positions
that agree, `measure.py:28-35`); the agent arms and the reference oracles are scored by **exact
match** (`measure.py:44-46`). There is **no p-value / significance test** anywhere in the
codebase — π is a point estimate, reported with its raw components so it can be recomputed.

## 8. Determinism and fingerprint

`fingerprint(inst)` (`frozen.py:13-23`) is a SHA-256 over four sections, in order: demos
(`expr, answer`), test items (`split, expr_src, answer_src`, in generation order), held pairs
(**sorted**, so pair-generation order can't perturb the hash), and val items
(`expr_src, answer_src`). It pins the instance's full identity.

`load_instance(id, verify=True)` (`frozen.py:60-85`) regenerates via
`generate(seed, PRESETS[preset])` and asserts the fingerprint matches the frozen manifest,
raising on drift. Because generation is single-RNG deterministic (§6), this is guaranteed to
reproduce the frozen data exactly — which is what makes the 15 benchmark instances reusable by
id without ever storing their (huge) expanded data.

---

*Companion: **Data Validation** — how arms and reference oracles are measured against an
instance.*

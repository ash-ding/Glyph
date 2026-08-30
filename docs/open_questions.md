# Open questions

Everything still to decide and everything still to run, with the measurement
each rests on. Written 2026-08-30, after the pass that froze most of the data
layer.

Three files, three jobs. `docs/progress.md` is the append-only record of what
was done and what it showed. This file is the **standing list** — it is edited
in place as items close, so it always reflects the present. The GitHub issues
(`#1`–`#29`) are one per item, for discussion and status.

Every number below is measured. Where a claim was later found wrong, the
correction is in `progress.md` under the entry that contains it.

---

## Settled — do not reopen without a reason

| | settled by |
|---|---|
| `decode = whiten` | reverified at 17³: 1163 distinct unary outputs over 4913 inputs, top output 2.1%, inside the range whiten was validated at |
| all three presets on `\|V\| = 17³`, `d_digit = 16`, `d_total = 48` | `4aa6c1f` |
| `binary_coupling = 0.25` | D2 in the original plan, option C+B |
| generation raises with a diagnosis rather than returning a short test set | `5fe521e` |
| held-out pairs drawn from realizable pairs, a fixed share per shape class | `8253c9a` |
| the skeleton-only and table-only ceilings ship with every score, plus `headroom` | `8253c9a` |
| `value_form = letter_sep` — four tokens per value, fixed width | `553abe5`, closes [#7](https://github.com/ash-ding/Glyph/issues/7) |

---

## Data layer

### [#1](https://github.com/ash-ding/Glyph/issues/1) — `n_structural` for `pi_low`: 3, 4 or 5 · P1

Raised from 2 to 3 because at 2 the generator failed on 13 of 20 seeds. 3 was
the smallest value that worked, not a choice on merit.

20 seeds each, full 10⁴, at the real `pi_low` config:

| n | π median | skeleton ceiling | lookups/item | comp ceiling | comp sd | possible held sets |
|---|---|---|---|---|---|---|
| 3 | 0.346 | 0.067 | 4.42 | [0.142, 0.167] | 0.0063 | **1** |
| 4 | 0.356 | 0.076 | 4.23 | [0.202, 0.476] | 0.0839 | **1** |
| 5 | 0.358 | 0.075 | 4.25 | [0.053, 0.767] | 0.2084 | 128 |

π, ceiling and lookups barely move. The only thing `n` changes is **what `comp`
is**. At `n ≤ 4` the stratified draw is deterministic — `STRUCT_SHAPES` is
`(UL, LB, L, KL, L, UL, KL, LB)` and the first four shapes are pairwise
distinct, so every `(outer shape, inner shape)` class holds exactly one pair and
nothing is left to choose. All 20 instances hold out the same pairs, making
`comp` a fixed probe: you cannot separate "this instance's comp is hard" from
"this composition is hard". Difficulty still varies (n=4's sd is 0.084) because
the skeleton's semantics differ by seed; the identity of the tested composition
does not.

**Recommendation: 5.** The cost it pays is comp's difficulty variance, and
`headroom` (#4) divides exactly that out. It also leaves `pi_low` differing from
`pi_mid` only in `atomic_ratio` and skeleton complexity.

*Counter-argument worth weighing:* if `pi_low`'s job is to be a pure table task,
`comp` matters least there, and a fixed probe is simpler.

### [#2](https://github.com/ash-ding/Glyph/issues/2) — is a preset a condition or a sampler?

Measured π, stratified, 20 seeds each:

```
pi_low   [0.11 0.12 0.17 0.18 | 0.38 0.41 0.44]
pi_mid   [0.36 … 0.58 | 0.61 0.63 0.65 0.66 0.66 0.68 0.72 0.76]
pi_high                       [0.62 0.65 0.66 … 0.84]
```

Seven of twenty `pi_mid` seeds fall inside `pi_high`'s range. Given an instance
at π = 0.70 you cannot say which preset made it.

`config.py` already takes the sampler position and the data supports it — the
union covers [0.12, 0.82] continuously. What is unsettled is whether it is
**stated**: the paper cannot say "three settings", only "N instances along a
continuous axis", and the grids should be organised by measured π rather than by
preset name.

Holding everything at `pi_mid`'s settings and moving only `atomic_ratio` sweeps
π from 0.765 to 0.358 monotonically, 42/42 generating, mode baseline steady:
one knob covers most of the axis where the presets move six.

Blocks nothing.

### [#3](https://github.com/ash-ding/Glyph/issues/3) — does π stay measured on `iid` only?

`measure_pi` scores `inst.test[:1500]`; at full size the first 6500 items are
all `iid`, so `comp` and `depth` have never entered a π measurement.

| preset | median \|iid-only − stratified\| | max |
|---|---|---|
| pi_low | 0.059 | **0.166** (seed 1014: 0.000 vs 0.166) |
| pi_mid | 0.017 | 0.049 |
| pi_high | 0.018 | 0.043 |

Per-split π explains it — on `pi_low` the skeleton contributes only on `comp`
(iid 0.000, comp 0.466, depth 0.000), the one split forced to use a held-out
composition. So this is a `pi_low`-only problem, and it survived the move to
three operators. An earlier note overstated it as general.

`measure.py` warns that π's two baselines are fixed once and never changed,
because changing them invalidates every previously measured π. Changing the
*sample* is close to the same act.

### [#4](https://github.com/ash-ding/Glyph/issues/4) — ceiling as covariate, normaliser, or both? · P1

`ScoreReport` now carries both ceilings, computed on the items actually scored,
plus `headroom = (score − ceiling) / (1 − ceiling)`. How the analysis uses them
is open.

pi_mid/1001, full test set, exact match:

| split | n | needs no table | true skeleton + identity table |
|---|---|---|---|
| iid | 6500 | 0.269 | **0.269** |
| comp | 2300 | 0.172 | **0.172** |
| depth | 1200 | 0.065 | **0.065** |
| all | 10000 | 0.222 | **0.222** |

The columns are *equal*: a true skeleton with an identity table answers the
items needing no lookup and only those. Against that line, on 200 items
(SE ≈ 0.03):

```
ceiling  0.222
A0'      0.255    +1.1 SE — indistinguishable
A4       0.255    the same
A2       0.055    far below
A6       0.035    far below
```

Without the line, "the frontier extracted no table knowledge" and "the frontier
did well" are the same number. Across 20 pi_mid seeds the overall ceiling ranges
**0.148–0.744**, more than the gap between any two arms.

`headroom` is `None` when the ceiling is 1.0, which happens — one pi_mid seed
has a `comp` split fully solvable by skeleton alone. That split measures
nothing and used to be averaged in silently.

### [#5](https://github.com/ash-ding/Glyph/issues/5) — match `comp`'s table demand to `iid`'s?

`8253c9a` fixed comp's **variance**, not its **level**.

| preset | comp−iid spread before | after |
|---|---|---|
| pi_low | sd 0.150 | sd 0.068 (−55%) |
| pi_mid | sd 0.255 | sd 0.117 (−54%) |
| pi_high | sd 0.140 | sd 0.050 (−64%) |

Medians moved +0.102 → +0.111 on pi_mid. `comp` is still systematically easier
than `iid` — the split meant to be the harder compositional test is where
knowing no table entry gets you furthest.

Mechanism, measured: `comp` is sampled by rejection until a held-out pair
appears, and that conditioning moves its operator mix off the one
`atomic_ratio` defines. Per item on pi_mid:

```
seed 1001   iid 2.74  comp 2.63    even
seed 1002   iid 1.80  comp 0.98    comp far easier
seed 1003   iid 2.27  comp 3.03    comp harder
seed 1004   iid 2.44  comp 3.12    comp harder
seed 1005   iid 2.51  comp 0.83    comp far easier
```

An earlier hypothesis — that held pairs with fewer table-consuming operators
make comp easy — was refuted before it was acted on (r = +0.254 / −0.237 /
−0.302, inconsistent in sign).

The fix is ~10 lines using the lookup count `eval_logged` already returns. The
reason to hold: it redefines `comp` from "compositional generalisation" to
"compositional generalisation at matched table demand", and would invalidate
every instance a third time.

### [#6](https://github.com/ash-ding/Glyph/issues/6) — sample expression depth? · P1

`_sample` stops only when the budget runs out, so depth **equals** budget and
`min_depth` is nearly a no-op. pi_mid's test set is two points: 8800 items at
depth 2, 1200 at depth 4, nothing at 3, zero variance inside `iid`.

`scripts/depth_sampling_probe.py`, pi_mid, 2000 items, 3 seeds:

```
stop=0.00   iid {2:1300}          depth {4:240}        ceiling 0.228  lookups 3.07
stop=0.15   iid {1:269, 2:1031}   depth {3:41,4:199}   ceiling 0.235  lookups 2.94
stop=0.30   iid {1:513, 2:787}    depth {3:82,4:158}   ceiling 0.259  lookups 2.76
stop=0.50   iid {1:803, 2:497}    depth {3:136,4:104}  ceiling 0.293  lookups 2.53
```

Generation does not slow and nothing fails. Depth diversity buys down the
discriminative range; at stop 0.15 that costs 0.007 of ceiling.

This is also half the explanation of the dev/test mismatch in #14: the agent
buys single-level probes because that is how you read a table, and **no test
item is single-level**.

**Recommendation: 0.15.** Measured, not implemented.

**`stop_prob` is not a config field.** It exists only inside
`scripts/depth_sampling_probe.py`, which copies `_sample`, adds the early stop,
and monkeypatches `glyph.instance._sample` for the duration of the run before
restoring it. Nothing in `src/` knows about it — deliberately, since the point
was to measure what changing the sampler would do before changing it.

Adopting this means three things: a field on `GlyphConfig` (say
`depth_stop_prob: float = 0.15`), one line in `_sample`'s `go()`, and
regenerating every instance.

### ~~[#7](https://github.com/ash-ding/Glyph/issues/7) — `value_form`~~ · settled

`letter_sep` (`v_k_e_e`), shipped in `553abe5`. Fixed at four tokens per value
against `underscore`'s 8.24 with a max of 10 — and the fixed width, not the
count, was the argument: base 17 has two-character digits 10–16, so a value's
token length used to correlate with its digit values.

Measured end to end after the switch, **smaller than the per-value ratio
implied** because operator and bracket scaffolding does not shrink: a 30-fact
prompt block 1650 → 1063 tokens (−36%), mean answer length 18.92 → 10.70
(−43%). So A2's deployment cost falls by about a third rather than half; the
"halves what the context arm pays" written here earlier was the per-value ratio
read as a per-query one.

Constraint taken on: `base ≤ 26`, which narrows #10.

### [#8](https://github.com/ash-ding/Glyph/issues/8) — `unary_coupling` · P1

Unary operators are one joint MLP; binary ones got a per-digit factorisation.
Self-check #5 found this makes **unary the harder half**, inverting the spec's
assumption.

| | per-entry reach |
|---|---|
| unary, joint (current) | 0.158 at 24k steps |
| unary, digit-wise (`unary_coupling=0.25`) | 0.597 |
| binary, c=0.25 | 0.734 |

`pi_low` needs 4.42 lookups per item and 100% of its items touch the table, so
item-level exact match is roughly `reach ** 4.4`:

```
reach 0.158  →  ~0.0006     pi_low is unsolvable
reach 0.597  →  ~0.11       pi_low is thin but measurable
```

All of #5's numbers were produced at lr 1e-5, which the LR sweep showed
understates the student by ~0.24 on binary. They are a floor.

### [#9](https://github.com/ash-ding/Glyph/issues/9) — test-set size and per-arm evaluation size · P1

Generating 10⁴ costs 1.5–11 s. Evaluating it costs:

```
A4   program in a sandbox            4.3 H100-s
A6   trained student, no prompt      25.8
A2   student + prompt                82.7
A0'  frontier re-reads the evidence  752,252    ← the only problem
```

So test-set size and evaluation size are two decisions. `scripts/subsample_power.py`,
synthetic arms correlated through shared table entries:

```
   n     single arm SE   independent   paired   paired+stratified
 100        0.0476         0.0625      0.0312        0.0306
 200        0.0334         0.0448      0.0221        0.0220
 500        0.0205         0.0274      0.0135        0.0136
1000        0.0142         0.0191      0.0095        0.0093
2000        0.0095         0.0128      0.0063        0.0063
```

At n = 200 a single arm's SE is 0.033 — **A6 at 0.035 and A2 at 0.055 are 7/200
and 11/200; there was never a difference there.** Every n = 200 comparison in
`progress.md` reads that way.

Pairing halves the SE on a difference and helps ~4.5× when arms nearly agree,
which is our case. Stratifying adds nothing on top. The reason to keep 10⁴ as
the population is per-split resolution: at 1000 total, `depth` gets 120 items
and an SE of 0.027 against a ceiling of 0.144.

**Recommendation:** generate 10⁴; score A2/A4/A6 on all of it; score A0' on a
fixed paired subset of 500–1000 from the same items.

### [#10](https://github.com/ash-ding/Glyph/issues/10) — digit layout

Open since the original plan's D3. Coupled to #8: more digit positions means
more lookups per item, and exact match compounds them.

| preset | lookups/item | item ceiling at reach 0.6 / 0.8 / 0.9 |
|---|---|---|
| pi_low | 4.21 | 0.17 / 0.42 / 0.65 |
| pi_mid | 3.08 | 0.39 / 0.58 / 0.75 |
| pi_high | 0.69 | 0.82 / 0.89 / 0.94 |

Decide together with #7 and #8; all three move the same quantity.

---

## Protocol

### [#11](https://github.com/ash-ding/Glyph/issues/11) — dead parameters in `synthesize_data`

Three of five parameters are accepted, traced, and never read:

| parameter | effect |
|---|---|
| `n` | how many rows to repeat to |
| `source` | queries / demos / mixture |
| `description` | **none** |
| `emphasis` (`uniform \| tail \| operator-focused`) | **none** |
| `include_reasoning` | **none** |

The agent filled `emphasis: "operator-focused"` on all three calls of the A6
rerun and wrote increasingly detailed descriptions — by the third it had derived
s2/s3/s4's permutation rules — believing it was designing a curriculum. It got
the same 511 rows at three lengths.

Worse than #16: there the agent describes itself, here it **operates a control
that is not connected**, and the trace records the intent as if it had effect.

Related: the tool repeats purchased rows rather than synthesising, which is
correct — the agent cannot manufacture labels it has not bought. A6's "50,000
examples" were 511 distinct rows at 98×, so `n` is epochs wearing a costume.

### [#12](https://github.com/ash-ding/Glyph/issues/12) — `filter_data` does not exist

The plan's §6.2 lists `propose_curriculum`, `synthesize_data`, `filter_data`.
Only the middle one exists, with its curation parameters inert (#11).

H3 predicts "the biggest lever is data curation, not target or
hyperparameters". **There is no curation lever on T1 to measure**, so H3 cannot
be tested as stated.

### [#13](https://github.com/ash-ding/Glyph/issues/13) — `evaluate` is gated behind the weights container · P1

| | A2 | A4 | A6 |
|---|---|---|---|
| tools | 3 | 3 | 7 |
| self-evaluation | **none** | `write_code(check_on=…)` | evaluate / inspect / retrain |
| what it did (rerun) | query ×5, set_context, seal | — | query ×11, synth ×3, train ×3, evaluate ×2, inspect ×2, seal |

A2 writes its prompt blind and seals. Since the dev signal strongly shapes
behaviour — once dev stopped reading zero the agent went from 187 purchased
facts to 565 — **A2's and A6's scores were not produced under equal feedback**.

Whether an arm can score itself is a harness choice, not a property of the
container. **Recommendation: move both out of the gate.**

### [#14](https://github.com/ash-ding/Glyph/issues/14) — dev does not estimate test · P1

`DEV_FRACTION = 0.15` of the purchased log becomes dev. A6's rerun reached
**dev 0.400, test 0.035**:

| | nesting depth | single-level expressions |
|---|---|---|
| what the agent bought | median 1, max 2 | **92%** |
| the test set | median 2, max 4 | **0%** |

A clearer dev signal made the agent optimise harder in the wrong direction.
Partly a consequence of #6. Arguably real rather than a defect — a narrow
probing strategy *should* yield a misleading self-estimate, and calibration is
worth measuring — but right now it is neither controlled nor reported.

### [#15](https://github.com/ash-ding/Glyph/issues/15) — the query cap Q · P1

Under the A+B split the main figure's axes are (π, Q), so Q must be a set
variable. It currently drifts between 103 and 472 across runs.

The earlier argument for a cap was **retracted**: it rested on "the agent only
buys ~200 facts", which was an artefact of the truncation bug. A cap may still
be right; the reasoning has to be rebuilt.

Query spend never exceeded 0.064% of the budget across 8 runs, and pi_mid's
test set needs ~10⁴ distinct unary entries. Q is a coverage knob, not a cost
knob.

### [#16](https://github.com/ash-ding/Glyph/issues/16) — `declare_target`'s roles are declarative

`_t_synthesize_data` never reads `role`; `_t_train` uses it only to check one
was declared. **All seven roles train identically**, so "the agent chose
`world_model`" is a fact about its self-description. Bears on H3 and Fig. 4.

### [#17](https://github.com/ash-ding/Glyph/issues/17) — implement A7?

`seal.py` allows a context *and* an adapter and `Student` carries both, but
there is no A7 module and `cli.py` knows only `a2`, `a4`, `a6`. A7 is where Q2
lives ("when the agent chooses, how good is that choice?"), and Q2 depends on
Q1 — if the containers are interchangeable, "chose well" means nothing.

### [#18](https://github.com/ash-ding/Glyph/issues/18) — update the Phase 1 artifact?

H1 still reads "a budget threshold B* exists" and E1 is organised around a B
sweep. A+B rewords it to "there is a region where only weights succeeds", with
T2 carrying amortisation. Untouched because the artifact is the source of truth
for three documents and A+B is recorded as tentative.

---

## Experiments

### P1 — decides whether the benchmark is usable

**[#19](https://github.com/ash-ding/Glyph/issues/19) — a trained adapter on the full test set.** Self-check #5 measures per-entry
reach; nothing measures what that implies for whole items. Predicted item
ceiling at reach 0.80: pi_mid 0.58, pi_low 0.42. Take an adapter #5 already
trained and compare. ~1 GPU-hour. If measured is far below predicted, the
compounding model is missing something and the usable band is smaller than
believed.

**[#20](https://github.com/ash-ding/Glyph/issues/20) — rerun A0' on a paired subset.** A0' at the ceiling is the strongest
form of H1 and rests on 200 items. ~75 H100-s per item, so the subset size is
#9's decision. Blocked on the data layer being frozen.

### P2 — invalidated, must be redone

**[#21](https://github.com/ash-ding/Glyph/issues/21) — self-check #4 (hiddenness).** Every preset changed; `pi_high` in
particular is a different task now. Costs teacher calls; last full run ~110
minutes.

**[#22](https://github.com/ash-ding/Glyph/issues/22) — self-check #5 (capacity).** Presets changed, and every full fine-tune
number was at lr 1e-5:

| binary c=0.25, 4000 steps | lr 1e-5 | lr 1e-4 |
|---|---|---|
| full fine-tune | 0.562 | **0.797** |
| LoRA r=32 | 0.556 | 0.715 |

**[#23](https://github.com/ash-ding/Glyph/issues/23) — E0's four arms.** Invalidated twice: by the preset change and by the
held-pair change (which touches every instance's `comp`, pi_mid included). And
at n = 200 the differences were never resolvable. **Do not rerun until the data
layer is frozen** — any of #1, #5, #6, #7, #8 would make it a third time.

### P3 — not started

- **[#24](https://github.com/ash-ding/Glyph/issues/24)** `unary_coupling` × `value_form`, matched settings, better LR
- **[#25](https://github.com/ash-ding/Glyph/issues/25)** variance decomposition. `instance_seed` and `run_seed` are separate
  on purpose and the split has never been measured. Instance variance is large
  (ceiling 0.148–0.744 across 20 pi_mid seeds); run variance is unknown
- **[#26](https://github.com/ash-ding/Glyph/issues/26)** the phase diagram. Also: `worker.py`'s grid timeout is 5400 s and A6
  runs span ~10 to >90 minutes, so a timeout that clips the tail biases toward
  agents doing *less* of what the experiment is about
- **[#27](https://github.com/ash-ding/Glyph/issues/27)** prompt caching. Specified in plan §6.3, never implemented, zero cache
  hits across eight runs. `declare_target` — one value from a seven-item enum —
  cost 311 H100-s, within 15% of the `train` call in the same run. ~20 lines
- **[#28](https://github.com/ash-ding/Glyph/issues/28)** T2. Zero lines, and under A+B it carries half the paper. Glyph cannot
  carry amortisation: preparing costs ~100× serving, so the rational choice is
  always "don't train, just answer"

---

## Order

E0 has been invalidated twice. Any of #1, #5, #6, #7, #8 would make it three
times.

```
1. settle #1 #6 #7 #8        CPU only, or data already in hand
2. run #19                   confirm there is a measurable band at all
3. settle #9 #4              evaluation size, and how scores are reported
4. run #21 #22, then #23     one pass, not three
5. settle #11–#18            then touch the arms
```

#2, #3 and #18 concern how results are written up rather than produced. They
block nothing and can go last.

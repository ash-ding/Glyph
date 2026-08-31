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
| digit layout stays `17³` | closes [#10](https://github.com/ash-ding/Glyph/issues/10) |
| a preset is a sampler; every report carries the instance's measured π | closes [#2](https://github.com/ash-ding/Glyph/issues/2) |
| `depth_stop_prob = 0.15` — depth is sampled, not the split's budget restated | closes [#6](https://github.com/ash-ding/Glyph/issues/6) |
| both ceilings ship on every scored subset; `headroom` is the primary figure | closes [#4](https://github.com/ash-ding/Glyph/issues/4) |
| `pi_low` gets five structural operators, so `comp` is a sample not a fixed probe | closes [#1](https://github.com/ash-ding/Glyph/issues/1) |
| `unary_coupling = 0.25` — widest arm gap on every preset, measured at the item level | closes [#8](https://github.com/ash-ding/Glyph/issues/8) |

---

## Data layer

### ~~[#1](https://github.com/ash-ding/Glyph/issues/1) — `n_structural` for `pi_low`~~ · settled

**5.** Neither of the two things this number decides is difficulty.

It started at 2, which could not generate. Filtering the held-pair draw to
realizable pairs fixed that at any n ≥ 3. What 5 buys is that `comp` becomes a
**sample** rather than a fixed probe: `STRUCT_SHAPES` is
`(UL, LB, L, KL, L, UL, KL, LB)`, the first four shapes are pairwise distinct,
so below 5 every `(outer shape, inner shape)` class holds exactly one pair and
the stratified draw has nothing to choose. Over 40 seeds — 1 distinct held set
at n = 2, 3 and 4; **35 at n = 5**; 40 at n = 6.

π does not pay for it. Ten seeds at full size, n = 3/4/5: median π 0.344 / 0.302
/ 0.332, ceiling 0.067 / 0.077 / 0.054, lookups 4.36 / 4.32 / 4.25 — noise, and
n = 4 is the lowest of the three. The column that moves is comp's spread across
seeds (sd 0.007 / 0.103 / 0.228), which is the variation being restored rather
than a cost — and is what `headroom` exists to divide out.

Re-measured under the current sampler; the earlier sweep predated
`depth_stop_prob` and the unary rebuild.

Side effect: `pi_low` and `pi_mid` now share `n_structural` and differ only in
`atomic_ratio`, transform depth, guard probability and expression depth.

### ~~[#2](https://github.com/ash-ding/Glyph/issues/2) — condition or sampler?~~ · settled

**A sampler.** Seven of twenty `pi_mid` seeds fall inside `pi_high`'s measured
range, so an instance at π = 0.70 cannot be attributed to a preset; the union
covers [0.12, 0.82] continuously, which is what the phase diagram wants.

The position was already in `config.py`. What this issue was really hiding is
that it was **unenforceable**: `measured_pi` was never recorded in a run report,
so every report said `preset: pi_mid` and nothing about the instance's actual π.
Plotting against π would have meant regenerating each instance afterwards and
trusting the config had not moved. The stated axis did not exist in the data.

`ScoreReport.instance` now carries `{seed, preset, pi, n_structural,
atomic_ratio}`, with `pi` being the whole `measure_pi` dict rather than the
ratio alone — so if [#3](https://github.com/ash-ding/Glyph/issues/3) changes
which items π is measured on, the new value is a recomputation from `a_skel`
and `a_tab` rather than a rerun. `measured_pi()` caches (0.22 s once).

Commits the paper to describing N instances along a continuous axis, not three
settings. Figures key on `instance["pi"]["pi"]`.

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

### ~~[#4](https://github.com/ash-ding/Glyph/issues/4) — ceiling as covariate or normaliser~~ · settled

**Both**, with `headroom` primary and raw scores alongside. The gap this closed
was `tail`, the one number in the report with nothing to read it against; it now
has its own ceiling.

**Correction.** This entry claimed the skeleton ceiling equals the fraction of
items needing no lookup "and not approximately". That held at pi_mid seed 1001
and is not general — equality on 3 of 5 seeds, exceeded by 7% and 9% on the
others. `needs_u`/`needs_b` record entries **touched**, not entries the answer
**depends on**: on seed 1002, `s0` is `map_skip(j=1) → dedup → rotate(1)` and
the outer `s3` keeps only the element the map skipped, so two lookups happen and
neither matters — the identity-table oracle answers it too.

That makes **`tail` over-inclusive**: an item counts as tail when it touched an
unbought entry, even where the answer did not depend on it, and its near-zero
ceiling measures exactly that over-inclusion. Whether `tail` should instead mean
"the answer changes without these entries" is a protocol question — one extra
evaluation per item, so cheap, but load-bearing for H1 — and is **not settled**.

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

### ~~[#6](https://github.com/ash-ding/Glyph/issues/6) — sample expression depth~~ · settled

`depth_stop_prob = 0.15`. Depth used to *equal* the split's budget, which made
`min_depth` nearly a no-op and left pi_mid's test set as two points. Measured on
2000 items: `iid` {2: 1300} → {1: 238, 2: 1062}, the depth split {4: 240} →
{3: 45, 4: 195}, skeleton ceiling 0.228 → 0.235. Generation does not slow.
`comp` stays pinned at depth 2, since a held-out pair needs two levels.

**Side effect worth knowing.** Depth-1 expressions are now reachable and carry
*no adjacent operator pair at all*, so every held-pair constraint passes over
them in both directions. Holding out every realizable pair used to empty the
demos and now empties `depth` instead. `iid` has gained a population the
held-pair machinery cannot touch.

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

### ~~[#8](https://github.com/ash-ding/Glyph/issues/8) — `unary_coupling`~~ · settled

**0.25 stays** — the value that was already there, for a different reason than
it arrived with, and against my own CPU-based recommendation.

Measured on GPU, lr 1e-4, 4000 steps, on the current tables:

| task | coupling | fit | reach |
|---|---|---|---|
| unary | **0.25** | 1.000 | **0.710** |
| unary | 0.5 | 1.000 | 0.505 |
| unary | 1.0 | 1.000 | 0.439 |
| binary | 0.25 | 0.858 | 0.857 |

**The 37% realisation rate was wrong** — the student realises 74–86% of the CPU
upper bound. The old figure came from the joint-MLP setting *at lr 1e-5*, and I
had folded the learning rate into what I read as a student-versus-bound ratio.
Realisation is stable enough that the CPU probe is a usable predictor.

**Reading it per-entry was the mistake.** The CPU sweep said the gap peaks near
coupling 1.0; that was per table entry. At the level items are scored, every
lookup has to be right, so compounding pushes both arms toward the floor at high
coupling and the gap closes. From each preset's real lookup distribution:

| | coupling 0.25 | 0.5 | 1.0 |
|---|---|---|---|
| pi_low, weights / code / gap | 0.398 / 0.219 / **+0.179** | 0.205 / 0.093 / +0.112 | 0.164 / 0.045 / +0.119 |
| pi_mid, weights / code / gap | 0.502 / 0.370 / **+0.133** | 0.360 / 0.283 / +0.077 | 0.331 / 0.253 / +0.079 |

0.25 wins on every preset by keeping the absolute level high, which after
compounding matters more than a wide gap between two numbers near zero.

**pi_low is measurable after all.** I projected an item-level ceiling of 0.006
and called it unsolvable; measured it is **0.398** — 66× out, from the wrong
realisation rate compounded with taking a power of the mean instead of the mean
of the powers. It is now the preset where the weights arm leads by the most.

Checkpoints kept at `~/glyph_runs/cap5/*.model` with their table config, so
[#19](https://github.com/ash-ding/Glyph/issues/19) does not retrain.

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

### ~~[#10](https://github.com/ash-ding/Glyph/issues/10) — digit layout~~ · settled

`17³` stays. Both first-order constraints hold comfortably and nothing else
improved either without a cost elsewhere:

| | 17³ *(kept)* | 8¹ *(replaced)* |
|---|---|---|
| distinct unary entries the test set needs | **11,387** | 24 |
| mode-answer baseline | 0.003 | 0.099 – 0.134 |
| lookups per item | 2.98 | 0.69 |
| largest share of random decode queries | 0.009 | 0.480 |

The table is not buyable, guessing the most common answer is worth nothing, and
decode does not pile onto a handful of values. `letter_sep` also needs
`base ≤ 26`, which 17 satisfies.

**Residue, and it moved rather than closed.** Digit-embedding separation is set
by `base / d_digit`: 17 points in 16 dimensions scores 0.707 nearest-over-mean
with its worst digit at 0.493, against 0.814 / 0.705 for the 8-in-24 layout it
replaced; `d_digit = 32` recovers it (0.784 / 0.709). Second-order — decode is
dominated by candidate count, where 17³ wins by a wide margin — but real, and it
is now a question about `d_digit`, not about the layout. Not tracked as its own
issue yet.

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

**Q also decides how clean `tail` is** — noticed while settling #4, and it makes
Q load-bearing for H1 rather than only for the figure's axis. `tail` selects
items that *touched* an unbought entry, not items whose answer *depends* on one,
so it over-counts. Measured on pi_mid, 4000 items, 5 seeds, treating the first N
as purchases:

| N purchased | contamination (median) | worst seed |
|---|---|---|
| 100 | 0.002 | 0.074 |
| 500 | **0.006** | 0.081 |
| 1000 | 0.012 | 0.088 |
| 2000 | 0.018 | **0.141** |

At the volume runs reach — a few hundred — 0.6% sits an order of magnitude below
the 1–3 point differences the arms will be separated by, which is why #4 records
the defect rather than fixing it. But it grows monotonically with Q and the
spread grows faster than the median, because buying more pushes tail membership
onto the margin: items missing one or two entries, where the chance those are
the discarded ones is highest. **A high Q buys coverage and pays in the
credibility of the split that carries H1.** Every run now carries
`ceiling["skeleton"]["tail"]`, which is this contamination measured on that run
rather than estimated from five seeds.

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

**~~[#19](https://github.com/ash-ding/Glyph/issues/19)~~ — the benchmark's real
ceiling. Done.** The answer is a curve, not a number: it depends on how much of
the table the student saw, which nobody had varied. pi_mid/1001, 500-item paired
subset, `scripts/table_ceiling.py` — true skeleton over student-learned tables,
so everything below 1.0 is the table.

| entries seen | unary reach | item score |
|---|---|---|
| 90% | 0.767 | 0.878 |
| 50% | 0.737 | 0.724 |
| 25% | 0.663 | 0.582 |
| **10%** *(agent volume)* | 0.605 | **0.498** |
| 5% | 0.531 | 0.454 |
| 2% | 0.279 | 0.360 |
| skeleton ceiling | — | *0.248* |

Extrapolation decays far more slowly than supervision — 98 entries still get
27.9% of the other 98% right. **The band the arms operate in is [0.248, 0.498]**,
25 points wide against the 1–3 point differences #9 expects. A6's 0.035 is now
readable: below the 0.248 that needs no table at all, against a reachable 0.498.

`fit` is 1.000 at every point including 98 entries — capacity is never the
constraint. Binary is inert to `seen_frac` (24M entries, always purely
extrapolating) so the curve's shape is unary's. `depth` collapses fastest,
0.817 → 0.217, being the most lookup-heavy split.

My projection of 0.502 was wrong twice in the same direction: it treated every
lookup as extrapolation, and it assumed lookup errors are independent when they
are correlated. **The independence model is a lower bound, not an estimate.**

**~~[#20](https://github.com/ash-ding/Glyph/issues/20)~~ — A0' on a paired
subset. Done.** Same 500 items as #19. Evidence deliberately more generous than
any arm could buy: 2000 queries revealing **2551 of 4913** unary entries, 70k
tokens, unlimited thinking.

| split | A0' | skeleton ceiling | headroom |
|---|---|---|---|
| overall | **0.258** | **0.248** | **+0.013** |
| depth | 0.067 | 0.083 | −0.018 |
| tail | 0.016 | 0.000 | +0.016 |

**A0' sits on the line that requires no table knowledge at all.** The split that
says why: items whose needed entries were *in* the evidence, 126 of them, score
**0.976**; items needing entries that were not, 374 of them, score **0.016**.
Retrieval saturated, extrapolation at zero.

Against the weights arm on the same items: frontier 0.258 with 2551 entries
given and unlimited thinking, against 0.498 for a 1.7B student that saw ~491.
Five times the table, hundreds of times the model, 24 points lower.

This settles H1 in its strong form — the advantage is learning-algorithmic, not
economic — and confirms the data layer's foundation, since A0' failing from
partial observation is the direct test of "learnable but not
in-context-extractable". The freeze stands.

### P2 — invalidated, must be redone

**[#21](https://github.com/ash-ding/Glyph/issues/21) — self-check #4 (hiddenness).** Every preset changed; `pi_high` in
particular is a different task now. Costs teacher calls; last full run ~110
minutes.

**~~[#22](https://github.com/ash-ding/Glyph/issues/22)~~ — self-check #5
(capacity). Done.** On the current tables, lr 1e-4, 4000 steps:

| task | coupling | fit | reach |
|---|---|---|---|
| unary | **0.25** | 1.000 | **0.710** |
| unary | 0.5 | 1.000 | 0.505 |
| unary | 1.0 | 1.000 | 0.439 |
| binary | 0.25 | 0.858 | 0.857 |

`fit` is 1.000 everywhere, so capacity is never the constraint. The failure this
check exists to catch — reach collapsing to chance, leaving the π→0 end with no
winner — did not happen.

Both questions the added axis asked came back inverted: the 37% realisation rate
was itself wrong (contaminated by lr 1e-5; actually 74–86%), and the CPU upper
bound does predict GPU reach but **only read at the item level** — per-entry it
pointed the opposite way. See [#8](https://github.com/ash-ding/Glyph/issues/8).

The `value_form` and digit-layout axes it expected to sweep collapsed when #7
and #10 settled; the holdout question it did not think to ask was answered more
thoroughly by [#19](https://github.com/ash-ding/Glyph/issues/19)'s `seen_frac`
curve. Not redone: LoRA versus full, which does not depend on the presets.

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

# Progress log

The running record of the work: milestones, self-checks that passed or failed,
decisions that moved, problems found and problems still open.

Entries are appended as the work happens and carry what was actually run — the
command, the config, the numbers — not only what was concluded. A conclusion
whose evidence is missing has to be re-run before it can be trusted, which is
the expensive way to find out.

---

## Current state

| self-check | state | where |
|---|---|---|
| #1 round-trip | pass | `pytest -q` |
| #2 oracle exactness | pass — 1.000 on every preset | `pytest -q` |
| #3 π ordering | pass — 0.25 / 0.43 / 0.69 | `pytest -q` |
| #4 hiddenness | **not run** — needs the teacher on a demos-only prompt | — |
| #5 capacity | **pass**, 2026-08-27 | `scripts/capacity_check.py` |
| #6 tokenizer probe | **pass**, 2026-08-27 | `scripts/probe_tokenizer.py` |

Decisions: D1 → B, D2 → C+B (**now verified**, see 2026-08-27 #5), D8 → A with
`value_form` **still open between `underscore` and a proposed `letter_sep`**.
D3 (digit layout) remains open; #5 did not separate the candidates decisively.

Open, in rough order of how much they matter:

1. **The unary tables have no digit-wise structure and the binary tables do.**
   Surfaced by #5, not by design. Unary reaches 0.46 on unseen values where
   binary reaches 0.80. This inverts the assumption written into the spec.
   Nothing is broken, but which half carries the π=0 difficulty is not what
   the documents say. See 2026-08-27 #5.
2. **`value_form`**: `letter_sep` costs half as many student tokens as
   `underscore` and is fixed-width, but needs a codec change and shifts the
   context arm's economics. Deferred pending a decision.
3. **D3**: 8⁴ and 17³ are within noise of each other; 10⁴ is handicapped by a
   larger value space at equal training budget. No winner yet.
4. `count_tokens` is blocked by org policy on this Vertex project; token
   accounting for the ledger has to come from `usage` on real requests.
5. Automatic prompt caching is unavailable on Vertex (explicit `cache_control`
   works). The ledger's caching discounts must not assume the automatic path.

---

## 2026-08-27 — environment on lumen-3

Dedicated conda env, deliberately **not** sharing `verl_discover`:

```
conda create -n glyph python=3.11        # 3.11.16
pip install -e ".[dev,agent]"
pip install torch transformers peft accelerate datasets safetensors
```

| | version |
|---|---|
| torch | 2.13.0+cu130 |
| transformers | 5.16.1 |
| peft | 0.20.0 |
| numpy | 2.4.6 |
| anthropic | 1.1.0 |

Hardware: 8 × H100 80GB, idle. Driver 595.71.05.

Baseline reproduced in the new env: **36 passed in 129.94s**. (The prior
environment reported 4m09s for the same suite.)

**Mis-step worth recording.** The first install ran through
`nohup bash -c "pip install ..."` after `conda activate glyph`; the activation
did not survive into the subprocess and the whole stack landed in base
(Python 3.14, `cp314` wheels). Caught by `ModuleNotFoundError: No module named
'torch'` inside the glyph env immediately afterwards. Fixed by calling the
env's pip by absolute path (`~/miniforge3/envs/glyph/bin/pip`) and uninstalling
the stray packages from base. If a background install has to be launched again,
use the absolute interpreter path, not an activation.

---

## 2026-08-27 — the teacher runs on Vertex

`src/glyph/vertex.py`, commit `468d918`.

Opus 5 is **not available** on this Vertex project, so Opus 4.8 is the main
teacher and Opus 4.6 the cross-replication teacher (E7). Endpoint config, read
from the host rather than hard-coded:

```
project_id  itpc-ca-273afd9a9f      (ANTHROPIC_VERTEX_PROJECT_ID)
region      global                  (CLOUD_ML_REGION)
auth        GCP ADC — no Anthropic API key
```

The `vertex_claude.py` this was based on pointed at a different project
(`itpc-gcp-ai-eng-claude`), a different region (`us-east5`), and Opus 4.6.
All three were wrong for this host.

Probe (`python -m glyph.vertex`) — **4/4 reachable**:

```
claude-opus-4-8    teacher              OK  in=16 out=4
claude-opus-4-6    cross-teacher (E7)   OK  in=14 out=4
claude-sonnet-4-5  ladder (E9)          OK  in=43 out=50
claude-haiku-4-5   ladder (E9)          OK  in=43 out=42
```

Both E9 ladder models work, so the capability ladder is runnable here and not
just the main teacher.

**The one that would have failed silently.** On Opus 4.8 and 4.6, omitting
`thinking` means the model does not think at all, and nothing errors. The
original file never passed it. Copied as-is, every arm would have run against a
quietly weaker teacher — visible in the results, invisible in the logs.
`request_kwargs` now always sets `{"type": "adaptive"}`, keeps `effort` at one
fixed level (`high`, a controlled variable swept only in E7), and branches to
`budget_tokens` for the 4.5-generation ladder models, which reject `effort`.

---

## 2026-08-27 — self-check #6: tokenizer probe

**Result: pass.** But the first two runs were wrong in opposite directions, and
both wrongs were in the criterion, not the notation.

### Pass 1 — criterion too strict, verdict impossible

The shipped probe asked that every digit land inside its own single token.
Qwen3 splits numbers one token per character, so a base-17 digit like `13`
occupies two tokens however it is written. Every candidate failed:

```
flat        v802        4 tok  fail  ['v','8','0','2']
underscore  v_3_13_2    8 tok  fail  ['v','_','3','_','1','3','_','2']
bracket     v[3,13,2]   9 tok  fail  ['v','[','3',',','1','3',',','2',']']
dot         v3.13.2     7 tok  fail  ['v','3','.','1','3','.','2']
spaced      v 3 13 2    8 tok  fail  ['v','Ġ','3','Ġ','1','3','Ġ','2']

NO CANDIDATE PASSES -- reopen D8.
```

That verdict would have reopened a decision that was never in trouble. The
criterion cannot pass at any base above 10 regardless of notation.

Two further problems in the shipped script: it hard-coded token strings rather
than calling `render_value`, and it offered `dot` and `spaced`, which the codec
cannot produce (`VALUE_FORMS` has only `underscore`, `bracket`, `flat`).

### Pass 2 — rewritten against `render_value`, still too strict

Rewritten to use the tokenizer's own character offsets against the real
renderer, with the criterion "no token may straddle a digit boundary". Verdict
on Qwen3-1.7B, six sample values:

```
17^3  underscore  7.8 tok  PASS      10^4  underscore  9.0  PASS
17^3  bracket     8.8 tok  PASS       8^4  underscore  9.0  PASS
17^3  flat          -      fail       7^4  underscore  9.0  PASS
```

Committed as `145dd00`. Two things were still wrong.

### Pass 3 — exhaustive, and the criterion corrected again

**Coverage.** Six sample values is not evidence: tokenisation is
content-dependent. Re-run over *every* value (4913 / 10000 / 4096). The
pass/fail verdicts survived, but the cost estimate did not — 17³ `underscore`
is **8.24** tok/value exhaustively, against 7.8 from the six-value sample.

**Criterion, second error.** `letter_sep` (digits as letters, `v_h_a_a`) was
being failed because the token `_a` "straddles". But `_a` spans a separator and
*one* digit — it smears nothing. It is one token per digit, which is the
cleanest possible signal. What actually breaks the design is a token covering
**two or more** digit fields, which is what `letter_flat`'s `aaa` does. Criterion
corrected to count overlapped fields and fail at > 1.

Full result, every value, Qwen3-1.7B:

| layout | form | checked | tok/value | verdict |
|---|---|---|---|---|
| 17³ | underscore | 4913 | 8.24 | PASS |
| 17³ | bracket | 4913 | 9.24 | PASS |
| 17³ | **letter_sep** | 4913 | **4.00** | PASS |
| 17³ | letter_flat | 4913 | 2.32 | fail — `'vaaa'`: `aaa` covers 3 digits |
| 17³ | flat | — | — | fail — renders the decimal index |
| 10⁴ | underscore | 10000 | 9.00 | PASS |
| 10⁴ | letter_sep | 10000 | 5.00 | PASS |
| 8⁴ | underscore | 4096 | 9.00 | PASS |
| 8⁴ | letter_sep | 4096 | 5.00 | PASS |

`letter_sep` at 17³ is **exactly 4.00 tokens for every value** — `v`, `_h`,
`_a`, `_a`. Fixed width, one token per digit.

`flat` fails for a structural reason, not a tokenisation one: `render_value`
emits the decimal index (`v802`), not the digits, so outside base 10 the
characters have no relation to digit positions — and even at base 10 it drops
leading zeros, so the field count varies with the value.

### Teacher side

Never measured before this. `count_tokens` returns **400** on this project:

```
Organization Policy constraint constraints/vertexai.allowedModels violated
... attempting to use a disallowed Gen AI model count-tokens
```

Measured instead from `usage.input_tokens` on real requests (envelope = 6 tok,
subtracted; 300 values per call, Opus 4.8):

| form | student (Qwen3-1.7B) | teacher (Opus 4.8) | fixed width |
|---|---|---|---|
| `underscore` ← current default | 8.24 | 8.00 | no |
| `bracket` | 9.24 | 8.00 | no |
| `letter_sep` | **4.00** | **7.00** | yes |

### Where #6 leaves things

`flat` is out. `underscore` and `bracket` are both sound. `letter_sep` is
cheaper on both sides — halving the student cost — and fixed-width, but needs a
codec change (`grammar.py`: `render_value`, `parse_value`, `VALUE_FORMS`).

The token cost is not neutral: the context arm re-pays it on every one of 10⁴
test queries, so halving it roughly halves A2's per-query cost and **weakens
the weights arm's economic advantage**. That is the honest direction — the plan
already refuses to handicap A2 by disabling its prefix cache, and a wasteful
notation would be the same kind of straw man. Deferred pending a decision.

---

## 2026-08-27 — self-check #5: capacity check

**Result: pass. The student is not mis-calibrated; no design change is forced.**

This is the check whose failure would have meant the student choice was wrong
and every downstream curve was measuring capacity instead of delegation.

### How it was run

New script, `scripts/capacity_check.py` — the repo had none. It trains directly
on `Tables`, so it needs no benchmark instance.

```
8 runs, one per GPU, in parallel
Qwen/Qwen3-1.7B, full fine-tune, bf16, gradient checkpointing
--steps 4000 --batch 128 --lr 1e-5  (OneCycle, 3% warmup)
512,000 training examples per run; held-out = 1 entry in 10, never trained on
--eval-n 1000, greedy decode, exact match + digit-level partial credit
config: PRESETS["pi_low"] tables, value_form=underscore
11–15 minutes per run
```

Two numbers per run, because "can it fit" and "can it extrapolate" are
different questions: **fit** = accuracy on trained entries, **reach** =
accuracy on entries never seen. `reach` is the one the design depends on.

### Results

| run | task | layout | coupling | fit exact | fit digit | **reach exact** | reach digit | chance |
|---|---|---|---|---|---|---|---|---|
| `u_b17` | unary | 17³ | — | 1.000 | 1.000 | **0.464** | 0.776 | 0.0002 |
| `u_b8` | unary | 8⁴ | — | 1.000 | 1.000 | **0.424** | 0.804 | 0.00024 |
| `b_b17_c000` | binary | 17³ | 0.00 | 1.000 | 1.000 | **1.000** | 1.000 | 0.0002 |
| `b_b17_c025` | binary | 17³ | 0.25 | 0.824 | 0.937 | **0.799** | 0.930 | 0.0002 |
| `b_b17_c050` | binary | 17³ | 0.50 | 0.652 | 0.867 | **0.622** | 0.856 | 0.0002 |
| `b_b17_c100` | binary | 17³ | 1.00 | 0.413 | 0.744 | **0.402** | 0.741 | 0.0002 |
| `b_b8_c025` | binary | 8⁴ | 0.25 | 0.869 | 0.966 | **0.845** | 0.958 | 0.00024 |
| `b_b10_c025` | binary | 10⁴ | 0.25 | 0.779 | 0.941 | **0.577** | 0.878 | 0.0001 |

Raw JSON in `~/glyph_runs/cap5/*.json` on lumen-3.

### D2 is verified

`binary_coupling` behaves exactly as designed, and the sweep is clean and
monotone:

```
coupling   0.00    0.25    0.50    1.00
reach      1.000   0.799   0.622   0.402
```

- **`coupling = 0` is confirmed unusable.** Reach is a perfect 1.000 — pure
  digit-wise is not merely learnable, it is *solved*, and with only
  `base² × n_digits = 867` parts an agent can enumerate it into a lookup table
  and hand the win to the code arm. The argument for rejecting it was
  theoretical; it is now measured.
- **`coupling = 0.25` is a good operating point**: reach 0.799, well learned,
  and the coupling term defeats the 867-entry enumeration.
- **The fear at the far end did not materialise.** D2 worried that a large
  `alpha` would leave the student unable to learn. At `alpha = 1.0` reach is
  still 0.402 — 2000× chance. The knob is safe across its whole range; the
  binding constraint is only on the low side.

### The finding that was not expected

**Binary extrapolates roughly twice as well as unary**, which is backwards from
what the spec assumes.

```
unary  17³           reach 0.464     fit 1.000
binary 17³ c=0.25    reach 0.799     fit 0.824
```

The cause is in `tables.py` and it is a design asymmetry, not a bug:

- **Binary is factorised.** `BinaryOp.per_digit[k]` maps
  `(digit_i[k], digit_j[k]) → digit_out[k]` — only `17² = 289` combinations per
  position, 867 in total. Across 512k training examples each part is seen ~590
  times, so the structure is learned thoroughly and unseen *pairs* compose from
  parts already known.
- **Unary is not.** `UnaryOp.mlp` is a single joint MLP over the whole 48-dim
  embedding, `d_total → d_total`. Nothing factorises it per digit. Its only
  learnable structure is the smoothness of a continuous map over digit-wise
  embeddings — the "terrain" argument — which supports interpolation but not
  composition.

The signature confirms it: unary has `fit 1.000 / reach 0.464`, the profile of
memorising 4422 trained values (~116 exposures each) while generalising to only
half the rest. Binary has `fit 0.824 / reach 0.799` — a gap of 0.025, meaning
it learned the function rather than the examples, which is precisely the
property the π = 0 end is built on.

D2 gave the binary operators a digit-wise decomposition to make them learnable.
**The unary operators never received the same treatment**, so they are now the
naive joint MLP that D2 itself identified as the unlearnable case. The
documents say the opposite — that unary has digit structure to extrapolate from
while binary is harder by an order of magnitude.

This is left **open**, not fixed. It is not obviously wrong to have one half
harder than the other, and the `tail` split will read most sharply on unary
because of it. But the spec and the code disagree about which half is which,
and that has to be resolved before the arms are built.

### D3 is not settled by this

At binary, `coupling = 0.25`:

| layout | \|V\| | reach |
|---|---|---|
| 8⁴ | 4096 | 0.845 |
| 17³ | 4913 | 0.799 |
| 10⁴ | 10000 | 0.577 |

and at unary, 17³ (0.464) edges out 8⁴ (0.424). 10⁴ is confounded — twice the
value space at the same training budget — so its deficit is not evidence about
the layout. 8⁴ and 17³ are close and disagree between tasks. No winner; 17³
keeps its incumbency on token cost (8.24 vs 9.00) until something separates them.

---

# State as of 2026-08-27, late evening

The `Current state` block at the top of this file is from the first entry and
has deliberately been left untouched. This is the newer summary; where the two
disagree, this one is later.

| self-check | state |
|---|---|
| #1 round-trip | pass |
| #2 oracle exactness | pass |
| #3 π ordering | pass |
| #4 hiddenness | **in flight** — two bugs found and fixed first, see below |
| #5 capacity | pass; a second sweep has now settled more of D2/D3 |
| #6 tokenizer probe | pass |

Built since the first entry: `sandbox.py`, `seal.py`, `train/{sft,infer}.py`,
`agent/{schema,tools,prompts,orchestrator}.py`, `arms/{base,a2,a4,a6,a0p}.py`.
That is W1·D6–10 of the plan. Not yet run end to end — vLLM was still
installing when this was written.

Still needing a decision from Asher, in the order they block things:

1. **`value_form`** — `letter_sep` is now measured on both cost *and*
   learning. It halves student token cost and does not hurt learning. The
   remaining question is whether to accept what that does to A2's economics.
2. **The unary/binary asymmetry** — now quantified. If unary is given the
   same digit-wise treatment binary has, its reach goes from 0.46 to 0.85.
   The knob exists (`unary_coupling`); nothing has been changed by default.
3. **D3 digit layout** — still no clean winner.

---

## 2026-08-27 — self-check #4: two bugs in the check before any measurement

Both were in the check, not in the benchmark, and both would have reported
"hidden" while measuring nothing. Recorded because the pattern repeats: every
self-check so far has needed its own verdict debugged before its subject
could be.

**Bug 1 — a truncated teacher scored as innocent.** First real run, `pi_mid`,
20 items:

```
exact  0/20 = 0.0000   chance 0.00020
digit  0.0000          chance 0.0588
tokens in=2942 out=32000   6.4m
verdict: at chance -- hidden
```

`out=32000` is exactly `max_tokens`. The teacher spent its whole budget
thinking and emitted no answers at all, and the empty parse scored as a clean
miss. The tell is `digit 0.0000` against a chance level of `0.0588` — below
chance is not a thing, and it means no answers were parsed rather than that
the answers were wrong. Fixed: the script now tracks `stop_reason` and answer
coverage and refuses to score a run that fails either, reporting `INVALID`
instead of a number.

**Bug 2 — a threshold that called 1000× chance "hidden".** With answers
actually flowing, `pi_mid`, 10 items:

```
exact  2/10 = 0.2000   chance 0.00020
digit  0.6389          chance 0.0588
answers parsed 10/10  truncated 0
verdict: at chance -- hidden      <- wrong
```

The cutoff was `exact > max(2, 3 * chance * n)`, so at n=10 it wanted more
than 2 hits — while at `1/4913` even one hit is p ≈ 0.002. Replaced with a
binomial tail test at p < 0.01.

**And a design point the numbers surfaced.** 2/10 with digit accuracy 0.64 is
not necessarily leakage. Many test expressions touch no table entry at all —
pure structural items whose answer is a permutation or subset of their inputs
— and the skeleton is *meant* to be inferable from demos. Those items cannot
test hiddenness, and mixing them in makes legitimate competence look like a
leak. Scoring is now split on `TestItem.needs_u | needs_b`, and the binomial
test runs only on the table-dependent half.

Full run in flight: `pi_low pi_mid pi_high`, 60 items each, chunk 20,
`max_tokens=64000`, teacher Opus 4.8 at effort `high`. It is slow — roughly
15–20 minutes per call because the teacher thinks hard on an unanswerable
task — so about three hours. Results land in
`~/glyph_runs/check4/hiddenness.json` on lumen-3.

---

## 2026-08-27 — self-check #5, second sweep: D2 nailed down, D3 not

Eight more runs, same hyper-parameters as the first sweep so the numbers are
directly comparable (4000 steps, batch 128, 512k examples, held-out 1-in-10,
`--eval-n 1000`). Raw JSON in `~/glyph_runs/cap5b/*.json`.

Two knobs were added to make this measurable, both defaulting to existing
behaviour so nothing was decided by adding them: `value_form="letter_sep"`
and `unary_coupling`.

### Everything measured so far, in one table

| run | task | layout | binary_c | unary_c | form | fit | **reach** |
|---|---|---|---|---|---|---|---|
| u_b17 | unary | 17³ | — | joint | underscore | 1.000 | **0.464** |
| u_lettersep | unary | 17³ | — | joint | letter_sep | 1.000 | **0.521** |
| u_b8 | unary | 8⁴ | — | joint | underscore | 1.000 | **0.424** |
| u_b8_lettersep | unary | 8⁴ | — | joint | letter_sep | 1.000 | **0.398** |
| u_dw000 | unary | 17³ | — | 0.0 | underscore | 1.000 | **1.000** |
| u_dw025 | unary | 17³ | — | 0.25 | underscore | 1.000 | **0.853** |
| u_dw050 | unary | 17³ | — | 0.5 | underscore | 1.000 | **0.696** |
| b_b17_c000 | binary | 17³ | 0 | joint | underscore | 1.000 | **1.000** |
| b_c0125 | binary | 17³ | 0.125 | joint | underscore | 0.901 | **0.873** |
| b_b17_c025 | binary | 17³ | 0.25 | joint | underscore | 0.824 | **0.799** |
| b_lettersep | binary | 17³ | 0.25 | joint | letter_sep | 0.842 | **0.803** |
| b_b17_c050 | binary | 17³ | 0.5 | joint | underscore | 0.652 | **0.622** |
| b_c075 | binary | 17³ | 0.75 | joint | underscore | 0.504 | **0.497** |
| b_b17_c100 | binary | 17³ | 1.0 | joint | underscore | 0.413 | **0.402** |
| b_b8_c025 | binary | 8⁴ | 0.25 | joint | underscore | 0.869 | **0.845** |
| b_b10_c025 | binary | 10⁴ | 0.25 | joint | underscore | 0.779 | **0.577** |

Chance is 2.0e-4 at 17³, 2.4e-4 at 8⁴, 1.0e-4 at 10⁴.

### `letter_sep` costs half and learns the same

Matched pairs, everything else held:

| | underscore | letter_sep | Δ |
|---|---|---|---|
| unary 17³ | 0.464 | 0.521 | +0.057 |
| binary 17³ c=0.25 | 0.799 | 0.803 | +0.004 |
| unary 8⁴ | 0.424 | 0.398 | −0.026 |

Small and inconsistent in sign — learning is indifferent to the notation. So
the earlier finding stands unopposed: `letter_sep` is a **pure cost win**,
4.00 student tokens per value against 8.24, and 7.00 teacher tokens against
8.00.

**That makes the decision purely about economics, which is why it is still
Asher's.** Halving the value cost roughly halves what A2 re-pays on each of
10⁴ test queries, and weakens the weights arm's advantage by exactly that
much. The honest direction — the plan already refuses to handicap A2 by
disabling prefix caching — but a change to what the experiment measures, not
a free optimisation.

### The unary/binary asymmetry is fully explained

The open question from the first entry now has a number attached.

| unary tables | reach |
|---|---|
| joint MLP (current default) | 0.464 |
| digit-wise, coupling 0.0 | **1.000** |
| digit-wise, coupling 0.25 | **0.853** |
| digit-wise, coupling 0.5 | 0.696 |
| *binary, coupling 0.25, for reference* | *0.799* |

Give the unary tables the same per-digit factorisation the binary tables got
from D2 and they land at 0.853 — right beside binary's 0.799. The whole gap
was the missing factorisation, nothing subtler.

One thing to weigh when deciding: **digit-wise unary at coupling 0 has only
`base × n_digits = 51` distinct parts.** Binary at coupling 0 has 867, which
was already judged small enough for an agent to enumerate into a lookup table
and hand the win to the code arm. 51 is enumerable in a single query batch.
So if unary is factorised, its coupling cannot be zero either — the same
argument that settled D2 applies with more force.

Three options, no recommendation acted on:

- **Symmetric**: set `unary_coupling` to a small value, so both halves are
  digit-wise-dominant. The spec's description then becomes true.
- **Asymmetric on purpose**: keep unary joint, and correct the spec to say
  binary is the extrapolatable half. `tail` then reads most sharply on unary,
  which is arguably useful.
- **Split the difference**: use the asymmetry as a designed axis — one half
  composable, the other only interpolable — and say so.

### D3 still has no winner

At binary, coupling 0.25: 8⁴ 0.845 > 17³ 0.799 > 10⁴ 0.577. At unary, 17³
0.464 > 8⁴ 0.424, and with `letter_sep` 17³ 0.521 > 8⁴ 0.398. The two tasks
disagree, and 10⁴ is confounded by twice the value space at equal training
budget. 17³ keeps its incumbency on token cost.

### A cleaner binary coupling curve

```
coupling   0      0.125   0.25    0.5     0.75    1.0
reach      1.000  0.873   0.799   0.622   0.497   0.402
```

Smooth and monotone across the whole range. Nothing pathological anywhere,
and the only unusable setting is 0.

---

## 2026-08-27 — the minimum closed loop, built

W1·D6–10 of the plan. Everything below is written and imports; none of it has
been run end to end yet.

### `sandbox.py` — running a solver we do not trust

The code arm's artifact is a program the teacher wrote, and it has to run
over 10⁴ expressions. Subprocess, wall clock, address-space cap, empty
working directory, JSON over stdin/stdout.

**The network was reachable on the first attempt and the test caught it.**
Clearing proxy environment variables does nothing to `socket`. A solver that
can reach the network after sealing is not a sealed artifact. Two layers now:
`unshare -rn` (a user namespace plus a network namespace, which works
unprivileged here — verified), and a socket block inside the harness for
hosts where it does not. `isolation_report()` says which is active; on
lumen-3 it is both.

7 tests, all passing, including: a solver that never defines `solve` fails
cleanly rather than crashing the run; one bad expression costs one item, not
the batch; a solver printing forged JSON cannot corrupt the result channel;
an infinite loop is killed.

### `seal.py` — one scoring entry point

`SealedArtifact` (context / program / adapter, with `entry` naming which one
answers) and `evaluate()`. Arms are scored by **exact match**, not the graded
credit `measure.py` uses for π — using graded credit here would quietly change
what the crossover figure measures. Sizes are recorded because they are costs.
`tail` is derived per run from the instance's own query log.

### `train/sft.py` — one loss for all seven roles

Token CE on the response only, masked so the prompt is given rather than
predicted. Per-example weighting for R4's score-weighted SFT, applied *after*
per-example normalisation so a long answer does not silently outweigh a short
one. LoRA r=32 by default, full fine-tune behind a flag. GPU seconds measured
through the ledger, not estimated.

### `agent/` — the tool surface and the loop

`schema.py` defines the seven roles and builds tool definitions with
`strict: true` and `additionalProperties: false`. **The arm boundary is
enforced by withholding tools, not by instruction**: A2 is simply never given
`train`. Asking a model to avoid a tool it can see makes the arm boundary
depend on compliance.

`tools.py` implements the API. One design decision worth flagging: `evaluate`
needs labelled dev items, and labelled items are exactly what the agent is
paying for — so **dev is carved out of what it has already queried** (a 15%
slice held back from training) rather than handed over free. A free labelled
dev set would refund part of the query budget and shrink the 10³→10⁴
asymmetry that is the entire reason the weights arm exists.

`orchestrator.py` is a hand-written loop, not the SDK tool runner, because
replay needs to own the loop: charge, trace, interrupt on budget exhaustion,
and **replay from the response cache instead of re-billing**. Costs follow the
real bill — cached reads at 0.1×, cache writes at 1.25×. If the agent never
seals, the harness seals for it so the run still produces a comparable number.

### `arms/` — one module each

`a2_context`, `a4_code`, `a6_weights`, `a0p_saturated`, plus shared setup.
Never a branch inside another arm. A0′ takes another arm's purchased query log
as input rather than buying its own, which holds the information constant and
varies only who reads it.

---

## 2026-08-27 — the prefix cache: investigated, then handed to vLLM

**Instruction received mid-session: do not hand-roll a KV cache, use vLLM.**
Done — `train/infer.py` now runs on vLLM with `enable_prefix_caching=True`.
Recording the investigation anyway, because its conclusion outlives the
implementation.

The original concern that led to hand-rolling: installing vLLM risked pulling
torch back from 2.13+cu130. That was raised and overruled, correctly — the
plan is explicit that A2 must have prefix caching or it is a straw man, and a
cache we wrote is one more thing that has to be proved equivalent before any
A2 number can be trusted.

The hand-rolled version was, as it happens, correct, and proving it produced
something worth keeping:

```
bf16: answers identical=False   max|logit diff|=0.18750
fp32: answers identical=True    max|logit diff|=0.00003
```

Prefilling a prefix on its own and computing it inside a longer sequence give
bitwise-different results in bf16 — different accumulation order — and on an
item near a decision boundary that is enough to flip a token. In fp32 the two
paths agree to 3e-5 and every answer matches.

**This is not specific to the discarded implementation. Every prefix-caching
implementation has it, vLLM included.** Two consequences that survive the
switch:

- A2's answers are not bit-reproducible against an uncached reference. Fine
  in itself — A2 always runs cached, which is its real deployment form — but
  replay determinism has this caveat, and A2's decode path must not be
  switched between cached and uncached across runs in a single figure.
- Any future equivalence check of this kind has to be run in fp32, or it
  measures bf16 noise.

Kept as `tests/test_infer.py` (marked `slow`, fp32, needs a GPU), rewritten
against the vLLM path.

---

## Open problems and things not yet done

**Not yet run end to end.** The closed loop has never executed. E0 and E0b
have not started. This is the immediate next thing.

**vLLM install was still in flight** when this was written. If it downgraded
torch, `capacity_check.py` and `train/sft.py` need re-verifying — they were
last known green on torch 2.13.0+cu130, transformers 5.16.1, peft 0.20.0.
The pre-install versions are recorded here for exactly that reason.

**`count_tokens` is blocked** by org policy on this Vertex project
(`constraints/vertexai.allowedModels` does not admit `count-tokens`). Token
accounting comes from `usage` on real requests instead. Works, but it means
the ledger cannot price a prompt before sending it.

**Automatic prompt caching is unavailable on Vertex** (explicit
`cache_control` works; the automatic path does not). The plan's ledger assumes
caching discounts are reachable — they are, but only by setting breakpoints
explicitly in the orchestrator. Not yet done.

**`worker.py` does not exist.** The plan requires the scheduler to group runs
by `instance_seed` so that all arms at a given budget point share an instance;
paired comparison is what makes the crossover's confidence interval narrow
enough to locate B*. Written only in the method section so far, which is the
condition under which the plan says it will break in practice.

**The E9 ladder's pricing is unverified.** Sonnet 4.5's rate must be looked up
before it enters the ledger; the plan warns explicitly against copying Sonnet
5's $2/$10.

---

# 2026-08-27, overnight — self-check #4 finds a real leak, and the loop runs

## The null hypothesis was wrong, and that mattered more than the bug

`1/|V|` was the criterion from the start. It asks whether the teacher can
guess a value uniformly, and nobody was ever going to. Two things are
legitimately free: the **skeleton**, which the design *intends* to be
inferable from demos, and whatever **table entries the demos reveal outright**.

Measured on the same 300 items, `seed=1001`:

| preset | true skeleton + identity table | true skeleton + demo-revealed table | trivial skeleton + demo-revealed | uniform |
|---|---|---|---|---|
| pi_low | 0.0000 / d 0.343 | 0.0000 / d 0.345 | 0.0000 / d 0.343 | 0.0002 |
| pi_mid | 0.2233 / d 0.237 | 0.2233 / d 0.238 | 0.1467 / d 0.162 | 0.0002 |
| pi_high | 0.7833 / d 0.763 | **0.8900** / d 0.896 | 0.2767 / d 0.468 | 0.1250 |

Demos reveal 99u+0b entries at pi_low, 75u+2b at pi_mid, 16u+0b at pi_high.

**On pi_high the legitimate null alone scores 0.89 exact.** Against a `1/|V|`
criterion of 0.125, the check would have reported a catastrophic leak from a
benchmark behaving exactly as designed. The check now measures its null
instead of assuming one.

Trivial strategies that know nothing at all, for reference:

| preset | copy the input list | first value | last value |
|---|---|---|---|
| pi_low | 0.0000 / d 0.343 | 0.0000 / d 0.115 | 0.0000 / d 0.022 |
| pi_mid | **0.1500** / d 0.168 | 0.0967 / d 0.102 | 0.0033 / d 0.040 |
| pi_high | 0.1533 / d 0.418 | **0.4200** / d 0.408 | 0.3467 / d 0.339 |

Echoing the input scores 15% on pi_mid and 42% on pi_high. The earlier
10-item pi_mid reading of 2/10 sits inside that band and was never evidence
of anything.

## pi_low fails, and this one is real

```
pi_low, 60 items, teacher Opus 4.8, effort high, no queries at all

  every legitimate null      exact 0.0000   digit 0.343
  best copy strategy         exact 0.0000   digit 0.343
  trivial skeleton + TRUE table   exact 0.0633   digit 0.680
  TEACHER, demos only        exact 0.1167   digit 0.7810
  answers parsed 60/60, no truncation, 23.5m, in=9443 out=125576
```

From 30 demos and zero queries the teacher **beats an oracle holding the
entire true table**. Its digit accuracy exceeds every baseline that has no
table knowledge by more than a factor of two. It is extrapolating table
entries it was never shown.

This is **risk R3 in the Glyph spec**, stated there as: *"ICL 其实能外推得很好
→ 学习算法那条理由消失，只剩经济学，论文变薄但不死。"* It is also, in
miniature, the question A0′ exists to answer — except A0′ hands over the full
purchased record, and this is 30 free demos.

**What it would mean if it holds.** H1's claim strength is set by whether the
weights arm wins on economics or on learning algorithm. If in-context
learning extrapolates these tables from 99 revealed entries, the
learning-algorithmic half weakens and the honest claim is the economic one.
The plan already commits to reporting that outcome plainly rather than
dressing it up, so this is a result rather than a problem — but it is a
result that changes the paper.

**Why it is not yet confirmed.** Teacher scored on 60 items, nulls on 300 —
different samples. The two scripts' digit-alignment differs slightly. One
preset, one instance seed, one teacher. The corrected check computes the null
on the same items with the same metric, and needs a clean re-run across seeds
before any of this is load-bearing.

**Worth trying before concluding.** pi_low has `n_structural=2`,
`max_transform_depth=0`, `guard_prob=0` -- the skeleton is nearly trivial by
construction, so almost all of the teacher's advantage has to be coming from
the table. Raising `mlp_temp` (the smoothness knob) is the designed response
if the table turns out to be too interpolable; that is what the knob is for.

## Two more org-policy walls on this Vertex project

Both are environment constraints, not code problems, and both are recorded
because they deviate from what the plan assumes.

**`structured_outputs` is refused for partner models.** The plan's §6.3 asks
for `strict: true` on every tool so the API guarantees argument shape and this
side does no defensive parsing. A request carrying it returns 400:
`constraints/vertexai.allowedPartnerModelFeatures`. `strict` is now off and
`agent/schema.py::validate()` does the same job client-side -- required keys,
unknown keys, enums, ranges. Weaker: a malformed call becomes an error the
model sees and retries rather than something the API made impossible. But
`declare_target` still cannot record a role outside the enum, which is the
property the analysis depends on. Turn it back on if the policy is ever
widened; the schemas already carry `additionalProperties: false`.

**`count_tokens` is refused** (recorded in the earlier entry, repeated here
because it now has a consequence): the ledger cannot price a prompt before
sending it, only after.

## The closed loop runs

A4 end to end on lumen-3 -- agent loop, oracle, sandbox, seal, score:

```
run_start  arm=a4  allowed=['code']  budget=4000 H100-s  tools=[query_oracle, write_code, seal]
query_oracle -> 31 queries   spent  199 H100-s
query_oracle -> 59 queries   spent  456
query_oracle -> 83 queries   spent  853
query_oracle -> 108 queries  spent 1339
query_oracle -> 134 queries  spent 1793
query_oracle -> 148 queries  spent 2571
```

No malformed queries, budget tracked, tools correctly restricted to the code
arm.

## The budget axis is measuring the wrong thing

The trace above is the finding. **148 oracle queries cost about 1.3 H100-s.
The other ~2570 are the agent's own tokens.**

At the ledger's rates -- `$5/MTok` in, `$4/hr` H100 -- one H100-second buys
222 input tokens, and one oracle query is priced at `$1e-5`, i.e. 0.009
H100-seconds. So:

| | cost in H100-s |
|---|---|
| 2000 oracle queries (the plan's Q) | ~18 |
| one agent turn with a 10k-token context | ~45 |
| a 20-turn conversation with growing context | ~2500 |

**Information purchase is two orders of magnitude cheaper than thinking about
it.** Nothing in the unified currency enforces the plan's `Q ≈ 2000`; an agent
should rationally query until it runs out of context, and the budget sweep in
E1 would mostly be sweeping *conversation length*, not *information bought*.

That undercuts the setup's central asymmetry. The whole reason the weights arm
exists is that `Q ≈ 2000` against `|V| = 4913` leaves the tables half-covered
at best -- but that only bites if something actually stops the agent at 2000.

Three ways out, none taken:

1. **Raise `usd_per_oracle_query`** until the query budget binds. Defensible
   -- a query is a call into a hidden system, and its price is ours to set --
   but the number would be chosen to produce the regime we want, which needs
   saying out loud in the paper.
2. **A separate hard cap on queries**, orthogonal to the compute budget. This
   matches how the spec actually talks (`Q` and `B` are discussed as different
   things), and makes the information axis explicit rather than emergent.
3. **Accept it** and re-frame: the binding constraint is the agent's own
   inference, and the crossover is about deliberation cost. A coherent story,
   but not the one the plan tells.

My read is that (2) is closest to what the documents already assume, but this
is a design decision and it is not mine to take. It needs settling **before
E1**, because the budget axis is Fig. 1's x-axis.

## Environment note

vLLM 0.28.0 installed, and **torch stayed at 2.13.0+cu130** -- the downgrade
risk that motivated the hand-rolled cache did not materialise. numpy moved
2.4.6 -> 2.3.5, which nothing depends on. Verified: `torch.cuda.is_available()`
true, 8 GPUs, vLLM generation working with prefix caching on, 32 prompts in
0.07 s.

vLLM's prefix-cached answers match the discarded hand-rolled cache's answers
exactly, including the one borderline item where both differ from an uncached
run -- independent corroboration that the bf16 divergence documented earlier
is a property of prefix caching in general rather than of that implementation.

## State at end of session

Done tonight: self-check #4 built and its criterion corrected twice; the pi_low
leak found; the second capacity sweep (D2 settled, `letter_sep` shown
cost-only, unary asymmetry quantified); sandbox, seal, train, agent and arms
written; A4 verified end to end; vLLM in place.

Not done: E0 and E0b have not been run. `worker.py` still does not exist.
A2, A6 and A0′ have not been executed even once. The corrected #4 has not been
re-run across seeds.

Blocking on a decision: `value_form`; the unary/binary asymmetry; the budget
axis above; D3. None of them were decided unilaterally.

---

# 2026-08-27, overnight (cont.) — the first arm result, and what it says about the budget

## A2 ran end to end

The first complete arm: agent loop, oracle, seal, student generation, scored.
`pi_mid`, `instance_seed=1001`, 120 test items, 3000 H100-s budget, 12.1
minutes.

```
overall   0.0583
by_split  iid 0.0513   comp 0.1071   depth 0.0000
tail      0.0215
artifact  a context prefix of 2141 characters
```

Not a result to interpret — one instance, one seed, a trimmed test set, and
the budget axis is unsettled. What matters is that the path works and that
the ledger is now populated with real numbers.

## The ledger settles the budget question

```
spent 2161 / 3000 H100-s      $2.40

  frontier_in     891.3   ┐
  frontier_out   1268.5   ┘  2159.8   99.95%   the agent's own conversation
  oracle_query      0.93            0.04%   buying information
  gpu_second        0.31            0.014%  the student actually answering
```

This is the same finding as the A4 trace, now with the whole run accounted
for. **Buying information is 0.04% of the budget. The student answering 120
queries is 0.014%.** Essentially the entire "compute budget" is the frontier
agent talking to itself.

Two things follow, and both matter before E1:

**The query budget does not exist as a constraint.** The plan's `Q ≈ 2000`
against `|V| = 4913` is the reason the weights arm exists — the tables cannot
be bought outright. But 2000 queries cost about 18 H100-s here, so nothing
stops an agent from buying far more. The asymmetry the design rests on is
currently enforced by the agent's disinclination rather than by the protocol.

**A2's per-query cost, the thing the crossover is supposed to turn on, is
invisible at this scale.** The whole argument for a crossover is that context
is re-paid on every test query while weights are paid once. Here the student's
inference over the full test set cost 0.31 H100-s against 2160 for the
prepare phase — a ratio of about 7000:1. At 10⁴ items instead of 120 it would
be roughly 26 H100-s, still around 1% of the prepare budget. **A2's cost
disadvantage cannot show up in a total that the agent's own tokens dominate
by three orders of magnitude.**

That is not a bug in the ledger; the rates follow published prices, which is
what the plan asks for. It is that prepare-phase frontier tokens and
test-phase student inference are quantities of very different size, and
summing them into one scalar buries the smaller one.

Options, none taken:

1. **Report test-phase cost on its own axis** alongside the total. The
   crossover in the plan is about the *marginal* cost of answering, and that
   is a quantity the current single total cannot express.
2. **A separate hard cap on oracle queries**, so the information axis is
   explicit. Closest to how the documents already talk about `Q` and `B` as
   different things.
3. **Raise the test-set size and the query price together** until both land
   in the same range as the agent's tokens. Makes the single scalar work, but
   the numbers would be chosen to produce the regime we want, which has to be
   said out loud.

Whatever is chosen, it should be settled before E1 — this is Fig. 1's x-axis
and Fig. 1 is the paper.

## A crash that would have killed every A6 run

vLLM starts its engine in a subprocess, and the default `fork` cannot inherit
an already-initialised CUDA context: *"Cannot re-initialize CUDA in forked
subprocess"*. It matters here because **A6 trains before it generates**,
which is the normal order, so the GPU is always already live when the student
is built.

*Corrected while writing this entry, having first overstated it.* vLLM 0.28
usually notices and rescues itself — the A6 run in flight was started from
the pre-fix code and logged *"We must use the `spawn` multiprocessing start
method. Overriding ... Reasons: CUDA is initialized"*, then carried on. But
that detection does not always fire: in the test suite it did not, and the
engine died. So this is not "every A6 run would have crashed"; it is that
whether an A6 run crashes depends on a heuristic firing.
`VLLM_WORKER_MULTIPROC_METHOD=spawn` is now set before vLLM is imported
anywhere, which removes the dependence rather than the symptom.

Alongside it, a subtler one with no error message: after training, the
optimiser state and model are dead but still resident, and vLLM sizes its KV
cache from whatever it sees free. Without an explicit reclaim, A6's student
would silently get a fraction of the memory it should and the arm would look
slower than it is — a performance difference that would have been read as a
property of the weights container. `Student` now reclaims on construction and
exposes `close()`; dev-evaluation students are released the same way.

## Self-check #4, pi_mid: INVALID, and the guard earned its place

```
pi_mid   |V|=4913  n=60
  overall        5/60 = 0.0833   digit 0.4138
  needs tables   2/47            digit 0.4379
  skeleton only  3/13            digit 0.3086
  answers parsed 40/60   truncated 1
  INVALID -- coverage 67%, 1 truncated call
```

The teacher ran out of output tokens on one chunk and only 40 of 60 answers
came back. The coverage guard added earlier refused to score it — which is
exactly the failure mode that produced the very first misleading "hidden"
verdict. Needs a re-run at a smaller `--chunk`.

Worth noting even from an invalid run: `2/47` on table-dependent items is
**below** the measured pi_mid null of 0.2233, so there is no sign of a leak
here. The pi_low result stands alone so far, which is consistent — pi_low is
the preset whose skeleton is nearly trivial by construction, so the table is
where all its difficulty lives and where a leak would show.

Also visible: 13 of 60 pi_mid items need no table at all, against 0 of 60 at
pi_low. The split by table-dependence is doing real work.

## Tests

60 green, up from 36 at the start of the session. New: `test_sandbox.py` (7),
`test_agent.py` (13), `test_infer.py` (3, rewritten for vLLM).

`test_agent.py` covers the invariants that would not raise if they broke and
would instead make a difference between arms look like a difference between
containers: the arm boundary is the tool list; a role outside the enum cannot
reach the trace; a malformed query is billed but not banked; dev does not
overlap training; training without `declare_target` is refused; a target
cannot be redeclared.

One of those tests found a mistake in itself rather than in the code — a
hand-written expression was rejected as malformed because the smoke preset
requires lists of at least two elements. Tests now draw expressions from the
instance's own demos instead of assuming the grammar.

## `worker.py` exists now

The plan's paired-comparison rule is a property of the scheduler rather than
a line in the method section. Jobs are emitted instance-major, so every arm
at a budget point runs on the same hidden interpreter; an instance with a
failed arm is reported loudly rather than averaged over, because a partially
finished instance cannot be used in a paired comparison and quietly using
what is left puts instance difficulty back into the error bar.

One subprocess per run — vLLM and torch do not survive being set up twice in
one process, and a crash should cost one cell rather than the sweep. Nothing
retries: a failed cell is a cell with a reason attached.

`python -m glyph.cli grid --arms a2 a4 a6 --instance-seeds 1001 1002
--budgets 5000 15000 --dry-run` prints the plan without running it.

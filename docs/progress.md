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

---

# 2026-08-27, overnight (cont. 2) — A4 completes, and the code arm behaves as designed

## A4, the first interpretable arm result

`pi_mid`, `instance_seed=1001`, 120 test items, 15000 H100-s budget, 27.8 min.

```
overall   0.3700
by_split  iid 0.3769   comp 0.4130   depth 0.2500
tail      0.1544
artifact  a 3499-byte Python solver

spent 6905 / 15000 H100-s      $7.67
  frontier_in    3861.0
  frontier_out   3040.2      99.94%  the agent
  oracle_query      4.25       0.06%
  gpu_second        0.086            the sandbox running the solver
```

**`tail` is 0.154 against an overall of 0.370 — a drop to 42%.** That is the
code container's predicted signature, arriving on the first run that got far
enough to show it. The plan states it outright: *"code 臂：只能硬编码查到的
2000 条，覆盖到的全对,tail 上归零"* — precise on what was bought, weak on what
was not. Half the phase diagram's explanation is this gap, and it is visible
in one run.

`depth` at 0.25 against `iid` at 0.377 is the other predicted weakness
showing up: expressions nested deeper than anything in the demos.

One run, one instance, one seed. Not evidence of anything yet, but the
mechanism the design is built on is producing the shape it was supposed to.

For contrast, A2 on the same instance and preset scored 0.058 overall with
`tail` 0.022 — but at a different budget (3000 vs 15000) and with 12 turns
against 12, so the two are **not** comparable. Paired comparison at matched
budgets is what `worker.py` exists to enforce, and it has not been run.

## A6 crashed on the bug that was already fixed

```
BudgetExhausted: 3224.4 / 3000.0 H100-s after frontier_in
```

Identical to A4's first failure. The A6 process had been launched *before*
the orchestrator fix landed, so it was running the old code — the crash
confirms the bug was real and general rather than something specific to A4.
Relaunched against the fixed orchestrator with a 15000 budget.

## Self-check #4: pi_mid and pi_high both INVALID, for the same reason

```
pi_mid   answers parsed 40/60   truncated 1   out=171222
pi_high  answers parsed  0/60   truncated 3   out=192000   (= 3 x the 64000 cap)
```

At `effort=high` the teacher spends its entire output budget thinking about
an unanswerable task and never reaches the answers. On `pi_high` every one of
the three calls hit the cap exactly and not a single answer came back.

The coverage guard refused to score both, which is the whole reason it was
added — an unguarded version would have reported `pi_high` as **0/60, perfectly
hidden**, the strongest possible pass, from a run that measured nothing at
all.

Relaunched at `--chunk 10 --max-tokens 100000`, so each call has roughly six
times the room per item.

**Only `pi_low` has produced a valid #4 result so far**, and that is the one
showing the leak. Whether pi_mid and pi_high leak is still unmeasured — the
numbers in the invalid runs are not evidence in either direction. Worth noting
that the earlier invalid pi_mid figure of 2/47 on table-dependent items sits
*below* that preset's measured null of 0.2233, so there is no hint of a leak
there, but a run with a third of its answers missing cannot support even that.

## A note on effort as a controlled variable

The plan fixes `effort=high` across all arms and sweeps it only in E7, which
is right for the arms. Self-check #4 is not an arm, and at `high` the teacher
burns 64000 output tokens per 20 items without answering. If the re-run at
chunk 10 still truncates, the options are a smaller chunk again, or running
#4 at a lower effort and saying so — a hiddenness check is asking "can it
guess", and a teacher that thinks itself out of an answer has not
demonstrated hiddenness, it has demonstrated a token budget.

---

# 2026-08-27 — can the student hold the whole table?

Asher asked directly: at the pi=0 end, can Qwen3-1.7B fit the *entire* hidden
mapping of the atomic operator tables? The earlier #5 sweep did not answer
that — it deliberately held out 1 entry in 10 because the interesting
question then was extrapolation. `--holdout-mod 0` was added to ask the
literal question, and six runs were spent on it.

## Unary: yes, completely

```
full_unary        holdout 0   4000 steps    fit 1.000
full_unary_long   holdout 0  12000 steps    fit 1.000
full_unary_dw     holdout 0   4000 steps    fit 1.000   (digit-wise variant)
```

All 4913 entries trained on, nothing held back, **not one wrong**. True for
the joint-MLP form and the digit-wise form alike. The earlier `fit 1.000` on
4422 entries had made this near-certain; it is now measured rather than
inferred.

Capacity was never the question: the table is 47,744 parameters against the
student's 1.7 billion — four orders of magnitude.

## Binary: not yet, but it is climbing, and I predicted the wrong reason

```
                        fit     reach
 4000 steps (hold 10)   0.824   0.799
12000 steps (hold 10)   0.853   0.849
24000 steps (hold 10)   0.877   0.883
 4000 steps (hold  0)   0.807     n/a
```

**I said I leaned toward "it cannot learn it" on the basis of how slowly the
loss was falling. That was wrong.** Six times the training moved reach from
0.799 to 0.883 and it is still rising, roughly logarithmically. Nothing here
says it has plateaued.

The `holdout 0` run is what separates the two candidate explanations. Trained
and evaluated on the same distribution with nothing held back it still only
reaches 0.807 — barely different from the 0.824 with a held-out slice. So the
obstacle is **not** coverage (512k samples against 24.1M pairs), it is how
fast the function itself is learned.

The most informative number is not fit at all:

> At 24000 steps **reach (0.883) has caught up with fit (0.877)**.

Held-out pairs do as well as trained ones. The model is learning the
*function*, not memorising entries — which is precisely the property the pi=0
end is built on, and a better answer to "can it hold the table" than a fit
number would have been. Under D2 the binary map is 867 digit-wise parts plus
a coupling term, not 24M independent facts, so "hold the whole table" was
always the wrong shape of question for this half.

D2's escape hatch — winding `binary_coupling` back — is **not** needed on this
evidence.

## A caveat on the framing

These use `PRESETS["pi_low"]`, whose *measured* pi averages 0.25, not 0. So
this is the low-pi end, not literally pi=0. No preset currently sits at pi
near zero; reaching it would mean pushing `atomic_ratio` higher or flattening
the skeleton further.

## Self-check #4: pi_mid and pi_high do not leak; only pi_low does

Re-run at `--chunk 10 --max-tokens 100000`, now scored against the measured
null:

```
pi_mid   needs tables  2/47 = 0.043   null 0.2167   coverage 83%  INVALID
pi_high  needs tables  4/18 = 0.222   null 0.8500   coverage 83%  INVALID
```

Both still flagged invalid — one truncated call each — but the direction is
worth reading with one correction stated first: **missing answers only push a
score down, so an incomplete run is biased toward "no leak"** and cannot be
taken at face value. The bound is what settles it:

- `pi_high`: even if all 10 missing answers were correct *and* table-dependent,
  the ceiling is 14/18 = 0.78, still under the null of 0.85. **No leak,
  safely.**
- `pi_mid`: the same ceiling is 12/47 = 0.255 against a null of 0.217. Too
  close to call. **Needs a clean run.**

So `pi_low` remains the only confirmed leak, and it is now more interesting
for being the only one: it is the preset whose skeleton is nearly trivial by
construction, so its difficulty lives entirely in the table — exactly where a
leak would matter most.

## The real obstacle in #4 was effort, not chunk size

The teacher burned **422k and 454k output tokens** on the two presets, 80 and
86 minutes, and still truncated once each at chunk 10 with a 100k cap.

The plan fixes `effort=high` as a controlled variable across all arms, and
that is right — for arms. **#4 is not an arm.** A teacher that thinks past its
output budget without answering has demonstrated a token budget, not
hiddenness. `request_kwargs` now takes an optional effort override, documented
as being for the self-checks only, and `hiddenness_check.py` defaults to
`medium`. Re-running all three presets at that setting.

---

# 2026-08-27 — A6 completes; pi_low's leak is confirmed cleanly; pi_high still will not answer

## A6, on the fourth attempt

Through the production path (`python -m glyph.cli run --arm a6`), which is
what `worker.py` dispatches, so this validated the code a sweep would
actually use.

```
overall   0.0333
tail      0.0323
artifact  an 81 MB LoRA adapter

spent 3053 / 15000 H100-s      $3.39
  frontier_in    1843.6
  frontier_out   1177.3     98.9%
  gpu_second       31.1     1.0%   training and inference together
  oracle_query      1.17    0.04%
```

`gpu_second` at 31.1 is the first non-trivial GPU figure in any run — A2's
was 0.31 and A4's 0.086 — and it is still 1% of the total.

**The three arms are not comparable yet.** They ran at different budgets (A2
at 3000, A4 at 15000, A6 spent 3053 of 15000), on the same instance but
without the paired discipline. Reading `A4 0.370 > A2 0.058 > A6 0.033` as an
ordering would be reading budget differences. That is exactly what
`worker.py` exists to prevent, and it has not yet been used for a real grid.

What this run does establish is that the whole loop executes: agent buys,
declares a target, synthesises, trains, evaluates on purchased dev, seals,
and the adapter answers the sealed test set through the same scoring path as
the other two.

## Four failures, and only the first was a product bug

Worth writing down because the pattern was mine, not the code's.

1. **Budget exhaustion during billing** propagated out and killed the run —
   a real bug, in the handler's coverage. Fixed.
2. `fork` cannot inherit an initialised CUDA context, and A6 trains before it
   generates. I set `spawn`.
3. `spawn` re-imports the entry module in the child, so a driver script
   without an `__main__` guard re-runs itself. I moved to in-process.
4. In-process, my `close()` tore down the process group, and the *second*
   student in the run died on half-initialised distributed state — "Process
   group is not initialized in the world group map".

Three, four and the fix for two were all consequences of the previous fix.
The root cause never moved: **A6 builds several students in one run** — one
per dev evaluation inside the agent loop, one more for the sealed test set —
and I kept rebuilding the engine around each.

The right shape is one engine per process with adapters swapped per request,
which is what `LoRARequest` is for. Verified: three students in sequence, one
with a context prefix, `engines built: 1`. It also removes minutes of GPU time
per dev evaluation that were being spent reloading 1.7B parameters.

I should have stopped after the second failure and looked at the shape rather
than patching the symptom three times.

## Self-check #4: pi_low's leak is now clean

```
pi_low   effort=medium   60/60 answers parsed   0 truncated
  needs tables   6/60 = 0.100   digit 0.771
  null (demos)   0.000          digit 0.384
  p = 3.5e-15                   LEAK SUSPECTED
```

Full coverage, no truncation, scored against the measured null. **Two
independent runs at different effort settings — `high` gave 7/60, `medium`
gives 6/60 — agree.** The teacher extrapolates table entries it was never
shown, from 30 demos and no queries.

## pi_high: my diagnosis was wrong

I wrote that the obstacle was `effort`, not chunk size, and lowered it.

```
effort=high     chunk 20   out=192000   3 truncated   0/60 answers
effort=medium   chunk 15   out=256000   4 truncated   0/60 answers
```

**Lowering effort made it worse, not better.** Every chunk burns the full
64000-token output cap and emits nothing. That is not a thinking-depth
problem, and my explanation did not survive its first test.

What is different about `pi_high`: `|V| = 8` but the skeleton is the hardest
of the three — 8 structural operators, depth 5, guards on almost everything —
and 42 of 60 items need no table at all. The teacher appears to get stuck
trying to pin the skeleton down and never reaches the answers.

`pi_mid` is also still INVALID at 75% coverage, so the only preset with a
valid #4 result remains `pi_low`.

Running a proper one-variable-at-a-time diagnostic rather than guessing
again: effort and `max_tokens` and chunk size varied separately on 2–5 items,
reporting `stop_reason` and how many characters of text come back.

---

# 2026-08-28 — the decode bug, and everything that had to be re-run because of it

The biggest entry so far, because the finding sits under everything measured
before it.

## The tables were not what the design says they are

`Tables._decode` mapped an MLP output back to a symbol by `argmin` over all
`|V|` embeddings. That collapses. `tanh` holds the output near the centre of
the space while the embeddings are Gaussian, so the few embeddings nearest
the centroid win almost every argmin — the hubness effect in high-dimensional
nearest-neighbour search.

Measured, `seed=1001`:

| preset | operator | distinct outputs | entropy | most common output |
|---|---|---|---|---|
| pi_low | unary u0 | **222** / 4913 | 5.30 / 12.26 | 18.7% |
| pi_low | binary b0 | 339 / 4913 | 6.30 | 8.9% |
| pi_mid | unary u0 | 160 / 4913 | 4.97 | 19.3% |
| pi_mid seed 1002 | unary u0 | 193 / 4913 | 4.58 | **30.1%** |
| pi_high seed 1002 | binary b0 | 3 / 8 | 0.72 | **85.8%** |

`|V| = 4913` was a fiction: the image was ~200 values. "The table cannot be
bought outright" — the reason the weights arm exists — rests on the size of
that image, not on the size of the domain.

### How it surfaced, and the wrong conclusion I drew first

Self-check #4 reported a leak on pi_low: the teacher scoring 0.117 exact and
0.781 digit from 30 demos and no queries, against a measured null of 0.000.
I wrote that up as in-context learning extrapolating the table — **risk R3 in
the Glyph spec, the finding that would have weakened H1's claim to the
economic half only.**

It was not that. Always answering the single most common output scores
**0.187 exact and 0.582 digit** on the broken pi_low tables — above the
teacher's exact and comparable on digits. The teacher was reading the output
distribution off the demos, which is free information, and not extrapolating
anything.

The check was right to fire. My reading of why was wrong, and it was wrong in
the direction of the more interesting story.

## Three candidate fixes, measured

Implemented behind `GlyphConfig.decode`, defaults untouched during the
comparison. Three things had to hold at once — a fix that buys one by losing
another is not a fix:

* **spread** — the image must not collapse, or `|V|` stays a fiction
* **structure** — digit-neighbours must still map near each other, because
  that correlation is the only reason anything extrapolates
* **pi** — the measurement has to survive

pi_low, seed 1001, unary u0:

| decode | distinct | entropy | top-1 | nbr-agree | far-agree | contrast | pi |
|---|---|---|---|---|---|---|---|
| `nearest` | 222 | 5.30 | 18.7% | 0.608 | 0.433 | 1.40x | 0.347 |
| `whiten` | **1222** | 9.46 | 0.8% | 0.361 | 0.109 | **3.31x** | 0.377 |
| `per_digit` | 222 | 5.30 | 18.7% | 0.608 | 0.433 | 1.40x | 0.347 |
| `assign` | 4913 | 12.26 | 0.0% | 0.152 | 0.059 | 2.58x | 0.352 |

**`per_digit` is refuted, not rejected.** I had recommended it as the fix. It
is the same function as `nearest`: with concatenated embeddings and every
digit combination legal, squared distance decomposes and global argmin *is*
per-digit argmin. Identical on every preset, seed and operator — six
configurations, every number the same. That also corrects the diagnosis: the
collapse is not "too many candidates", since 17 candidates per digit collapse
just as hard. It is `tanh` pushing every segment toward its digit bank's
centroid.

**`assign`** builds a true bijection but cannot exist for binary — it needs
all 24M outputs materialised, which is the thing the table exists not to do.
A bound, not an option.

**`whiten`** — putting the MLP output on the embeddings' own per-dimension
scale before the argmin — is now the default. Across pi_low and pi_mid, two
seeds each: distinct outputs 160-224 to 1132-1265, top output 15-30% to
0.8-2%, and the neighbour-versus-random contrast roughly doubles. It works
for binary, which is what settles it.

Confirmed from the other side: the mode baseline falls **0.224 to 0.011**.

### One caveat recorded rather than fixed

On `pi_high`, `n_digits = 1`, and no decode preserves neighbour structure
because there is none to preserve — `whiten` and `assign` both drive
neighbour agreement to or below the random level. That end of the phase
diagram has no extrapolable table at all. The spec should say so: at pi -> 1
the weights arm is not extrapolating poorly, it has nothing to extrapolate
from.

### A test that was passing because of the bug

`test_digit_neighbours_are_correlated` asserted that perturbing a digit left
the output *identical* more often than chance — 28% against 0.02%. That was
measuring the collapse, not checking against it: the test the design relies
on was being satisfied by the defect it should have caught. It now compares
digit agreement for neighbours against unrelated values.

## Self-check #5, re-run on fixed tables

Everything below is `whiten`. The old column is the same run on broken
tables, kept because the size of the shift is the point.

| run | steps | adapter | old reach | **new reach** | mode |
|---|---|---|---|---|---|
| unary, joint MLP | 4000 | full | 0.464 | **0.091** | 0.011 |
| unary, joint MLP | 12000 | full | — | 0.137 | 0.011 |
| unary, joint MLP | 24000 | full | — | 0.158 | 0.011 |
| unary, digit-wise 0.25 | 4000 | full | 0.853 | **0.583** | 0.014 |
| unary, digit-wise 0.25 | 12000 | full | — | 0.597 | 0.014 |
| binary c=0 | 4000 | full | 1.000 | 1.000 | 0.015 |
| binary c=0.25 | 4000 | full | 0.799 | **0.562** | 0.013 |
| binary c=0.25 | 12000 | full | 0.849 | 0.675 | 0.013 |
| binary c=0.25 | 24000 | full | 0.883 | 0.734 | 0.013 |
| binary c=0.5 | 4000 | full | 0.622 | 0.379 | 0.011 |
| binary c=1.0 | 4000 | full | 0.402 | 0.221 | 0.008 |

Every number fell, because part of every old number was the output
distribution. The mode baseline fell twenty-fold and the reach numbers
followed.

### The unary table has no extrapolable structure, and that is now measured

```
joint MLP     4000 -> 0.091   12000 -> 0.137   24000 -> 0.158    fit 1.000 throughout
digit-wise    4000 -> 0.583   12000 -> 0.597                     fit 1.000
```

Six times the training moves the joint MLP by 0.067. There are only 4,421
trainable unary entries, so at 24000 steps each has been seen **695 times** —
this is not undertraining. `fit 1.000` with `reach 0.158` is memorisation
with nothing generalisable underneath.

I had asserted this earlier from the shape of the loss curve. It needed
measuring, because I had already been wrong once in the same way (predicting
binary would plateau when it was still climbing), and this time the
measurement agreed.

The digit-wise variant converges by 4000 steps at 0.58-0.60. **The gap is not
training, it is whether the structure exists.** D2 gave the binary operators a
per-digit factorisation; the unary operators never got one, and with the
collapse removed that shows as 0.16 against 0.60.

### Binary is still climbing

```
4000 -> 0.562    12000 -> 0.675    24000 -> 0.734
```

and `fit ~= reach` throughout (0.740 / 0.734 at 24000): it is learning the
function, not the entries. 24000 steps covers 14% of the 21.7M trainable
pairs, so there is no reason to expect a plateau yet.

## Self-check #4, re-run: all three presets pass

```
verdict: no preset beats chance -- semantics are hidden

pi_low    table-dependent  0/60    null 0.0000    digit 0.419 vs 0.390
pi_mid    table-dependent  0/47    null 0.2167    digit 0.056 vs 0.240
pi_high   table-dependent  2/18    null 0.8833    digit 0.182 vs 0.879
60/60 answers parsed on every preset, no truncation
```

**The pi_low leak is gone.** It was the collapse, exactly as the mode-baseline
arithmetic predicted.

pi_high is the clearest demonstration that the design's separation works: the
teacher scores **20/42 on skeleton-only items** — the skeleton is meant to be
inferable, and it infers it — while scoring **2/18 on table-dependent items**,
far below the null. Skeleton buyable by reasoning, table not buyable at all.

### And the two numbers together are the evidence H1 wanted

On the same tables, at pi_low:

* **student, gradient descent**: 0.562-0.583, forty times the mode baseline
* **teacher, in-context**: 0.000, not one point above the null

"Some structure gradient descent reaches and in-context learning does not" is
now a measured pair rather than an assumption — and a clean one, because the
collapse used to inflate both sides at once.

### What fixed #4 in the end was `max_tokens`, not chunk size or effort

Two wrong diagnoses before the right one, both extrapolated from a single
observation:

1. "The chunks are too big" — smaller chunks did not help.
2. "The obstacle is effort" — lowering effort made it *worse*
   (out 192000 to 256000, truncations 3 to 4, answers 0/60 either way).

A one-variable-at-a-time scan settled it in 25 minutes:

```
effort=low     max_tok=64000   5 items   end_turn      out=43416   574 chars
effort=medium  max_tok=128000  5 items   end_turn      out=55612   809 chars
effort=low     max_tok=32000   2 items   max_tokens    out=32000     0 chars
```

**The reasoning cost is per call, not per item.** Two items burn the same cap
as five. So chunks should be *larger*, to amortise it, and the binding
constraint is `max_tokens` clearing the fixed cost (~55k at medium, ~43k at
low). At `chunk 30 / max_tokens 128000` every preset came back complete, and
total token spend on pi_high *fell* from 256k to 165k.

The two hours lost to guessing were worth less than the 25 minutes of scan.

## LoRA versus full fine-tuning: my first comparison was confounded

The plan calls for this ablation at the pi=0 end specifically. First results:

| | fit | reach |
|---|---|---|
| binary c=0.25, full | 0.578 | 0.562 |
| binary c=0.25, LoRA r=8 | 0.708 | 0.691 |
| binary c=0.25, LoRA r=32 | 0.744 | **0.715** |

I explained this as the low-rank constraint acting as a regulariser. **That
explanation does not fit its own data**: regularisation shows up as *lower*
fit and higher reach, and LoRA was higher on both.

The actual cause was in my own defaults:

```python
lr = args.lr if args.lr is not None else (1e-4 if args.lora_rank else 1e-5)
```

Full fine-tuning ran at 1e-5 and LoRA at 1e-4 — conventional defaults for
each, and a confound the moment they are compared. The comparison varied two
things at once and could not support any conclusion about rank.

Closing the 2x2 settles it:

| binary c=0.25, 4000 steps | lr 1e-5 | lr 1e-4 |
|---|---|---|
| full | 0.562 | **0.797** |
| LoRA r=32 | *running* | 0.715 |

| unary digit-wise, 4000 steps | lr 1e-5 | lr 1e-4 |
|---|---|---|
| full | 0.583 | **0.697** |
| LoRA r=32 | *running* | 0.655 |

**At a matched learning rate, full fine-tuning beats LoRA on both tasks.** The
apparent LoRA advantage was the 10x learning rate, and the intermediate rate
confirms the trend (binary full at 3e-5: 0.720, between 0.562 and 0.797).

Two things follow. The claim that "#5's full-FT numbers are an upper bound
A6 may not reach with LoRA r=32" is back on — I had briefly retracted it on
the strength of the confounded comparison. And every full fine-tune number in
this entry was produced at 1e-5, which the sweep now shows is well below the
best setting: **the capacity results understate what the student can learn**,
by roughly 0.2 on binary.

## The query budget does not constrain anything

Asked directly, and worth recording because it bears on the budget axis.
Queries *are* charged (`ledger.charge("oracle_query", ...)`, 1e-5 USD each),
but there is no cap, and across eight completed runs:

| arm | queries | query cost | total spent | share |
|---|---|---|---|---|
| a4 | 472 | 4.25 | 6905.5 | 0.062% |
| a6 | 230 | 2.08 | 3257.3 | 0.064% |
| a4 | 215 | 1.96 | 6159.2 | 0.032% |
| a6 | 158 | 1.42 | 3031.1 | 0.047% |
| a2 | 110 | 0.99 | 3060.1 | 0.032% |
| a2 | 103 | 0.93 | 2160.8 | 0.043% |

**No run bought more than 500 queries** against the plan's assumed
`Q ~ 2000`, and query spend never exceeded 0.064% of the budget. The
asymmetry the design rests on — Q against |V| — is currently maintained by
the agent's disinclination to query rather than by the protocol. A different
prompt or a different teacher would move it, and Fig. 1's x-axis would move
with it invisibly.

Still open, still not decided unilaterally.

## The arms, on fixed tables

A2 and A4 completed paired — same instance, same budget, same test set, which
is the first genuinely comparable pair. A6 **failed on OOM**: the agent chose
`lora_rank=64` and trained on 120,000 examples while another job held 59 GB
of the card. Not a bug; the machine was full.

`worker.py` did its job and refused to report the group:

```
1 instance(s) have a failed arm and must be excluded or re-run as a group: [1001]
2/3 ok in 90.0m
```

Re-running A6 on a free card. **Not yet explained**: its first training run
reported `final_loss 0.0` and then `dev_accuracy 0.0` on 13 items. Loss at
zero with dev at zero is not a resource problem and will not fix itself on a
re-run — either the evaluation path is wrong or the agent's synthetic data
does not match the test distribution. To be read from the trace.

---

# 2026-08-28 — the LoRA comparison settled, and a look back at what the experiment is for

## LoRA versus full fine-tuning: settled, and my explanation was wrong twice over

The 2x2 closes:

| binary c=0.25, 4000 steps | lr 1e-5 | lr 1e-4 |
|---|---|---|
| full fine-tune | 0.562 | **0.797** |
| LoRA r=32 | 0.556 | 0.715 |

| unary digit-wise, 4000 steps | lr 1e-5 | lr 1e-4 |
|---|---|---|
| full fine-tune | 0.583 | **0.697** |
| LoRA r=32 | 0.562 | 0.655 |

At a matched learning rate full fine-tuning wins on both tasks, and LoRA at
1e-5 (0.556) sits on top of full at 1e-5 (0.562). **Rank has no effect worth
measuring here; the apparent LoRA advantage was entirely the 10x learning
rate in my own defaults.**

So: the first reading ("low-rank acts as a regulariser") was wrong, and it was
wrong in a way its own data contradicted — regularisation shows as lower fit
and higher reach, and LoRA was higher on both. The second reading (that this
retired the "full-FT is an upper bound A6 may not reach" caveat) was wrong
too, and the caveat stands.

**A consequence that affects every number in this file.** All full fine-tune
capacity runs were done at 1e-5, which the sweep now shows is well short of
the best setting. The capacity results **understate** what the student can
learn — by about 0.24 on binary (0.562 against 0.797) and 0.11 on unary
digit-wise. Directional conclusions hold; absolute levels are a floor, not a
measurement.

## Stepping back: what is this experiment actually for?

Asher asked to stop patching the budget mechanism and re-examine the research
question first. That was the right call and this section is the result.

### The question, in its two halves

> Given a verifiable task the frontier cannot solve alone and a fixed compute
> budget: when does putting capability into a small model's weights beat
> putting it into context or into code? And when the agent chooses for
> itself, how good is that choice?

Two questions: **Q1, the economics** — when is it worth it — and **Q2, the
choice** — does the agent know. Q2 depends on Q1: if the three paths are
interchangeable, "chose well" means nothing.

### The scenario, and the ratio it implicitly requires

The scenario is purchase and amortisation: the teacher does not hold the
knowledge, buys it during prepare, seals, and the student is better at
*deploying* it. **Amortisation carries a ratio requirement — serving has to
be expensive enough that the up-front cost is worth paying.**

Measured, at the full 10^4 test set, against a prepare phase of 3053 H100-s:

| container | deployment cost | vs prepare |
|---|---|---|
| A6, trained student, no prompt | 25.8 | 0.008x |
| A4, program in a sandbox | 4.3 | 0.001x |
| A2, student + prompt (110 facts, as actually bought) | 82.7 | 0.03x |
| A2, student + prompt (2000 facts, as the design assumes) | 1059 | 0.3x |
| A0', frontier re-reads all 2000 facts per query | **752,252** | **246x** |

**Preparing costs about a hundred times more than serving.** In that regime
the rational choice is always "don't train, just answer" — there is nothing
to amortise.

### This is Glyph's task shape, not a bug in the harness

A Glyph item is a short expression mapping to a short answer; a 1.7B model
answers one in milliseconds. **Deployment is cheap by construction.**

The original framing listed four gaps that could make delegation necessary.
The first was throughput amortisation — tasks needing 10^4-10^6 evaluations,
where deployment is expensive *by construction*. Glyph instantiates the
second, the informational gap. It was never the task that carries the
amortisation story, and **T2, which is, does not exist yet — not a line of
code**.

### Where the economics does live

A0' costs 29,000x what A6 costs to serve. That is a real economic story, and
it is already in the design — but its mechanism is **who runs the inference**
(frontier re-reading evidence per query, versus a small model trained once),
not **which container holds the capability**.

A2 sits awkwardly in between: cheap (83) but capacity-limited. The plan casts
it as the expensive context container; the expensive way to use context is
A0'.

## Decision: Glyph carries capacity, T2 carries economics (tentative)

Asher's call, recorded as tentative.

**A — Glyph's question is capacity, and the paper says so.** Glyph measures,
cleanly, which container can hold this knowledge at all. The evidence is
already in hand: on the same tables, gradient descent reaches 0.56-0.58 and
in-context reaches 0.000 against its null. The main figure becomes a (pi, Q)
phase diagram — where does each container succeed — rather than a crossover
curve. H1 is reworded from "a budget threshold B* exists" to "there is a
region where only weights succeeds".

**B — T2 carries amortisation.** Search tasks are throughput-bound, so
deployment is expensive by construction and the money plot is real there.

The point of the split: **each task carries one argument, instead of Glyph
carrying two and failing at one of them.**

### What this changes in the near-term work

**The budget-axis problem drops in urgency.** B is no longer the main
figure's x-axis, so "98% of B is the agent talking" no longer ruins Fig 1.

**A hard cap on Q goes from optional to required.** The new axes are Q
(information bought) and pi. Q has to be a set variable, not a by-product of
how many turns the agent felt like taking — it currently drifts between 103
and 472 across runs, and that drift would become noise on the phase diagram.

**A0' is promoted from a side-check to the load-bearing arm.** Under this
framing the claim *is* "gradient descent reaches what in-context does not",
and A0' is the strongest form of in-context: unlimited context, unlimited
thinking, the whole purchased record. Self-check #4 already gives 0.000, but
from 30 demos; A0' with 2000 bought facts is the real test.

**E1 (budget sweep, 210 runs) is deferred; E2 (the pi phase diagram) is
promoted to the main figure.**

**T2 moves into view earlier than W5**, since it now carries half the paper.
The plan's discipline still holds — nothing on T2 until the minimum loop
closes — so E0 comes first.

### Not done, deliberately

The Phase 1 artifact still states H1 as a budget threshold and organises E1
around a B sweep. It is the source of truth for three documents, so it stays
unchanged until the decision firms up. This entry is the record in the
meantime.

---

# 2026-08-29 — two bugs in the seal seam, and the operational facts worth writing down

## A run must always end with something scoreable

Two failures in the same seam, both found by A6, and both with a direction
that matters.

**The test phase was capped by the prepare budget.** Scoring a sealed
artifact charges the ledger, and a run whose deployment pushed it past `B`
died with `BudgetExhausted` instead of producing a report:

```
BudgetExhausted: 15039.3 / 15000.0 H100-s after gpu_second
  in seal.py evaluate() -> a6_weights answer() -> the gpu_timer's charge
```

The protocol gives the agent a budget for *preparation* and then seals.
Test-phase cost must be **recorded** — A2's per-query cost is the entire
mechanism the comparison is about — but it cannot be **capped** by a budget
the agent can no longer respond to. `Ledger.sealed_mode()` records without
enforcing; `evaluate()` runs inside it.

**The harness's fallback seal could itself be refused.** When every training
attempt OOMed there was no checkpoint, so the harness seal hit the same
validation an agent seal does, returned `{"error": ...}`, and left
`box.sealed` as `None` — which the arm then dereferenced:

```
AttributeError: 'NoneType' object has no attribute 'adapter_path'
```

The fallback existed precisely so that every run yields a comparable data
point, and it had no fallback of its own. A forced seal now always succeeds,
and a forced `program` seal with no program falls back to the student.

**Both bugs delete data preferentially from the arms whose preparation or
deployment costs most** — which are exactly the arms the comparison exists to
measure. An agent whose training all failed should score badly, not vanish: a
base student with nothing attached is a legitimate and very poor artifact,
and the paired design needs the cell filled.

Six tests: the prepare budget still bites before sealing, the test phase
records without being capped, enforcement returns afterwards, an ordinary
seal still needs something to seal, a forced seal cannot be refused, a forced
program seal falls back.

## A6's runtime is set by the agent, not by the task

Same instance, same budget, same preset:

| run | tool calls | GPU s | tool time | shape |
|---|---|---|---|---|
| cli | 11 | 48 | 1.6m | query x5, synth, train x2, eval, seal |
| cli2 | 11 | 31 | 1.6m | one pass, sealed |
| grid_whiten2 | 33 | 1170 | 22.2m | six synth/train/eval cycles |
| pilot | 47 | 1029 | 83.9m | twenty-one query rounds, four seals attempted |
| grid_whiten3 | 43 | 4.6 | 2.9m | **31 train calls, all OOM, retried** |

**Wall clock ranges from about 10 minutes to over 90**, and the spread is
behavioural: an agent that converges in one pass costs a tenth of one that
iterates. `grid_whiten3` is the pathological case — 31 of its 43 calls were
`train`, each failing instantly on OOM, the agent retrying.

`gpu_second` across completed A6 runs spans **4.6 to 1170** — so "training is
1.5% of the budget" was true of one run, not of the arm. It is 0.2%-35%
depending on what the agent chooses.

**This has a consequence for E1.** `worker.py`'s grid timeout defaults to 5400
seconds, and the iterating runs sit right against it. At 210 runs a timeout
that clips the long tail would silently bias the sample toward agents that
converge quickly — the ones doing *less* of what the experiment is about.
Either raise it well above the observed tail or record how many runs it cut.

## Prompt caching was specified and never implemented

The plan §6.3 asks for it explicitly: *"账本的 token 折算跟随真实计费，缓存读
0.1×、缓存写 1.25×，这样 orchestrator 用 prompt caching 省下的钱在账本里也是
真省的"*.

`_charge_usage` reads `cache_read_input_tokens` and
`cache_creation_input_tokens` and prices them correctly. **But the
orchestrator never sets a `cache_control` breakpoint**, so nothing is ever
cached. Across eight completed runs the ledger shows zero cache hits, and
every turn re-sends the whole history at full price.

This is why cost tracks turn count rather than work done: `declare_target`,
which selects one value from a seven-item enum, cost **311 H100-s** — within
15% of the `train` call in the same run.

Not fixed yet. It is ~20 lines and would cut `frontier_in` by roughly 10x;
output tokens are not cacheable, so the total falls by about half rather than
by ten.

## The machines: no single host has both halves

| host | GPUs | Vertex |
|---|---|---|
| `lumen-1`, `lumen-2` | 8 x 80GB, free | same project as lumen-3 |
| `lumen-3` | shared, 4-80GB free depending on the hour | works |
| `node1` (`rh-h100-01`) | 8 x 80GB, free | **403** — ADC lacks `aiplatform.endpoints.predict` |
| `rh-h100-09` | 6 x 80GB free | untested, same project as node1 |
| `rh-h100-10` | connection refused | — |
| `rh-h100-02/03/06/07/08` | no key for this account | — |

The arms need **both** — Vertex for the agent loop, a GPU for the student —
so they can only run where both are present. The capacity checks need only a
GPU and ran on `node1`.

`lumen-3`'s GPUs are shared with someone else's job and have gone from 81GB
free to 4.7GB free and back within a single session; two A6 failures trace to
that. `lumen-1` and `lumen-2` are the right home for arm runs: full cards and
a working Vertex project.

Worth noting for later: pointing node1 at lumen-3's project did **not** help,
so it is the credential rather than the configuration. Getting Vertex access
on node1 would double the hosts able to run arms, which E1's 210 runs will
want.

## Still open, and not from lack of resources

**`dev_accuracy 0.0` after `final_loss 0.0`.** Flagged two entries ago and
still unexplained. A6 trained to zero loss on 120,000 synthesised examples and
then scored zero on 13 dev items. Loss at zero with dev at zero is not a
resource problem: either the evaluation path is wrong, or what the agent
synthesised does not match what it is evaluated on. It survives every re-run
because nothing about it has been fixed.

**A4's solver source has not been read.** The open question from the decode
work — whether making the tables more learnable lets the agent enumerate the
structure into code, which is what rules out `binary_coupling = 0` — can only
be answered by looking at what A4 actually wrote. The source is in its trace,
under `write_code`. Not yet examined.

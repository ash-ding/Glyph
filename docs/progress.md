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

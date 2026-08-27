# Glyph

A hidden-semantics DSL execution benchmark, built for research on **weight-space
delegation**: when a frontier agent faces a task it cannot solve on its own,
does it pay to put capability into *context*, into *code*, or into a small
model's *weights* — and does the agent itself know which?

Glyph is the instrument that makes that question measurable. It is a ruler with
known graduations, not a discovery in its own right.

## What an instance is

A generator draws a hidden interpreter `P`. **The syntax is public; the
semantics are private.**

```
expr:  s1( s3( s0(u2, [v_0_1_0, v_3_13_2, v_6_1_12]) ), b0 )
out:   v_1_4_14
```

Operator names are deliberately opaque — `s0`/`u2`/`b0`, never `map`/`fold` —
so that naming priors leak nothing.

`P` has exactly two halves, because an expression is a tree and a tree node is
only ever a value, a structural operator, or an atomic operator:

| half | what it is | where it comes from | can you write it down? |
|---|---|---|---|
| **skeleton** | the semantics of `s0…s7` | sampled from a finite combinator grammar | **yes** — finitely many rules, by construction |
| **table** | the semantics of `u*` / `b*` | digit embeddings + frozen random MLPs | **no** — you would be transcribing weight matrices |

That asymmetry is the whole design. The skeleton is *cheap to buy and exact to
write* — code's home ground. The table is *impossible to buy out and only ever
approximable* — weights' home ground. **π** measures which half a given
instance's difficulty sits in, and it is a knob you can sweep.

## The agent's situation

It gets the syntax spec and 30 demos for free, then a **metered query oracle**:
it may ask `P` about any well-formed expression, and every query costs budget
(syntax errors included — probing the grammar is not free). At the end it
seals an artifact that must answer 10,000 expressions it has never seen, with
no further access to `P`.

`Q ≈ 2000` queries against `|V| = 4913` values means the unary tables are at
best half covered, and the binary tables (≈24 M pairs) essentially not at all.
That gap is why fitting can beat looking things up.

## Quick start

```bash
pip install -e ".[dev]"
python -m glyph.cli show --preset pi_mid --seed 1001   # one instance, end to end
python -m glyph.cli pi --n-seeds 4                     # measure π across presets
pytest -q                                              # self-checks #1–#3
python scripts/probe_tokenizer.py Qwen/Qwen3-1.7B      # self-check #6
```

## Layout

```
src/glyph/
├── config.py      GlyphConfig + presets (pi_low / pi_mid / pi_high / smoke)
├── grammar.py     public syntax: AST, value codec, printer, parser, checker
├── semantics.py   combinator grammar → skeleton; the trivial-skeleton baseline
├── tables.py      digit embeddings, frozen MLPs, the identity-table baseline
├── interp.py      P = Interpreter(skeleton, tables); lookup logging
├── instance.py    generation, demos, query oracle, splits, tail derivation
├── measure.py     π via two crippled oracles
├── budget.py      the ledger — the single metered entry point
├── trace.py       JSONL trace + response cache (replay, not re-billing)
└── cli.py
```

## Splits

`iid` / `comp` (held-out operator pairs) / `depth` (deeper than any demo) are
fixed at generation time. **`tail` is derived per run** — the items whose table
entries this agent never bought. It cannot be fixed in advance, because the
agent may query anything; and it doubles as a read on how smart its query
strategy was.

There is no `floor` split: the test set is fully solvable and the ceiling is a
clean 100%. Leakage detection lives in the separate E8 audit.

## π

```
a_skel = accuracy(true skeleton + identity table)    # only the table is missing
a_tab  = accuracy(trivial skeleton + true table)     # only the skeleton is missing

π = (1 - a_tab) / ((1 - a_tab) + (1 - a_skel))       # the SKELETON's share
```

π → 1 means the difficulty is programmable (code should win); π → 0 means it is
not (weights should win). π is *measured*, never configured — phase-diagram axes
use `measured_pi()`, never a preset name.

Two things about it are easy to get wrong and both fail silently:

- **The numerator is `L_skel`.** Flip it and the entire phase diagram runs
  backwards while the curves still plot.
- **The trivial skeleton must be textbook defaults, not "do nothing."** If `s0`
  stopped mapping, the atomic operators would never be invoked and the
  baseline would ablate the table too, folding `L_table` into `L_skel`.
- The two baselines are **fixed once and never changed**. π is a ratio relative
  to them, not an absolute quantity.

## Status

Self-checks #1 (round-trip), #2 (oracle exactness), #3 (π ordering) pass;
#6 (tokenizer probe) ships as a script and needs a real tokenizer to run.
Design decisions and open questions are tracked in the Glyph specification.

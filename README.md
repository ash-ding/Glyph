# Glyph

A hidden-semantics DSL execution benchmark, built for research on **weight-space
delegation**: when a frontier agent faces a task it cannot solve on its own,
does it pay to train a small model on what it learns from the environment, or to
just keep reasoning with the frontier — and does the agent itself know which?

Glyph is the instrument that makes that question measurable. It is a ruler with
known graduations, not a discovery in its own right.

**→ [ash-ding.github.io/Glyph](https://ash-ding.github.io/Glyph/)** — the
project page: the first result, how an instance is built, and what is still
open.

## What an instance is

A generator draws a hidden interpreter `P`. **The syntax is public; the
semantics are private.**

```
expr:  s1( s3( s0(u2, [v_a_b_a, v_d_n_c, v_g_b_m]) ), b0 )
out:   v_b_e_o
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

## The protocol

A run has **two phases** and compares **two arms** at equal stop conditions.

**Practice.** The agent gets the syntax spec and 30 demos for free, plus a
*visible validation set* and a **metered query oracle**: it may ask `P` about
any well-formed expression, up to a cap of `Q` queries (syntax errors billed
too — probing the grammar is not free; queries of validation or test items are
refused and unbilled). It may submit answers to the validation set up to 20
times, each time getting back an *aggregate* score only (overall + by depth,
never per item or per split).

**Final.** The oracle is gone. The hidden held-out test — 10,000 expressions,
split `iid` / `comp` / `depth` — is revealed with no answers, and the agent
commits its answers exactly once with `final_answer`. That commit is scored; the
agent never sees a per-item or per-split test score.

**Two arms**, differing in exactly one thing:

| arm | what it has |
|---|---|
| **`no_train`** | the frontier model only |
| **`train`** | the frontier model **plus** a trainable Qwen3-1.7B student — it can build a dataset from what it purchased, fine-tune, and run inference |

The question is whether putting purchased knowledge into a small model's
*weights* beats keeping it in the frontier's *context*, measured at matched
coverage. `Q` is the coverage knob: a few hundred to a thousand queries against
`|V| = 4913` values leaves the unary tables at best half covered and the binary
tables (≈24 M pairs) essentially untouched — and that gap is why fitting can
beat looking things up.

## Quick start

```bash
pip install -e ".[dev]"

# fast, offline self-checks (the `slow` marker covers the real-API + GPU tests)
pytest -m "not slow"

# inspect one instance end to end — data layer only, no API, no GPU
python -c "from glyph.data import PRESETS, generate; \
i = generate(1001, PRESETS['pi_mid']); print(i.measured_pi())"

# tokenizer digit-split probe (self-check #6)
python scripts/probe_tokenizer.py Qwen/Qwen3-1.7B
```

**Running the protocol itself** is a real, *metered* agent run. It needs the
Claude Agent SDK (`claude-agent-sdk==0.2.152`) and its bundled CLI, a reachable
metering gateway, a `bwrap` sandbox host, a GPU (train arm only), and Vertex
credentials — see [Harness](#harness):

```bash
python -m glyph.v2 run  --arm no_train --preset pi_mid --seed 1001
python -m glyph.v2 run  --arm train    --preset pi_mid --seed 1001   # student needs a free GPU
python -m glyph.v2 grid --arms no_train train --presets pi_mid --seeds 1001 1002
```

`python -m glyph.v2 run --help` lists every cap. Defaults are the plan's Global
Constraints: `Q=1000`, `T_p=100`, `T_f=30`, 20 submissions, `V (n_val)=5000`,
USD safety line `$300`, `effort=high`, model `claude-opus-4-8`.

## Harness

The agent runs under the **Claude Agent SDK**, so it keeps the CLI's own `Bash`
and file tools; the eight Glyph tools are registered as **in-process MCP tools**
(`mcp__glyph__*` — see [`docs/tools.md`](docs/tools.md)). Isolation and
accounting are enforced around it:

- the whole CLI runs inside a **`bwrap` sandbox** that hides the source, the
  GPUs, and the credentials;
- all model traffic is routed through a **host-side metering gateway** that
  bills every request, pins the model (a non-pinned model is 403'd), and
  injects the Vertex credentials so the sandbox holds none.

## Viewing runs

Every `python -m glyph.v2 run` writes a self-contained `run.json` into its run
dir — config, the measurement prompts, the task the agent saw, the report, and
the full turn-by-turn transcript (its visible reasoning, every tool call and
result). A static browser viewer reads them:

```bash
python tools/collect_runs.py --out glyph-viewer/runs <your out_root>   # gather run.json files
python -m http.server 8000                                       # from the repo root
# open http://localhost:8000/glyph-viewer/
```

The viewer is a gallery filterable by arm and π; pick a run to see its config,
prompts, and transcript (practice→final, tool inputs/results, errors, reasoning
collapsed by default).

**Running remotely?** Serve on the box and forward a port — no local checkout,
no `scp`:

```bash
python tools/serve_viewer.py --port 8000 <out_root>   # on the remote: collect + serve
ssh -N -L 8000:localhost:8000 <remote>                # on your laptop; open http://localhost:8000/glyph-viewer/
```

See [`glyph-viewer/README.md`](glyph-viewer/README.md).

## Layout

```
src/glyph/
├── data/          the data layer — makes an instance, never scores one
│   ├── config.py      GlyphConfig + presets (pi_low / pi_mid / pi_high / smoke)
│   ├── grammar.py     public syntax: AST, value codec, printer, parser, checker
│   ├── semantics.py   combinator grammar → skeleton; the trivial-skeleton baseline
│   ├── tables.py      digit embeddings, frozen MLPs, the identity-table baseline
│   ├── interp.py      P = Interpreter(skeleton, tables); lookup logging
│   ├── instance.py    generation, demos, query oracle, splits, tail, validation
│   └── measure.py     π via two crippled oracles
├── v2/            protocol v2 — the two-phase, two-arm harness
│   ├── session.py     run state + the (arm, phase) tool-availability matrix
│   ├── tools.py       the eight tool handlers (query / submit / … / final_answer)
│   ├── mcp.py         wraps the handlers as in-process SDK MCP tools
│   ├── harness.py     the orchestrator: phases, caps, the agent loop
│   ├── gateway.py     the metering model gateway (pin, meter, /v1 fix, retry)
│   ├── sandbox.py     the bwrap wrapper; bridge.py is the in-sandbox proxy
│   ├── student.py     the train arm's student pool (build_dataset / train / infer)
│   ├── ledger.py      USD + GPU-second accounting
│   ├── answers.py     answer-file legality + scoring
│   ├── report.py      the final ScoreReport (ceilings, headroom, covariates)
│   └── workspace · prompts · trace · cli
├── train/         full fine-tuning (sft.py) and vLLM inference (infer.py)
├── seal.py        scoring primitives: headroom, score_answers
└── trace.py       JSONL trace + response cache
```

Nothing in `data/` imports from outside it, and `tests/test_data_boundary.py`
holds that line. The arrow runs one way — the protocol layer (`v2/`) imports the
generator, never the reverse — so the generator can be lifted whole into a
second task without dragging along an evaluation protocol written for this one.

## Splits

`iid` / `comp` (held-out operator pairs) / `depth` (deeper than any demo) are
fixed at generation time. The visible validation set is drawn from `iid` only,
generated *after* the test so the two never share an item. **`tail` is derived
per run** — the items whose table entries this agent never bought. It cannot be
fixed in advance, because the agent may query anything; and it doubles as a read
on how smart its query strategy was.

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

**The data layer is frozen** as of 2026-08-31. **Protocol v2 is implemented**
and its harness validated end to end (a real `train`-arm run completes and is
scored); `pytest -m "not slow"` = 194 tests green (the `slow` marker covers the
real-API and GPU tests). The first data-layer measurement:

| oracle | knows | overall |
|---|---|---|
| skeleton ceiling | every structural rule, not one table entry | 0.248 |
| A0′ | frontier model, 2551 of 4913 entries in context, unlimited thinking | 0.258 |
| weights | Qwen3-1.7B fine-tuned on 491 entries | **0.498** |

Split by whether the evidence held the entries an item needed, the frontier
scores **0.976** on the 126 it had and **0.016** on the 374 it did not.
Retrieval saturated, extrapolation at zero — which is the separation the
benchmark was built to produce. Details on the
[project page](https://ash-ding.github.io/Glyph/#result).

Arm results for v2 are not published yet — pilot calibration is in progress. The
v1 explore/prepare/seal protocol (arms A2/A4/A6/A7) was removed in favour of the
two-phase, two-arm v2 design; see the migration entry in `docs/progress.md`.

- [`docs/tools.md`](docs/tools.md) — the v2 tool set and the (arm, phase) matrix
- [`docs/progress.md`](docs/progress.md) — append-only record of what was run
  and what it showed, corrections included
- [`docs/open_questions.md`](docs/open_questions.md) — the standing list of what
  is undecided, edited in place, one GitHub issue per item

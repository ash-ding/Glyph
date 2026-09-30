# PR D — Reference side: public-syntax oracle, per-op ceilings, student model parameter

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The weights oracle trains and answers in the public syntax (`u0(v_…) =`), gains a per-op specialist mode scored on the probe set, and the student base model becomes an end-to-end parameter. Actual GPU runs (0.6B capacity check, weights re-measurement, per-op ceilings) are separate user-gated executions, NOT part of this PR.

**Architecture:** `prompt_unary`/`prompt_binary` in `reference/weights_ceiling.py` switch to `render(AtomApp(...)) + " ="` — one change that flows through both the training stream and `score_ceiling`. `train_student` gains an `ops` restriction; a new `score_probes` scores a trained student on one op's probe items with a seen/unseen split by the training hash. `tools/run_reference.py` gains `--student-model` and a `weights-per-op` oracle; the v2 CLI exposes `--student-model`.

**Tech Stack:** Python 3.11, pytest; torch only inside `[GPU]`-tagged functions (unchanged rule).

**Spec:** `docs/superpowers/specs/2026-09-29-bare-atomics-multistudent-probes-design.md` (§D)

## Global Constraints

- Host lumen1, repo `~/code/Glyph`, branch `feat-reference-d` off `main`. Interpreter `~/miniforge3/envs/glyph/bin/python`.
- No torch at module scope in `reference/` — the CPU tests must import everything without a GPU (the `_student_tables_from` seam exists exactly for this).
- **Comparability break recorded, not hidden:** the published weights number (0.498) was measured under the old private format; progress.md must say all weights columns need re-measurement.
- `pytest -q -m "not slow"` green before the merge. Commits end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Public-syntax prompts

**Files:**
- Modify: `src/glyph/reference/weights_ceiling.py` (`prompt_unary`, `prompt_binary`, imports)
- Test: `tests/reference/` (append to the existing weights-ceiling CPU test file; follow its imports)

**Interfaces:**
- Produces: `prompt_unary(op, i, cfg) == render(AtomApp(op, (i,)), cfg) + " ="`; binary likewise with two args. Everything downstream (stream, score_ceiling) picks it up automatically.

- [ ] **Step 1: Write the failing test**

```python
def test_oracle_prompts_are_public_syntax():
    """The oracle must train/answer in the same surface form the agent uses
    (CLAUDE.md rule 7). The prompt minus ' =' must parse under the public
    grammar as a bare atomic application."""
    from glyph.data import PRESETS, parse
    from glyph.data.grammar import AtomApp
    from glyph.reference.weights_ceiling import prompt_binary, prompt_unary
    cfg = PRESETS["smoke"]
    pu = prompt_unary("u0", 3, cfg)
    pb = prompt_binary("b0", 3, 4, cfg)
    for p, want in ((pu, AtomApp("u0", (3,))), (pb, AtomApp("b0", (3, 4)))):
        assert p.endswith(" =")
        assert parse(p[:-2], cfg) == want
```

- [ ] **Step 2: Run to verify failure**

Run: `~/miniforge3/envs/glyph/bin/python -m pytest tests/reference -q -k public_syntax`
Expected: FAIL — old format `"u0 v_… ="` does not parse.

- [ ] **Step 3: Implement.** In `weights_ceiling.py`, extend the grammar import with `AtomApp, render` and replace the two helpers:

```python
def prompt_unary(op: str, i: int, cfg) -> str:
    return render(AtomApp(op, (i,)), cfg) + " ="


def prompt_binary(op: str, i: int, j: int, cfg) -> str:
    return render(AtomApp(op, (i, j)), cfg) + " ="
```

(`generate_answers` decodes `txt.split()[0]` — value literals contain no spaces, so decoding is unaffected.)

- [ ] **Step 4: Run the reference tests**

Run: `~/miniforge3/envs/glyph/bin/python -m pytest tests/reference -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/glyph/reference/weights_ceiling.py tests/reference
git commit -m "feat(reference): weights oracle trains and answers in the public syntax"
```

---

### Task 2: `ops` restriction + `score_probes`

**Files:**
- Modify: `src/glyph/reference/weights_ceiling.py`
- Test: `tests/reference/` (same file as Task 1)

**Interfaces:**
- Produces: `train_student(inst, seen_frac, *, model=..., ops: list[str] | None = None, ...)` — restricts the training stream to the given atomic ops. `score_probes(inst, probe_items, seen_frac, answer_unary, answer_binary) -> dict` — CPU-pure scorer over two answer callables (the same seam as `_student_tables_from`), returning `{"overall", "n", "seen": {"n","acc"}, "unseen": {"n","acc"}}`, where seen/unseen is decided by the TRAINING hash (`seen_u`/`seen_b`) — the reference-side analogue of the agent report's LookupLog split. A thin `[GPU]` wrapper `score_probes_model(inst, model, tok, probe_items, seen_frac, device)` builds the callables from `generate_answers`.

- [ ] **Step 1: Write the failing tests**

```python
def test_score_probes_perfect_student_and_seen_split():
    """A perfect student scores 1.0 on its op's probes; seen/unseen n's are
    decided by the training hash, not by any query log."""
    from glyph.data import PRESETS, generate
    from glyph.data.probe import probe_set
    from glyph.reference.weights_ceiling import score_probes, seen_b, seen_u
    inst = generate(1001, PRESETS["smoke"])
    frac = 0.5
    probes = [t for t in probe_set(inst, n_per_op=10) if t.split == "u0"]
    out = score_probes(
        inst, probes, frac,
        answer_unary=lambda name, i: inst.tables.apply_unary(name, i),
        answer_binary=lambda name, i, j: inst.tables.apply_binary(name, i, j))
    assert out["overall"] == 1.0 and out["n"] == len(probes)
    n_seen = sum(1 for t in probes for (_, i) in t.needs_u if seen_u(i, frac))
    assert out["seen"]["n"] == n_seen
    assert out["seen"]["n"] + out["unseen"]["n"] == out["n"]


def test_train_student_ops_restriction_signature():
    """CPU-only: the ops filter must reject unknown op names before any GPU
    work starts (fail fast on a typo, not 6000 steps in)."""
    import pytest
    from glyph.data import PRESETS, generate
    from glyph.reference.weights_ceiling import train_student
    inst = generate(1001, PRESETS["smoke"])
    with pytest.raises(ValueError):
        train_student(inst, 0.1, ops=["u9"])
```

- [ ] **Step 2: Run to verify failure**

Run: `~/miniforge3/envs/glyph/bin/python -m pytest tests/reference -q -k "score_probes or ops_restriction"`
Expected: FAIL — `ImportError: cannot import name 'score_probes'`.

- [ ] **Step 3: Implement.** In `weights_ceiling.py`:

`train_student` signature gains `ops: list[str] | None = None` (keyword). At the TOP of the function, before any torch import:

```python
    cfg = inst.cfg
    us, bs = unary_names(cfg), binary_names(cfg)
    if ops is not None:
        unknown = [o for o in ops if o not in us + bs]
        if unknown:
            raise ValueError(f"unknown atomic op(s) {unknown!r}")
        us = [o for o in us if o in ops]
        bs = [o for o in bs if o in ops]
```

(then delete the old `cfg`/`us`/`bs` lines below the imports). In `stream`, the unary/binary coin flip must respect empty pools:

```python
                if (rng.random() < 0.5 and us) or not bs:
                    ...unary branch (unchanged)...
                else:
                    ...binary branch (unchanged)...
```

New functions after `score_ceiling`:

```python
def score_probes(inst, probe_items, seen_frac, answer_unary, answer_binary) -> dict:
    """Score a student on bare-atomic probe items, split seen/unseen by the
    TRAINING hash (`seen_u`/`seen_b`) -- the reference-side analogue of the
    agent report's LookupLog split. CPU-pure via the answer-callable seam."""
    cfg = inst.cfg
    need_u, need_b = set(), set()
    for t in probe_items:
        need_u |= set(t.needs_u)
        need_b |= set(t.needs_b)
    tables = _student_tables_from(answer_unary, answer_binary, need_u, need_b, cfg)

    def _is_seen(t) -> bool:
        for (_, i) in t.needs_u:
            return seen_u(i, seen_frac)
        for (_, i, j) in t.needs_b:
            return seen_b(i, j, seen_frac)
        return False

    seen = [t for t in probe_items if _is_seen(t)]
    unseen = [t for t in probe_items if not _is_seen(t)]

    def _acc(sub):
        return _score_on(inst, tables, sub)["overall"] if sub else None

    return {
        "overall": _acc(probe_items),
        "n": len(probe_items),
        "seen": {"n": len(seen), "acc": _acc(seen)},
        "unseen": {"n": len(unseen), "acc": _acc(unseen)},
    }


def score_probes_model(inst, model, tok, probe_items, seen_frac, *,
                       device="cuda") -> dict:
    """[GPU] `score_probes` with the callables built from one batched
    `generate_answers` pass per operator kind."""
    cfg = inst.cfg
    need_u, need_b = set(), set()
    for t in probe_items:
        need_u |= set(t.needs_u)
        need_b |= set(t.needs_b)
    nu, nb = sorted(need_u), sorted(need_b)
    au = generate_answers(model, tok, cfg, [prompt_unary(o, i, cfg) for o, i in nu], device)
    ab = generate_answers(model, tok, cfg, [prompt_binary(o, i, j, cfg) for o, i, j in nb], device)
    u_ans, b_ans = dict(zip(nu, au)), dict(zip(nb, ab))
    return score_probes(inst, probe_items, seen_frac,
                        lambda name, i: u_ans.get((name, i), -1),
                        lambda name, i, j: b_ans.get((name, i, j), -1))
```

Note `_score_on` calls `inst.ceilings(items)` for headroom — on probe items the skeleton oracle just evaluates the bare atomic with identity tables, which is fine; `score_probes` only reads `"overall"` from `_score_on`.

- [ ] **Step 4: Run to verify pass**

Run: `~/miniforge3/envs/glyph/bin/python -m pytest tests/reference tests/test_data_boundary.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/glyph/reference/weights_ceiling.py tests/reference
git commit -m "feat(reference): per-op training restriction and probe scoring with seen/unseen split"
```

---

### Task 3: CLI plumbing — `--student-model` and `weights-per-op`

**Files:**
- Modify: `tools/run_reference.py` (`ALL_ORACLES`, args, main loop), `src/glyph/v2/cli.py` (`--student-model` on `run` and `grid`)
- Test: `tests/v2/test_cli.py` and `tests/reference/`'s runner test if one exists — run both; add a parse-level test

**Interfaces:**
- Produces: `run_reference.py --only weights-per-op --seen-frac … --student-model …` → per instance, for each atomic op: train a specialist on that op's cells only and score it on that op's probes; merged as oracle `"weights_per_op"` with payload `{frac: {op: score_probes(...)}}`. v2 CLI `--student-model` (default `Qwen/Qwen3-1.7B`) → `RunConfig.student_model`.

- [ ] **Step 1: Add a parse-level failing test** (in `tests/v2/test_cli.py`, following its existing arg-parse test style)

```python
def test_cli_exposes_student_model():
    from glyph.v2.cli import build_parser
    ns = build_parser().parse_args(
        ["run", "--arm", "no_train", "--student-model", "Qwen/Qwen3-0.6B"])
    assert ns.student_model == "Qwen/Qwen3-0.6B"
```

(If `cli.py` has no `build_parser` seam, adapt to however `test_cli.py` invokes parsing today — the assertion is the contract.)

- [ ] **Step 2: Run to verify failure, then implement.**

`cli.py`: add to BOTH the `run` and `grid` subparsers:

```python
    s.add_argument("--student-model", default="Qwen/Qwen3-1.7B",
                   help="the trainable student's base model (train arm); "
                        "NOT the frontier model id")
```

and pass `student_model=args.student_model` where RunConfig is built (both paths).

`run_reference.py`:

- `ALL_ORACLES` gains `"weights-per-op"`; `--only` help text updated.
- `--student-model` argument, default `"Qwen/Qwen3-1.7B"`.
- the weights branch passes `model=args.student_model` into `train_student`.
- new branch after the weights branch:

```python
        if "weights-per-op" in oracles:
            # [GPU] one specialist per atomic op, scored on that op's probes.
            from glyph.data import probe_set
            from glyph.data.grammar import binary_names, unary_names
            from glyph.reference.weights_ceiling import score_probes_model
            probes = probe_set(inst)
            by_frac = {}
            for frac in args.seen_frac:
                per_op = {}
                for op in unary_names(inst.cfg) + binary_names(inst.cfg):
                    model, tok = train_student(inst, frac, ops=[op],
                                               model=args.student_model)
                    op_items = [t for t in probes if t.split == op]
                    per_op[op] = score_probes_model(inst, model, tok, op_items, frac)
                    del model
                by_frac[str(frac)] = per_op
            merge_reference(args.out, instance_id, "weights_per_op", by_frac)
            print(f"{instance_id}: weights-per-op done for seen_frac={args.seen_frac}")
```

(`train_student`'s existing signature has `model` as a keyword — keep the call keyword-only. Free the model between ops with `del` so 5 sequential fine-tunes fit one GPU.)

- [ ] **Step 3: Run the CLI and reference tests**

Run: `~/miniforge3/envs/glyph/bin/python -m pytest tests/v2/test_cli.py tests/reference -q`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tools/run_reference.py src/glyph/v2/cli.py tests/v2/test_cli.py
git commit -m "feat(cli): --student-model everywhere; weights-per-op reference oracle"
```

---

### Task 4: Docs, full suite, PR

**Files:**
- Modify: `docs/data-validation.en.md`, `docs/data-validation.zh.md` (weights oracle format + the new per-op oracle), `docs/open_questions.md` (starvation item: option (c) decided — bare-atomic access opened; note what remains open there), `docs/progress.md`

- [ ] **Step 1: Full fast suite**

Run: `~/miniforge3/envs/glyph/bin/python -m pytest -q -m "not slow"`
Expected: all pass.

- [ ] **Step 2: Docs.**
  - `data-validation.{en,zh}.md` §3.4 (weights ceiling): note the surface form is now the public syntax (`u0(v_…) =`), that this **invalidates comparability with the previously published 0.498**, and add a short §3.4-adjacent paragraph for `weights-per-op` (one specialist per op, trained on that op's cells at `seen_frac`, scored on that op's probes with the training-hash seen/unseen split — directly comparable to the agent report's `probe.by_op` and to a multi-specialist train arm).
  - `open_questions.md` starvation item: record that option (c) is decided — bare-atomic queries are legal (PR A), the train arm can now buy the same kind of data the oracle trains on (format aligned in this PR); what remains open there: the steps∝dataset-size coupling and the Q budget (#15).
  - `progress.md`: PR D entry — what changed, the comparability-break sentence, the three deferred [GPU] runs this unlocks (0.6B capacity check via `scripts/capacity_check.py --model Qwen/Qwen3-0.6B`, weights re-measurement, per-op ceilings), command + pass count.

- [ ] **Step 3: Commit docs, push, open PR; stop for the user's merge approval**

```bash
git add docs
git commit -m "docs: reference-side changes (public-syntax oracle, per-op ceilings, model param)"
git push -u origin feat-reference-d
gh pr create --title "feat: reference side — public-syntax oracle, per-op ceilings, student-model parameter" --body-file <body>
```

**Stop: the user approves the merge.**

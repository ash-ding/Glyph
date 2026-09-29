# PR A — Bare atomic expressions (`AtomApp`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make bare atomic-operator applications (`u0(v_a_b_c)`, `b1(v_x, v_y)`, value-literal args only) legal expressions at the query layer, leaving generation untouched.

**Architecture:** One new frozen dataclass `AtomApp` joins the `Expr` union in `data/grammar.py`; parser/printer/checker/`result_type`/`syntax_spec` each grow one branch; `data/interp.py` evaluates it as a single logged table lookup. Nothing in generation (`_sample`, `_make_*`) emits it, so the 15 frozen instances regenerate bit-identically — `tests/test_frozen_instances.py` is the proof and must pass unchanged.

**Tech Stack:** Python 3.11, numpy-only data layer, pytest.

**Spec:** `docs/superpowers/specs/2026-09-29-bare-atomics-multistudent-probes-design.md` (§A)

## Global Constraints

- Work on host lumen1, repo `~/code/Glyph`, branch `feat-atomapp` off `main`. Interpreter: `~/miniforge3/envs/glyph/bin/python` (a bare `python` resolves to a conda base without pytest).
- The data layer imports numpy and stdlib only; no torch.
- `pytest -q -m "not slow"` green before the merge; `tests/test_frozen_instances.py` must pass **without modification**.
- `test_syntax_spec_leaks_no_semantics` must stay green: new spec lines name form and arity only, never meaning.
- Commit messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: `AtomApp` node — AST, parser, printer, `result_type`, exports

**Files:**
- Modify: `src/glyph/data/grammar.py` (AST section ~line 120–170, printer `render`, parser `_P.expr`, `result_type`)
- Modify: `src/glyph/data/__init__.py` (export `AtomApp`)
- Test: `tests/test_grammar.py`

**Interfaces:**
- Produces: `AtomApp(op: str, args: tuple[int, ...])` frozen dataclass; `Expr = Val | Lit | App | AtomApp`; `result_type(AtomApp(...)) == "VAL"`; `depth(AtomApp(...)) == 1` (falls out of the existing `depth` — the args generator matches no `(Val, Lit, App)`, so `1 + max(default=0)`); `parse`/`render` round-trip the form `op(value[, value])`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_grammar.py`; note `AtomApp` import must be added to the existing `from glyph.data.grammar import (...)` line)

```python
@pytest.mark.parametrize("cfg", CFGS)
def test_atomapp_roundtrip(cfg):
    """render -> parse -> render is a fixed point for bare atomic applications."""
    from glyph.data.grammar import AtomApp, depth, result_type
    rng = np.random.default_rng(23)
    for _ in range(100):
        i, j = int(rng.integers(cfg.n_values)), int(rng.integers(cfg.n_values))
        u = AtomApp(f"u{int(rng.integers(cfg.n_unary))}", (i,))
        b = AtomApp(f"b{int(rng.integers(cfg.n_binary))}", (i, j))
        for e in (u, b):
            src = render(e, cfg)
            back = parse(src, cfg)
            assert back == e
            assert render(back, cfg) == src
            assert result_type(back) == "VAL"
            assert depth(back) == 1


def test_atomapp_parse_from_source():
    """The written form parses to the node, not to a Val fallback."""
    from glyph.data.grammar import AtomApp
    cfg = PRESETS["smoke"]
    v0, v1 = render_value(0, cfg), render_value(1, cfg)
    assert parse(f"u0({v0})", cfg) == AtomApp("u0", (0,))
    assert parse(f"b0({v0}, {v1})", cfg) == AtomApp("b0", (0, 1))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/miniforge3/envs/glyph/bin/python -m pytest tests/test_grammar.py -q -k atomapp`
Expected: FAIL — `ImportError: cannot import name 'AtomApp'`

- [ ] **Step 3: Implement**

In `src/glyph/data/grammar.py`, after the `App` dataclass:

```python
@dataclass(frozen=True)
class AtomApp:
    """A bare atomic-operator application, e.g. u0(v_a_b_c) or b1(v_x, v_y).

    Args are value indices: value literals only, no nesting.  This is the
    query-layer channel for buying clean table cells; generation never emits
    it, so test/val distributions (and frozen-instance fingerprints) are
    untouched.
    """
    op: str
    args: tuple[int, ...]


Expr = Val | Lit | App | AtomApp
```

(Replace the existing `Expr = Val | Lit | App` line.)

In `result_type`, before the `App` shape lookup:

```python
    if isinstance(e, AtomApp):
        return "VAL"
```

In `render`, before the `App` branch (the App loop would stringify raw ints):

```python
    if isinstance(e, AtomApp):
        return f"{e.op}({', '.join(render_value(i, cfg) for i in e.args)})"
```

In `_P.expr`, after the structural-operator branch and before the `Val` fallback:

```python
        if (t[0] in ("u", "b") and t[1:].isdigit()
                and self.i + 1 < len(self.toks) and self.toks[self.i + 1] == "("):
            op = self.take()
            self.take("(")
            args = [parse_value(self.take(), self.cfg)]
            while self.peek() == ",":
                self.take(",")
                args.append(parse_value(self.take(), self.cfg))
            self.take(")")
            return AtomApp(op, tuple(args))
```

(No change to `depth` or `op_pairs`: `depth` returns 1 via its `default=0` path, and `op_pairs` only recurses into `App`, so an `AtomApp` contributes no pairs — both are asserted by tests.)

In `src/glyph/data/__init__.py`: add `AtomApp` to the `from .grammar import (...)` list and to `__all__`, next to `Expr`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/miniforge3/envs/glyph/bin/python -m pytest tests/test_grammar.py -q`
Expected: all PASS (old round-trip tests included — the parser change must not disturb App/Val/Lit parsing)

- [ ] **Step 5: Commit**

```bash
git add src/glyph/data/grammar.py src/glyph/data/__init__.py tests/test_grammar.py
git commit -m "feat(grammar): AtomApp — bare atomic applications parse, print, type"
```

---

### Task 2: `check()` — well-formedness for `AtomApp`

**Files:**
- Modify: `src/glyph/data/grammar.py` (`check`'s `walk`)
- Test: `tests/test_grammar.py`

**Interfaces:**
- Consumes: `AtomApp` from Task 1.
- Produces: `check(AtomApp(...), cfg)` raises `SyntaxError` for a disabled op, wrong arity, or out-of-range value; passes otherwise.

- [ ] **Step 1: Write the failing test**

```python
def test_atomapp_check():
    from glyph.data.grammar import AtomApp
    cfg = PRESETS["smoke"]
    v0, v1 = render_value(0, cfg), render_value(1, cfg)
    # legal
    check(parse(f"u0({v0})", cfg), cfg)
    check(parse(f"b0({v0}, {v1})", cfg), cfg)
    # wrong arity, disabled op, value out of range
    for e in (AtomApp("u0", (0, 1)), AtomApp("b0", (0,)),
              AtomApp(f"u{cfg.n_unary}", (0,)), AtomApp(f"b{cfg.n_binary}", (0, 1)),
              AtomApp("u0", (cfg.n_values,))):
        with pytest.raises(SyntaxError):
            check(e, cfg)
    # nesting and non-literal args never reach check: the parser rejects them
    for src in (f"u0(u1({v0}))", f"u0([{v0}, {v1}])", f"u0(s2([{v0}, {v1}]))", "u0()"):
        with pytest.raises(SyntaxError):
            parse(src, cfg)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/miniforge3/envs/glyph/bin/python -m pytest tests/test_grammar.py::test_atomapp_check -q`
Expected: FAIL — `check` currently falls through to `ops.get(node.op)` → it does raise for AtomApp, but with the wrong message and it never validates arity/range; the `check(parse(f"u0({v0})"...)` line raises, which the test treats as failure.

- [ ] **Step 3: Implement** — in `check`'s `walk`, after the `Lit` branch and before the structural lookup:

```python
        if isinstance(node, AtomApp):
            if node.op in us:
                want = 1
            elif node.op in bs:
                want = 2
            else:
                raise SyntaxError(f"operator {node.op!r} is not enabled")
            if len(node.args) != want:
                raise SyntaxError(f"{node.op} takes {want} value argument(s)")
            for i in node.args:
                if not 0 <= i < nv:
                    raise SyntaxError(f"value out of range: {i}")
            return
```

(`AtomApp` must be in the module's own namespace already — it is defined in this file.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/miniforge3/envs/glyph/bin/python -m pytest tests/test_grammar.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/glyph/data/grammar.py tests/test_grammar.py
git commit -m "feat(grammar): check() validates AtomApp arity, op, value range"
```

---

### Task 3: Interpreter — one logged table lookup

**Files:**
- Modify: `src/glyph/data/interp.py` (`Interpreter._eval`; import line)
- Test: `tests/test_interp.py`

**Interfaces:**
- Consumes: `AtomApp` from Task 1.
- Produces: `Interpreter.eval(AtomApp("u0", (i,)))` == `tables.apply_unary("u0", i)`; binary likewise; the lookup lands in `LookupLog.unary` / `.binary`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_interp.py`, matching its existing imports/style — it already builds instances via `generate`)

```python
def test_atomapp_eval_is_one_logged_lookup():
    from glyph.data import PRESETS, generate
    from glyph.data.grammar import AtomApp
    inst = generate(31, PRESETS["smoke"])
    interp, cfg = inst.P, inst.cfg
    i, j = 5 % cfg.n_values, 7 % cfg.n_values   # stay in range on any preset
    out, log = interp.eval_logged(AtomApp("u0", (i,)))
    assert out == inst.tables.apply_unary("u0", i)
    assert log.unary == {("u0", i)} and not log.binary
    out, log = interp.eval_logged(AtomApp("b0", (i, j)))
    assert out == inst.tables.apply_binary("b0", i, j)
    assert log.binary == {("b0", i, j)} and not log.unary
```

(If `tests/test_interp.py` reaches the interpreter/tables under different attribute names, adapt the access path to what that file already uses — the assertions stay the same.)

- [ ] **Step 2: Run test to verify it fails**

Run: `~/miniforge3/envs/glyph/bin/python -m pytest tests/test_interp.py -q -k atomapp`
Expected: FAIL — `KeyError`/`TypeError` in `_eval` (AtomApp hits the App path and looks up a structural shape that does not exist)

- [ ] **Step 3: Implement** — in `src/glyph/data/interp.py`, add `AtomApp` to the `from .grammar import` line, and in `_eval` after the `Lit` branch:

```python
        if isinstance(e, AtomApp):
            if len(e.args) == 1:
                return self._u(e.op, e.args[0], log)
            return self._b(e.op, e.args[0], e.args[1], log)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/miniforge3/envs/glyph/bin/python -m pytest tests/test_interp.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/glyph/data/interp.py tests/test_interp.py
git commit -m "feat(interp): evaluate AtomApp as a single logged table lookup"
```

---

### Task 4: Query path + public syntax spec

**Files:**
- Modify: `src/glyph/data/grammar.py` (`syntax_spec`)
- Test: `tests/test_query_policy.py` (query acceptance/billing/logging), `tests/test_grammar.py` (leak test must still pass, no edit expected)

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces: `inst.query("u0(v_…)")` answers, bills one query, logs the cell; `query_violation` returns `None` for it under both policies; `syntax_spec` documents the form.

- [ ] **Step 1: Write the failing test** (append to `tests/test_query_policy.py`, reusing however that file builds its instance — it already exercises `query_violation`)

```python
def test_bare_atomic_query_is_legal_billed_and_logged():
    from glyph.data import PRESETS, generate, render_value
    inst = generate(47, PRESETS["smoke"])
    cfg = inst.cfg
    i = 9 % cfg.n_values   # stay in range on any preset
    src = f"u0({render_value(i, cfg)})"
    for policy in ("strict", "open"):
        assert inst.query_violation(src, policy) is None
    n0, before = inst.query_count, len(inst.query_log)
    out = inst.query(src)
    assert out == render_value(inst.tables.apply_unary("u0", i), cfg)
    assert inst.query_count == n0 + 1
    assert ("u0", i) in inst.query_log.unary and len(inst.query_log) == before + 1


def test_syntax_spec_documents_bare_atomics():
    from glyph.data import PRESETS, syntax_spec
    s = syntax_spec(PRESETS["smoke"])
    assert "u*(value) -> value" in s
    assert "b*(value, value) -> value" in s
```

- [ ] **Step 2: Run tests to verify the spec one fails**

Run: `~/miniforge3/envs/glyph/bin/python -m pytest tests/test_query_policy.py -q -k "bare_atomic or documents"`
Expected: the query test may already PASS (Tasks 1–3 make the whole path legal — that is the point: verify it); the syntax-spec test FAILS.

- [ ] **Step 3: Implement** — in `syntax_spec`, after the structural-operators block (the `for name, shape in enabled_ops(cfg)` loop) and before the `Integer arguments` line:

```python
    lines += [
        "Atomic operators may also be applied directly to value literals:",
        "    u*(value) -> value",
        "    b*(value, value) -> value",
        "Their arguments must be value literals -- no nesting, no lists.",
    ]
```

- [ ] **Step 4: Run tests, including the leak gate**

Run: `~/miniforge3/envs/glyph/bin/python -m pytest tests/test_query_policy.py tests/test_grammar.py -q`
Expected: PASS, including `test_syntax_spec_leaks_no_semantics` (the added lines state form and arity only)

- [ ] **Step 5: Commit**

```bash
git add src/glyph/data/grammar.py tests/test_query_policy.py
git commit -m "feat(grammar): document bare atomics in the public syntax spec"
```

---

### Task 5: Full-suite proof, docs, PR

**Files:**
- Modify: `docs/data-generation.en.md`, `docs/data-generation.zh.md` (syntax section), `docs/progress.md` (append entry)
- No code changes.

**Interfaces:**
- Consumes: Tasks 1–4 complete on `feat-atomapp`.

- [ ] **Step 1: Run the full fast suite — the frozen-integrity gate is the point**

Run: `~/miniforge3/envs/glyph/bin/python -m pytest -q -m "not slow"`
Expected: all pass, ~270 tests, `tests/test_frozen_instances.py` untouched and green — this is the proof that generation did not move.

- [ ] **Step 2: Docs.** In both `docs/data-generation.en.md` and `docs/data-generation.zh.md`, find the public-syntax description and add one short paragraph (English / Chinese respectively):

> Bare atomic applications are legal expressions: `u*(value)` and
> `b*(value, value)`, with value-literal arguments only (no nesting, no
> lists). Generation never emits them — they exist so an agent can buy
> individual table cells at query time. One cell costs one query.

Append to `docs/progress.md` (under a `## 2026-09-29 — PR A: bare atomic expressions` heading): what changed (AtomApp in grammar/interp/spec), the command run (`pytest -q -m "not slow"`), the pass count, and the sentence "Generation untouched; `test_frozen_instances.py` passed unmodified."

- [ ] **Step 3: Commit docs**

```bash
git add docs/data-generation.en.md docs/data-generation.zh.md docs/progress.md
git commit -m "docs: bare atomic applications in data-generation + progress entry"
```

- [ ] **Step 4: Push and open the PR**

```bash
git push -u origin feat-atomapp
gh pr create --title "feat: bare atomic expressions (AtomApp) at the query layer" --body-file <body>
```

PR body: summary of §A of the spec, the full-suite pass count, the frozen-integrity sentence, ending with the "Generated with Claude Code" line. **Stop: the user approves the merge.**

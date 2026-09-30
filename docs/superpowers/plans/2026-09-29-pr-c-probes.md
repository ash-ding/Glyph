# PR C — Per-op probe set Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A fingerprint-exempt, per-atomic-op probe set delivered with the Final test, mandatory in `final_answer`, scored in a separate `probe.by_op` report block (never part of `overall`), each op split seen/unseen via the run's `LookupLog`.

**Architecture:** New data-layer module `data/probe.py` generates bare-atomic `TestItem`s (split = op name) from a dedicated RNG stream. The harness attaches them to the session; `workspace.write_test_file` appends them to `final/test.jsonl` with `probe_`-prefixed ids; `t_final_answer` checks legality against test+probes combined; `report.build_report` adds the `probe` block; the viewer renders it.

**Tech Stack:** Python 3.11, numpy-only data layer, pytest; vanilla-JS static viewer.

**Spec:** `docs/superpowers/specs/2026-09-29-bare-atomics-multistudent-probes-design.md` (§C)

## Global Constraints

- Host lumen1, repo `~/code/Glyph`, branch `feat-probes` off `main`. Interpreter `~/miniforge3/envs/glyph/bin/python`.
- Generation (`_sample`, `_make_*`) untouched; `tests/test_frozen_instances.py` must pass unmodified. Probes are generated on demand, never hashed into the fingerprint.
- `tests/test_data_boundary.py` must stay green: `data/probe.py` imports only from within `glyph.data`.
- Old sessions without probes must keep working: every protocol-layer read of probes goes through `getattr(session, "probes", [])`.
- `pytest -q -m "not slow"` green before the merge. Commits end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: `data/probe.py` — the generator

**Files:**
- Create: `src/glyph/data/probe.py`
- Modify: `src/glyph/data/__init__.py` (export `probe_set`)
- Test: `tests/test_probe.py` (new)

**Interfaces:**
- Produces: `probe_set(inst, n_per_op=100) -> list[TestItem]` — for each enabled atomic op, `n_per_op` bare-atomic items (unary capped at `n_values` distinct inputs; binary pairs distinct), `split` = op name, `needs_u`/`needs_b` = that one cell, `answer_src` from the true tables. Deterministic per instance; disjoint from `inst.test`/`inst.val` by form.

- [ ] **Step 1: Write the failing tests** (`tests/test_probe.py`)

```python
"""probe_set: deterministic per-op bare-atomic probes, outside the fingerprint."""
from glyph.data import PRESETS, generate, parse, render_value
from glyph.data.grammar import AtomApp, binary_names, unary_names
from glyph.data.probe import probe_set


def test_probe_set_shape_and_determinism():
    inst = generate(1001, PRESETS["smoke"])
    cfg = inst.cfg
    a, b = probe_set(inst, n_per_op=20), probe_set(inst, n_per_op=20)
    assert [(t.expr_src, t.answer_src, t.split) for t in a] == \
           [(t.expr_src, t.answer_src, t.split) for t in b]
    ops = unary_names(cfg) + binary_names(cfg)
    for op in ops:
        items = [t for t in a if t.split == op]
        want = min(20, cfg.n_values) if op.startswith("u") else 20
        assert len(items) == want
        assert len({t.expr_src for t in items}) == want   # distinct cells
    assert {t.split for t in a} == set(ops)


def test_probe_items_are_correct_single_cells():
    inst = generate(1001, PRESETS["smoke"])
    cfg = inst.cfg
    for t in probe_set(inst, n_per_op=10):
        e = parse(t.expr_src, cfg)
        assert isinstance(e, AtomApp)
        if len(e.args) == 1:
            assert t.needs_u == frozenset({(e.op, e.args[0])}) and not t.needs_b
            out = inst.tables.apply_unary(e.op, e.args[0])
        else:
            assert t.needs_b == frozenset({(e.op, *e.args)}) and not t.needs_u
            out = inst.tables.apply_binary(e.op, *e.args)
        assert t.answer_src == render_value(out, cfg)


def test_probes_disjoint_from_test_and_val():
    inst = generate(1001, PRESETS["smoke"])
    fixed = {t.expr_src for t in inst.test} | {t.expr_src for t in inst.val}
    assert not ({t.expr_src for t in probe_set(inst, n_per_op=20)} & fixed)


def test_probe_cells_are_queryable_not_refused():
    inst = generate(1001, PRESETS["smoke"])
    t = probe_set(inst, n_per_op=5)[0]
    assert inst.query_violation(t.expr_src, "strict") is None
```

- [ ] **Step 2: Run to verify failure**

Run: `~/miniforge3/envs/glyph/bin/python -m pytest tests/test_probe.py -q`
Expected: FAIL — `ModuleNotFoundError: glyph.data.probe`

- [ ] **Step 3: Implement** (`src/glyph/data/probe.py`)

```python
"""Per-op probe set: bare-atomic items generated OUTSIDE the instance.

Derived on demand from a dedicated RNG stream, so it is exempt from the
frozen-instance fingerprint (which hashes demos/test/held_pairs/val only).
`split` carries the op name, which makes score_file's by_split grouping a
per-op score for free.  Probe cells are deliberately NOT refused by the
query oracle -- refusing them would leak which cells are probes; seen vs
unseen is derived after the fact from the run's LookupLog (same stance as
`tail`).
"""

from __future__ import annotations

import numpy as np

from .grammar import AtomApp, binary_names, render, render_value, unary_names
from .instance import GlyphInstance, TestItem

PROBE_SALT = 20260929


def probe_set(inst: GlyphInstance, n_per_op: int = 100) -> list[TestItem]:
    cfg = inst.cfg
    rng = np.random.default_rng((inst.seed, PROBE_SALT))
    out: list[TestItem] = []

    for op in unary_names(cfg):
        n = min(n_per_op, cfg.n_values)
        for i in rng.choice(cfg.n_values, size=n, replace=False):
            i = int(i)
            e = AtomApp(op, (i,))
            out.append(TestItem(
                expr_src=render(e, cfg),
                answer_src=render_value(inst.tables.apply_unary(op, i), cfg),
                split=op,
                needs_u=frozenset({(op, i)}),
                needs_b=frozenset(),
            ))

    for op in binary_names(cfg):
        pairs: set[tuple[int, int]] = set()
        while len(pairs) < n_per_op:
            i, j = int(rng.integers(cfg.n_values)), int(rng.integers(cfg.n_values))
            pairs.add((i, j))
        for i, j in sorted(pairs):
            e = AtomApp(op, (i, j))
            out.append(TestItem(
                expr_src=render(e, cfg),
                answer_src=render_value(inst.tables.apply_binary(op, i, j), cfg),
                split=op,
                needs_u=frozenset(),
                needs_b=frozenset({(op, i, j)}),
            ))

    return out
```

In `src/glyph/data/__init__.py`: add `from .probe import probe_set` and `"probe_set"` to `__all__`.

- [ ] **Step 4: Run to verify pass, plus the two boundary gates**

Run: `~/miniforge3/envs/glyph/bin/python -m pytest tests/test_probe.py tests/test_data_boundary.py tests/test_frozen_instances.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/glyph/data/probe.py src/glyph/data/__init__.py tests/test_probe.py
git commit -m "feat(data): probe_set — per-op bare-atomic probes outside the fingerprint"
```

---

### Task 2: Workspace — probe ids and delivery in `final/test.jsonl`

**Files:**
- Modify: `src/glyph/v2/workspace.py` (`write_test_file`, new `probe_ids_for`, `_README` text)
- Test: `tests/v2/test_workspace.py`

**Interfaces:**
- Consumes: `probe_set` items (plain `TestItem`s).
- Produces: `probe_ids_for(inst, probes) -> Callable[[Any], str]` (ids `probe_00000`… in a shuffled order keyed off `inst.seed`); `write_test_file(paths, inst, test_id_of, probes=None, probe_id_of=None)` appends probe rows after the test rows.

- [ ] **Step 1: Write the failing test** (append to `tests/v2/test_workspaces.py`'s actual filename `tests/v2/test_workspace.py`, following its existing fixtures for building a workspace; if it has none, build inline as below)

```python
def test_write_test_file_appends_probe_rows(tmp_path):
    import json
    from glyph.data import PRESETS, generate
    from glyph.data.probe import probe_set
    from glyph.v2 import workspace as W

    inst = generate(1001, PRESETS["smoke"])
    paths, _, test_id_of = W.build_workspace(inst, tmp_path)
    probes = probe_set(inst, n_per_op=5)
    probe_id_of = W.probe_ids_for(inst, probes)
    out = W.write_test_file(paths, inst, test_id_of, probes=probes,
                            probe_id_of=probe_id_of)
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    test_rows = [r for r in rows if r["id"].startswith("test")]
    probe_rows = [r for r in rows if r["id"].startswith("probe")]
    assert len(test_rows) == len(inst.test)
    assert len(probe_rows) == len(probes)
    assert {r["id"] for r in probe_rows} == {probe_id_of(t) for t in probes}
    assert all(set(r) == {"id", "expr"} for r in rows)   # no answers, no split


def test_write_test_file_without_probes_is_unchanged(tmp_path):
    import json
    from glyph.data import PRESETS, generate
    from glyph.v2 import workspace as W
    inst = generate(1001, PRESETS["smoke"])
    paths, _, test_id_of = W.build_workspace(inst, tmp_path)
    out = W.write_test_file(paths, inst, test_id_of)
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len(rows) == len(inst.test)
```

- [ ] **Step 2: Run to verify failure**

Run: `~/miniforge3/envs/glyph/bin/python -m pytest tests/v2/test_workspace.py -q -k probe`
Expected: FAIL — `AttributeError: ... has no attribute 'probe_ids_for'`

- [ ] **Step 3: Implement.** In `workspace.py`:

```python
def probe_ids_for(inst, probes: list) -> Callable[[Any], str]:
    """Identity-keyed probe ids, in an order shuffled off the instance seed
    (same convention as the test ids)."""
    shuffled = list(probes)
    shuffle_for(inst.seed).shuffle(shuffled)
    id_of, _ = assign_ids("probe", shuffled)
    return id_of
```

Change `write_test_file` to:

```python
def write_test_file(paths: Paths, inst, test_id_of: Callable[[Any], str],
                    probes: list | None = None,
                    probe_id_of: Callable[[Any], str] | None = None) -> Path:
    """Write final/test.jsonl: one {"id","expr"} per test item (shuffled order,
    as before), followed by the probe rows when a probe set is attached.  No
    answer, no split label on either kind."""
    items = sorted(inst.test, key=test_id_of)
    lines = [json.dumps({"id": test_id_of(t), "expr": t.expr_src}) for t in items]
    if probes:
        p_items = sorted(probes, key=probe_id_of)
        lines += [json.dumps({"id": probe_id_of(t), "expr": t.expr_src})
                  for t in p_items]
    out = paths.final_dir / "test.jsonl"
    out.write_text("\n".join(lines) + ("\n" if lines else ""))
    return out
```

In `_README`, after the sentence describing the final phase / test.jsonl submission, add one line:

```
  - final/test.jsonl also contains bare-atomic probe rows (ids probe_*).
    They are scored separately from the main test, but a legal answer file
    must cover every row, probes included.
```

(Adapt indentation/bullet style to the surrounding `_README` text.)

- [ ] **Step 4: Run to verify pass**

Run: `~/miniforge3/envs/glyph/bin/python -m pytest tests/v2/test_workspace.py -q`
Expected: PASS (including all pre-existing workspace tests — the no-probes call path is unchanged)

- [ ] **Step 5: Commit**

```bash
git add src/glyph/v2/workspace.py tests/v2/test_workspace.py
git commit -m "feat(workspace): deliver probe rows in final/test.jsonl with probe_ ids"
```

---

### Task 3: `final_answer` legality covers probes

**Files:**
- Modify: `src/glyph/v2/tools.py` (`t_final_answer`)
- Test: `tests/v2/test_tools.py`

**Interfaces:**
- Consumes: `session.probes` (list) and `session.probe_id_of` (callable), both optional — set by the harness in Task 5.
- Produces: with probes attached, `t_final_answer` checks the answer file against `inst.test + probes` under a combined id function; a file missing probe answers is illegal and not committed. Without probes: behavior byte-identical to today.

- [ ] **Step 1: Write the failing tests** (append to `tests/v2/test_tools.py`; reuse its `make_session` / `write_test` helpers)

```python
def _attach_probes(s, inst, n=5):
    from glyph.data.probe import probe_set
    s.probes = probe_set(inst, n_per_op=n)
    s.probe_id_of = T.default_id_of("probe", s.probes)
    return s.probes


def test_final_answer_requires_probe_answers(inst, tmp_path):
    s = make_session(inst, tmp_path, phase="final")
    probes = _attach_probes(s, inst)
    f = tmp_path / "final.jsonl"
    write_test(f, inst, s.test_id_of, correct=True)   # test rows only
    out = T.t_final_answer(s, path=str(f))
    assert "error" in out
    assert out["violations"]["missing_ids"] >= len(probes)


def test_final_answer_with_probes_commits(inst, tmp_path):
    s = make_session(inst, tmp_path, phase="final")
    probes = _attach_probes(s, inst)
    f = tmp_path / "final.jsonl"
    rows = [{"id": s.test_id_of(t), "answer": t.answer_src} for t in inst.test]
    rows += [{"id": s.probe_id_of(t), "answer": t.answer_src} for t in probes]
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = T.t_final_answer(s, path=str(f))
    assert out.get("committed") is True
```

- [ ] **Step 2: Run to verify failure**

Run: `~/miniforge3/envs/glyph/bin/python -m pytest tests/v2/test_tools.py -q -k probe`
Expected: `test_final_answer_requires_probe_answers` FAILS (file with only test rows commits today); the other fails on `unknown_ids` (probe ids not in the target set).

- [ ] **Step 3: Implement.** In `tools.py`, replace the two `t_final_answer` lines that build `id_of`/`check_file` with:

```python
    id_of = _test_id_of(session)
    items = list(session.inst.test)
    probes = getattr(session, "probes", None)
    if probes:
        p_id_of, t_id_of = session.probe_id_of, id_of
        probe_ids = {id(t) for t in probes}
        items += list(probes)
        id_of = lambda t: p_id_of(t) if id(t) in probe_ids else t_id_of(t)
    v = check_file(path, items, session.inst.cfg, session.run_dir, id_of)
```

(The rest of the handler is unchanged.)

- [ ] **Step 4: Run to verify pass**

Run: `~/miniforge3/envs/glyph/bin/python -m pytest tests/v2/test_tools.py -q`
Expected: PASS, including the two pre-existing final_answer tests (no probes attached → unchanged path).

- [ ] **Step 5: Commit**

```bash
git add src/glyph/v2/tools.py tests/v2/test_tools.py
git commit -m "feat(tools): final_answer legality covers the probe rows"
```

---

### Task 4: Report — the `probe` block

**Files:**
- Modify: `src/glyph/v2/report.py`
- Test: `tests/v2/test_report.py`

**Interfaces:**
- Consumes: `session.probes` / `session.probe_id_of` (optional), `glyph.v2.answers.score_file`, `inst.is_tail`.
- Produces: report key `"probe"`: `None` without probes, else `{"n_per_op": {...}, "by_op": {op: {"overall", "n", "seen": {"n","acc"}, "unseen": {"n","acc"}}}}`. Never touches `overall`/`by_split`.

- [ ] **Step 1: Write the failing test** (append to `tests/v2/test_report.py`, reusing its existing session/report fixtures for `build_report`; construct as its other tests do)

```python
def test_report_probe_block_scores_by_op_and_seen_unseen(tmp_path):
    import json
    from glyph.data import PRESETS, generate
    from glyph.data.probe import probe_set
    from glyph.v2.ledger import Ledger
    from glyph.v2.report import build_report
    from glyph.v2.session import Session
    from glyph.v2 import tools as T

    inst = generate(1001, PRESETS["smoke"])
    s = Session(inst=inst, ledger=Ledger(), run_dir=tmp_path, arm="no_train")
    s.phase = "final"
    test_id_of = T.default_id_of("test", inst.test)
    s.test_id_of = test_id_of
    s.probes = probe_set(inst, n_per_op=6)
    s.probe_id_of = T.default_id_of("probe", s.probes)

    # mark the first probe's cell as purchased -> it is "seen"
    first = s.probes[0]
    inst.query(first.expr_src)

    # answer everything correctly except the second probe
    rows = [{"id": test_id_of(t), "answer": t.answer_src} for t in inst.test]
    for k, t in enumerate(s.probes):
        ans = t.answer_src if k != 1 else "v_9_9"
        rows.append({"id": s.probe_id_of(t), "answer": ans})
    f = tmp_path / "final.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    rep = build_report(s, str(f), test_id_of)
    probe = rep["probe"]
    assert set(probe["by_op"]) == {t.split for t in s.probes}
    op0 = first.split
    assert probe["by_op"][op0]["seen"]["n"] >= 1        # the purchased cell
    total_n = sum(v["n"] for v in probe["by_op"].values())
    assert total_n == len(s.probes)
    # the one wrong answer shows up in exactly one op's accuracy
    wrong_op = s.probes[1].split
    assert probe["by_op"][wrong_op]["overall"] < 1.0
    # and the main scores are untouched by probes
    assert rep["overall"] == 1.0


def test_report_probe_block_absent_without_probes(tmp_path):
    import json
    from glyph.data import PRESETS, generate
    from glyph.v2.ledger import Ledger
    from glyph.v2.report import build_report
    from glyph.v2.session import Session
    from glyph.v2 import tools as T

    inst = generate(1001, PRESETS["smoke"])
    s = Session(inst=inst, ledger=Ledger(), run_dir=tmp_path, arm="no_train")
    test_id_of = T.default_id_of("test", inst.test)
    rows = [{"id": test_id_of(t), "answer": t.answer_src} for t in inst.test]
    f = tmp_path / "final.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert build_report(s, str(f), test_id_of)["probe"] is None
```

(`inst.query(first.expr_src)` is what makes the cell land in `inst.query_log`; if smoke's `query_violation` refuses nothing here, this is exactly the agent's purchase path. `"v_9_9"` must be adjusted to a valid-but-wrong value literal for the smoke preset's value form — build it as `render_value((parse_value(t.answer_src, cfg) + 1) % cfg.n_values, cfg)` if the literal form differs.)

- [ ] **Step 2: Run to verify failure**

Run: `~/miniforge3/envs/glyph/bin/python -m pytest tests/v2/test_report.py -q -k probe`
Expected: FAIL — `KeyError: 'probe'`

- [ ] **Step 3: Implement.** In `report.py`, add after the `tail` computation:

```python
    probes = getattr(session, "probes", None)
    if probes:
        p_id_of = session.probe_id_of

        def _acc(sub):
            if not sub:
                return None
            if committed_path is None:
                return 0.0
            return score_file(committed_path, sub, cfg, p_id_of)["overall"]

        by_op: dict[str, dict] = {}
        for op in sorted({t.split for t in probes}):
            its = [t for t in probes if t.split == op]
            unseen = [t for t in its if inst.is_tail(t)]
            seen = [t for t in its if not inst.is_tail(t)]
            by_op[op] = {
                "overall": _acc(its),
                "n": len(its),
                "seen": {"n": len(seen), "acc": _acc(seen)},
                "unseen": {"n": len(unseen), "acc": _acc(unseen)},
            }
        probe_block = {"by_op": by_op}
    else:
        probe_block = None
```

and add `"probe": probe_block,` to the returned dict (after `"tail"`).

- [ ] **Step 4: Run to verify pass**

Run: `~/miniforge3/envs/glyph/bin/python -m pytest tests/v2/test_report.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/glyph/v2/report.py tests/v2/test_report.py
git commit -m "feat(report): probe block — per-op accuracy split seen/unseen"
```

---

### Task 5: Harness wiring + viewer

**Files:**
- Modify: `src/glyph/v2/harness.py` (attach probes to the session; pass them to `write_test_file`)
- Modify: `glyph-viewer/index.html` (`renderRun`)
- Test: `tests/v2/test_harness_logic.py` and/or `tests/v2/test_e2e_smoke.py` — run them and adapt any fixture that writes a final answer file to also answer the probe rows (the scripted agent that reads `final/test.jsonl` row-by-row needs no change).

**Interfaces:**
- Consumes: everything above.
- Produces: every run generates `probe_set(inst)` (default `n_per_op=100`), sets `session.probes` / `session.probe_id_of`, delivers them at the phase switch; `run.json`/`report.json` carry the block automatically.

- [ ] **Step 1: Wire the harness.** In `harness.py`, where the session and workspace are built (before `drive_practice`), add:

```python
    from glyph.data.probe import probe_set
    from glyph.v2.workspace import probe_ids_for
    probes = probe_set(inst)
    session.probes = probes
    session.probe_id_of = probe_ids_for(inst, probes)
```

and change the phase-switch call to:

```python
    workspace.write_test_file(paths, inst, test_id_of,
                              probes=session.probes,
                              probe_id_of=session.probe_id_of)
```

- [ ] **Step 2: Run the protocol-layer tests; fix fixtures forward**

Run: `~/miniforge3/envs/glyph/bin/python -m pytest tests/v2 -q`
Expected: any failure is a fixture that commits a final answer file built from `inst.test` alone — extend it with probe rows exactly as Task 3's `test_final_answer_with_probes_commits` does. Re-run to green.

- [ ] **Step 3: Viewer.** In `glyph-viewer/index.html`, add above `function renderRun(rj){`:

```javascript
function probeTable(probe){
  const table=el("table");
  const hd=el("tr");
  hd.append(el("th",null,"op"),el("th",null,"overall"),el("th",null,"seen"),el("th",null,"unseen"));
  table.append(hd);
  for(const op of Object.keys(probe.by_op||{}).sort()){
    const v=probe.by_op[op]||{}, s=v.seen||{}, u=v.unseen||{};
    const r=el("tr");
    r.append(el("td",null,op),
             el("td",null,num(v.overall,3)+" (n="+(v.n??0)+")"),
             el("td",null,num(s.acc,3)+" (n="+(s.n??0)+")"),
             el("td",null,num(u.acc,3)+" (n="+(u.n??0)+")"));
    table.append(r);
  }
  return table;
}
```

and inside `renderRun`, right before the `m.append(section("Full report (JSON)"...)` line:

```javascript
  if(rep.probe&&rep.probe.by_op) m.append(section("Per-op probes",probeTable(rep.probe),true));
```

Verify by serving the repo (`python -m http.server`) against an existing run JSON — the section simply stays absent for old runs.

- [ ] **Step 4: Commit**

```bash
git add src/glyph/v2/harness.py glyph-viewer/index.html tests/v2
git commit -m "feat(harness,viewer): attach probe set to every run; render per-op block"
```

---

### Task 6: Full suite, docs, PR

**Files:**
- Modify: `docs/data-validation.en.md`, `docs/data-validation.zh.md`, `docs/tools.md`, `docs/progress.md`

- [ ] **Step 1: Full fast suite**

Run: `~/miniforge3/envs/glyph/bin/python -m pytest -q -m "not slow"`
Expected: all pass; `test_frozen_instances.py` and `test_data_boundary.py` unmodified and green.

- [ ] **Step 2: Docs.** `data-validation.{en,zh}.md`: in the scoring/metrics section, add a short "Per-op probes" paragraph (English/Chinese): probe set = `n_per_op` bare-atomic items per op, generated per instance from a dedicated RNG stream (fingerprint-exempt), delivered with the final test as `probe_*` rows, mandatory in `final_answer`, scored in `report.probe.by_op` outside `overall`, split seen/unseen by the run's `LookupLog` (unseen = the per-op generalization read). `docs/tools.md`: note under `final_answer` that legality covers the probe rows. `docs/progress.md`: append a PR C entry with the command, pass count, and the design sentence about not refusing probe-cell queries.

- [ ] **Step 3: Commit docs, push, open PR**

```bash
git add docs
git commit -m "docs: per-op probe set (data-validation, tools, progress)"
git push -u origin feat-probes
gh pr create --title "feat: per-op probe set, mandatory at final, scored outside overall" --body-file <body>
```

PR body: §C summary, pass count, frozen/boundary gates, ending with the "Generated with Claude Code" line. **Stop: the user approves the merge.**

"""probe_set: deterministic per-op bare-atomic probes, outside the fingerprint."""
from glyph.data import PRESETS, generate, parse, render_value
from glyph.data.grammar import (App, AtomApp, binary_names, enabled_ops,
                                unary_names)
from glyph.data.probe import probe_set


def test_probe_set_shape_and_determinism():
    inst = generate(1001, PRESETS["smoke"])
    cfg = inst.cfg
    a, b = probe_set(inst, n_per_op=20), probe_set(inst, n_per_op=20)
    assert [(t.expr_src, t.answer_src, t.split) for t in a] == \
           [(t.expr_src, t.answer_src, t.split) for t in b]
    atomic_ops = unary_names(cfg) + binary_names(cfg)
    for op in atomic_ops:
        items = [t for t in a if t.split == op]
        want = min(20, cfg.n_values) if op.startswith("u") else 20
        assert len(items) == want
        assert len({t.expr_src for t in items}) == want   # distinct cells
    struct_ops = {op for op, shape in enabled_ops(cfg) if shape in ("L", "KL")}
    assert {t.split for t in a} == set(atomic_ops) | struct_ops


def test_probe_items_are_correct_single_cells():
    inst = generate(1001, PRESETS["smoke"])
    cfg = inst.cfg
    for t in probe_set(inst, n_per_op=10):
        if not (t.needs_u or t.needs_b):
            continue                      # structural probes: covered elsewhere
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


def _cells_for(shape):
    """Mirror of the behavior-cell grid: 8 for L, 24 for KL."""
    n = sum(2 if ln == 2 else 3 for ln in (2, 3, 4))
    return n * (3 if shape == "KL" else 1)


def test_structural_probe_items():
    from glyph.data import depth
    inst = generate(1001, PRESETS["smoke"])
    cfg = inst.cfg
    probes = probe_set(inst, n_per_op=5, struct_reps=2)
    struct = {op: shape for op, shape in enabled_ops(cfg) if shape in ("L", "KL")}
    assert struct, "smoke preset must enable at least one pure-structure op"
    fixed = {t.expr_src for t in inst.test} | {t.expr_src for t in inst.val}
    for op, shape in struct.items():
        items = [t for t in probes if t.split == op]
        assert len(items) == 2 * _cells_for(shape)
        assert len({t.expr_src for t in items}) == len(items)   # distinct
        for t_ in items:
            e = parse(t_.expr_src, cfg)
            assert isinstance(e, App) and e.op == op
            assert depth(e) == 1                     # single application
            assert not t_.needs_u and not t_.needs_b  # table-free
            assert t_.expr_src not in fixed           # sealed sets untouched
            out = inst.P.eval(e)
            want = (render_value(out, cfg) if isinstance(out, int)
                    else "[" + ", ".join(render_value(i, cfg) for i in out) + "]")
            assert t_.answer_src == want


def test_structural_probes_deterministic_and_atomic_prefix_stable():
    inst = generate(1001, PRESETS["smoke"])
    a = probe_set(inst, n_per_op=5, struct_reps=2)
    b = probe_set(inst, n_per_op=5, struct_reps=2)
    assert [(t.expr_src, t.answer_src) for t in a] == \
           [(t.expr_src, t.answer_src) for t in b]
    atomic_only = probe_set(inst, n_per_op=5, struct_reps=0)
    assert [(t.expr_src, t.answer_src) for t in a[:len(atomic_only)]] == \
           [(t.expr_src, t.answer_src) for t in atomic_only]

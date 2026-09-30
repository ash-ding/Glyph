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

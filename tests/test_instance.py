"""Split construction, determinism, and the derived tail split."""
import pytest

from glyph.data.config import PRESETS
from glyph.data.grammar import depth, op_pairs, parse
from glyph.data.instance import generate

FAST = PRESETS["pi_mid"].with_(n_iid=300, n_comp=120, n_depth=80)


def test_same_seed_same_instance():
    a, b = generate(42, FAST), generate(42, FAST)
    assert a.demos == b.demos
    assert [t.expr_src for t in a.test] == [t.expr_src for t in b.test]


def test_different_seeds_differ():
    a, b = generate(1, FAST), generate(2, FAST)
    assert a.demos != b.demos


def test_splits_respect_their_constraints():
    inst = generate(7, FAST)
    cfg = inst.cfg
    for t in inst.test_set("iid"):
        e = parse(t.expr_src, cfg)
        assert depth(e) <= cfg.demo_max_depth
        assert not (op_pairs(e) & inst.held_pairs)
    for t in inst.test_set("comp"):
        assert op_pairs(parse(t.expr_src, cfg)) & inst.held_pairs
    for t in inst.test_set("depth"):
        assert depth(parse(t.expr_src, cfg)) > cfg.demo_max_depth


def test_demos_avoid_held_out_pairs():
    inst = generate(7, FAST)
    for src, _ in inst.demos:
        assert not (op_pairs(parse(src, inst.cfg)) & inst.held_pairs)


def test_test_items_are_unique_and_unseen():
    inst = generate(7, FAST)
    srcs = [t.expr_src for t in inst.test]
    assert len(set(srcs)) == len(srcs)
    assert not (set(srcs) & {a for a, _ in inst.demos})


def test_tail_shrinks_as_the_agent_queries():
    inst = generate(7, FAST)
    # items that touch no table entry at all are never in `tail`
    touching = [t for t in inst.test if t.needs_u or t.needs_b]
    before = len(inst.derive_tail())
    assert before == len(touching)           # nothing bought yet
    for t in inst.test[:150]:
        inst.query(t.expr_src)
    after = len(inst.derive_tail())
    assert after < before


def test_syntax_errors_are_still_charged():
    inst = generate(7, FAST)
    n = inst.query_count
    with pytest.raises(Exception):
        inst.query("s0(nonsense")
    assert inst.query_count == n + 1

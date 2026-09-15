from glyph.data import PRESETS, generate
from glyph.data.grammar import render, parse, depth, op_pairs
from glyph.data.config import QUERY_OOD_POLICIES

FAST = PRESETS["pi_mid"].with_(n_iid=300, n_comp=120, n_depth=80, n_val=120)

def test_policies_constant():
    assert QUERY_OOD_POLICIES == ("strict", "open")

def test_refuses_validation_item():
    inst = generate(7, FAST)
    v = inst.val[0].expr_src
    assert inst.query_violation(v) == "is_validation_item"
    assert inst.query_violation(v, "open") == "is_validation_item"   # val/test refused under both

def test_refuses_test_item():
    inst = generate(7, FAST)
    t = inst.test_set("iid")[0].expr_src
    assert inst.query_violation(t) == "is_test_item"

def test_canonicalises_before_matching():
    inst = generate(7, FAST)
    src = inst.val[0].expr_src
    messy = render(parse(src, inst.cfg), inst.cfg).replace(", ", ",")   # drop the spaces
    assert messy != src
    assert inst.query_violation(messy) == "is_validation_item"

def test_malformed_is_not_a_violation():
    inst = generate(7, FAST)
    assert inst.query_violation("s0(nope", "strict") is None
    assert inst.query_violation("this is not glyph", "strict") is None

def test_strict_refuses_a_deep_nontest_expr_open_allows():
    inst = generate(7, FAST); cfg = inst.cfg
    # Build a depth-3 expr from an L-shaped op (list->list) so it needs no held pair
    # to be deep. Find an enabled op whose shape is "L".
    from glyph.data.grammar import enabled_ops, SHAPE_RESULT
    l_ops = [op for op, shape in enabled_ops(cfg) if shape == "L"]
    assert l_ops, "pi_mid should enable at least one L-shaped op"
    o = l_ops[0]
    src = f"{o}({o}({o}([v_a_a_a, v_b_b_b])))"   # depth 3 > demo_max_depth (2)
    e = parse(src, cfg)
    assert depth(e) > cfg.demo_max_depth
    r_strict = inst.query_violation(src, "strict")
    # It is deep; it may or may not also contain a held pair (o,o). Either OOD
    # reason is correct. It must NOT be a val/test item (too deep for those).
    assert r_strict in ("deeper_than_demos", "contains_held_out_pair")
    assert inst.query_violation(src, "open") is None   # open lifts both OOD checks

def test_strict_refuses_a_held_pair_expr():
    inst = generate(7, FAST); cfg = inst.cfg
    # Take a held-out (outer, inner) pair and build a depth-2 expr using it, if the
    # pair's shapes allow a direct nesting. Held pairs are (outer, inner) where
    # inner returns LIST. Build outer(inner(...)) minimally.
    # If constructing a clean one is not possible for this seed, fall back to a
    # comp test item, which is guaranteed to contain a held pair (but is a test
    # item, so the reason will be is_test_item -- still a correct refusal).
    comp = inst.test_set("comp")[0]
    r = inst.query_violation(comp.expr_src, "strict")
    assert r in ("contains_held_out_pair", "is_test_item")
    # And under open, a comp *test item* is still refused (test check runs always):
    assert inst.query_violation(comp.expr_src, "open") == "is_test_item"


def test_contains_held_out_pair_branch_is_reached():
    import numpy as np
    from glyph.data.instance import _sample_constrained
    from glyph.data.grammar import render, op_pairs, parse
    inst = generate(7, FAST); cfg = inst.cfg
    rng = np.random.default_rng(999)
    hit = None
    for _ in range(200):
        e = _sample_constrained(rng, cfg, cfg.demo_max_depth,
                                forbid=set(), require=inst.held_pairs, min_depth=1)
        if e is None:
            continue
        src = render(e, cfg)
        r = inst.query_violation(src, "strict")
        if r == "contains_held_out_pair":
            hit = src
            # sanity: it really does contain a held pair and is not a test/val item
            assert op_pairs(parse(src, cfg)) & inst.held_pairs
            break
        # else it collided with a real test/val item (is_test_item / is_validation_item) -- keep trying
    assert hit is not None, "could not construct a non-test expr that trips the held-pair branch"

def test_open_policy_allows_a_held_pair_nontest_expr():
    import numpy as np
    from glyph.data.instance import _sample_constrained
    from glyph.data.grammar import render
    inst = generate(7, FAST); cfg = inst.cfg
    rng = np.random.default_rng(999)
    for _ in range(200):
        e = _sample_constrained(rng, cfg, cfg.demo_max_depth,
                                forbid=set(), require=inst.held_pairs, min_depth=1)
        if e is None:
            continue
        src = render(e, cfg)
        if inst.query_violation(src, "strict") == "contains_held_out_pair":
            assert inst.query_violation(src, "open") is None   # open lifts the held-pair check
            return
    assert False, "no held-pair non-test expr found to check open policy"

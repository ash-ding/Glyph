from glyph.data import PRESETS, generate
from glyph.data.interp import LookupLog

FAST = PRESETS["pi_mid"].with_(n_iid=200, n_comp=80, n_depth=40, n_val=80)

def test_empty_log_covers_only_lookup_free_items():
    inst = generate(7, FAST)
    frac = inst.lookup_solvable(inst.val, LookupLog())
    free = sum(1 for t in inst.val if not t.needs_u and not t.needs_b) / len(inst.val)
    assert abs(frac - free) < 1e-9

def test_full_log_covers_everything():
    inst = generate(7, FAST)
    log = LookupLog()
    for t in inst.val:
        log.unary |= set(t.needs_u)
        log.binary |= set(t.needs_b)
    assert inst.lookup_solvable(inst.val, log) == 1.0

def test_empty_items_is_zero():
    inst = generate(7, FAST)
    assert inst.lookup_solvable([], LookupLog()) == 0.0

def test_partial_log_is_monotone_on_test_split():
    # Buying more entries never lowers coverage.
    inst = generate(7, FAST)
    items = inst.test
    empty = inst.lookup_solvable(items, LookupLog())
    full = LookupLog()
    for t in items:
        full.unary |= set(t.needs_u); full.binary |= set(t.needs_b)
    assert inst.lookup_solvable(items, full) >= empty
    assert inst.lookup_solvable(items, full) == 1.0

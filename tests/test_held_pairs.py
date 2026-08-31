"""How held-out pairs are drawn, and the ceiling a score is read against.

A uniform draw over the Cartesian product of operator names spent part of its
quota on pairs the grammar cannot build, which left `comp` -- the split that
*requires* a held-out pair -- empty on 13 of 20 pi_low seeds, and left the
held share drifting between 17% and 50% on the rest.
"""
import collections
import statistics as st

import numpy as np
import pytest

from glyph.config import PRESETS
from glyph.grammar import SHAPE_RESULT, enabled_ops
from glyph.instance import _draw_held_pairs, generate
from glyph.seal import headroom

PRESET_NAMES = ["pi_low", "pi_mid", "pi_high"]
SEEDS = [1001, 1002, 1003, 1004, 1005]


def _realizable(cfg):
    ops = enabled_ops(cfg)
    shape = dict(ops)
    return {(a, b) for a, _ in ops for b, _ in ops
            if SHAPE_RESULT[shape[b]] == "LIST"}


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_every_held_pair_is_one_the_grammar_can_build(preset):
    cfg = PRESETS[preset]
    real = _realizable(cfg)
    for seed in SEEDS:
        held = _draw_held_pairs(cfg, np.random.default_rng(seed))
        assert held <= real, (preset, seed, held - real)


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_the_held_share_is_the_same_for_every_seed(preset):
    cfg = PRESETS[preset]
    sizes = {len(_draw_held_pairs(cfg, np.random.default_rng(s)))
             for s in range(1001, 1021)}
    assert len(sizes) == 1, (preset, sizes)
    assert sizes.pop() == max(1, len(_realizable(cfg)) // 3)


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_the_type_composition_is_the_same_for_every_seed(preset):
    """Which pair inside a class is drawn varies; how many of each class does
    not.  That is what keeps `comp`'s operator mix comparable across seeds."""
    cfg = PRESETS[preset]
    shape = dict(enabled_ops(cfg))
    profiles = set()
    for seed in range(1001, 1021):
        held = _draw_held_pairs(cfg, np.random.default_rng(seed))
        c = collections.Counter((shape[a], shape[b]) for a, b in held)
        profiles.add(tuple(sorted(c.items())))
    assert len(profiles) == 1, profiles


def test_a_class_is_never_held_out_entirely():
    """Holding every pair of some shape class would remove that composition
    from `iid` and `depth` rather than hold it out for `comp`."""
    for preset in PRESET_NAMES:
        cfg = PRESETS[preset]
        shape = dict(enabled_ops(cfg))
        by_kind = collections.defaultdict(set)
        for a, b in _realizable(cfg):
            by_kind[(shape[a], shape[b])].add((a, b))
        held = _draw_held_pairs(cfg, np.random.default_rng(1001))
        for kind, group in by_kind.items():
            if len(group) > 1:
                assert not group <= held, (preset, kind)


# -- the ceiling ------------------------------------------------------
@pytest.mark.parametrize("seed", [1001, 1002, 1003, 1004, 1005])
def test_the_skeleton_ceiling_is_at_least_the_items_needing_no_table(seed):
    """A true skeleton with an identity table answers every item that needs no
    lookup -- and sometimes more.

    `needs_u` / `needs_b` record entries *touched*, not entries the answer
    *depends on*: `eval_logged` logs every table call, including calls whose
    result a later transform discards. On seed 1002, `s0` is
    `map_skip(j=1) -> dedup -> rotate(1)` and the outer `s3` keeps only the
    element the map skipped, so two lookups happen and neither matters.

    Equality held on 3 of these 5 seeds and the ceiling exceeded the free
    fraction by 7% and 9% on the other two. An earlier note called the two
    "equal, and not approximately"; that was seed 1001, not a property."""
    inst = generate(seed, PRESETS["pi_mid"].scaled(600))
    ceil = inst.ceilings()
    for split in ("iid", "comp", "depth"):
        group = inst.test_set(split)
        free = sum(1 for t in group if not t.needs_u and not t.needs_b)
        assert ceil["skeleton"][split] >= free / len(group) - 1e-9


def test_ceilings_follow_the_items_they_are_asked_about():
    inst = generate(1001, PRESETS["pi_mid"].scaled(600))
    sub = inst.test_set("depth")
    got = inst.ceilings(sub)
    assert set(got["skeleton"]) == {"depth", "overall"}
    assert got["skeleton"]["depth"] == pytest.approx(got["skeleton"]["overall"])


def test_headroom_places_a_score_against_its_ceiling():
    assert headroom(0.222, 0.222) == pytest.approx(0.0)   # knows only structure
    assert headroom(1.0, 0.222) == pytest.approx(1.0)     # perfect
    assert headroom(0.035, 0.222) < 0                     # below it: no skeleton either
    assert headroom(0.5, 1.0) is None                     # nothing left to measure


def test_tail_is_asked_of_the_item_not_its_position():
    """`derive_tail` returns indices into the full test set; scoring on a
    subset must not read those as positions in the subset."""
    inst = generate(1001, PRESETS["pi_mid"].scaled(600))
    for t in inst.test[:40]:
        inst.query(t.expr_src)
    by_index = {inst.test[i].expr_src for i in inst.derive_tail()}
    by_item = {t.expr_src for t in inst.test if inst.is_tail(t)}
    assert by_index == by_item

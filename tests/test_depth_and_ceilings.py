"""Depth as a distribution, and a reference line for every number reported."""
import collections

import pytest

from glyph.budget import Ledger
from glyph.data.config import PRESETS
from glyph.data.grammar import depth, parse
from glyph.data.instance import generate
from glyph.seal import SealedArtifact, evaluate

FAST = PRESETS["pi_mid"].scaled(2000)


# ---- #6: depth is sampled, not restated from the budget ----
def test_iid_spans_more_than_one_depth():
    """It used to be a single point: every iid item at exactly demo_max_depth,
    which is also why the agent's single-level probes could not overlap the
    test set at all."""
    inst = generate(1001, FAST)
    h = collections.Counter(depth(parse(t.expr_src, inst.cfg))
                            for t in inst.test_set("iid"))
    assert len(h) > 1, h
    assert max(h) <= inst.cfg.demo_max_depth


def test_the_depth_split_is_no_longer_a_single_point():
    inst = generate(1001, FAST)
    h = collections.Counter(depth(parse(t.expr_src, inst.cfg))
                            for t in inst.test_set("depth"))
    assert len(h) > 1, h
    assert min(h) > inst.cfg.demo_max_depth


def test_stop_prob_zero_restores_the_old_behaviour():
    inst = generate(1001, FAST.with_(depth_stop_prob=0.0))
    h = collections.Counter(depth(parse(t.expr_src, inst.cfg))
                            for t in inst.test_set("iid"))
    assert len(h) == 1 and inst.cfg.demo_max_depth in h


def test_the_root_never_stops():
    """A stopped root is a bare literal, not an expression."""
    inst = generate(1001, FAST.with_(depth_stop_prob=0.9))
    assert all(depth(parse(t.expr_src, inst.cfg)) >= 1 for t in inst.test)


# ---- #4: every reported number has a line to read it against ----
def _scored():
    inst = generate(1001, FAST)
    for t in inst.test[:60]:
        inst.query(t.expr_src)
    r = evaluate(inst, SealedArtifact(arm="t", entry="model"),
                 Ledger(total_h100s=1e9), answer_fn=lambda e: [""] * len(e))
    return inst, r


def test_every_reported_score_has_a_ceiling():
    _, r = _scored()
    for key in list(r.by_split) + ["overall"]:
        assert key in r.ceiling["skeleton"], key
        assert key in r.headroom, key
    if r.tail is not None:
        assert "tail" in r.ceiling["skeleton"] and "tail" in r.headroom


@pytest.mark.parametrize("seed", [1001, 1002, 1003])
def test_tails_ceiling_is_near_zero_and_that_is_the_point(seed):
    """`tail` is where almost nothing is free, which is what makes it the
    cleanest read on table knowledge.

    Not *exactly* zero, and the gap is informative rather than noise. `tail`
    means "touched an entry this run never bought", while the skeleton-only
    oracle answers whatever the identity table happens to get right -- which
    includes items whose lookups were performed and then discarded by a later
    transform. So `tail` is over-inclusive, and its ceiling is the size of that
    over-inclusion."""
    inst = generate(seed, FAST)
    for t in inst.test[:60]:
        inst.query(t.expr_src)
    r = evaluate(inst, SealedArtifact(arm="t", entry="model"),
                 Ledger(total_h100s=1e9), answer_fn=lambda e: [""] * len(e))
    assert 0.0 <= r.ceiling["skeleton"]["tail"] < 0.15
    assert r.headroom["tail"] == pytest.approx(
        (r.tail - r.ceiling["skeleton"]["tail"])
        / (1 - r.ceiling["skeleton"]["tail"]))


def test_ceilings_are_computed_on_the_scored_subset():
    inst = generate(1001, FAST)
    sub = inst.test_set("depth")
    r = evaluate(inst, SealedArtifact(arm="t", entry="model"),
                 Ledger(total_h100s=1e9),
                 answer_fn=lambda e: [""] * len(e), items=sub)
    assert set(r.ceiling["skeleton"]) >= {"depth", "overall"}
    assert r.ceiling["skeleton"]["depth"] == pytest.approx(
        r.ceiling["skeleton"]["overall"])

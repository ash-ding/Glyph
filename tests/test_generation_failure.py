"""Generation must fail loudly rather than return a short test set.

`pi_low` seed 1002 silently produced 8800 items with no `depth` split at all,
and was then compared against 10000-item instances as though the two were the
same measurement.  These tests pin the failure to an exception.
"""
import collections

import pytest

from glyph.data import instance as inst_mod
from glyph.data.config import PRESETS
from glyph.data.grammar import SHAPE_RESULT, enabled_ops
from glyph.data.instance import GenerationFailed, generate

FAST = PRESETS["pi_mid"].with_(n_iid=300, n_comp=120, n_depth=80)


def test_healthy_config_still_generates_exactly_what_it_asked_for():
    inst = generate(7, FAST)
    by_split = collections.Counter(t.split for t in inst.test)
    assert len(inst.test) == FAST.n_test
    assert by_split == {"iid": 300, "comp": 120, "depth": 80}


# The ways a split can be left unfillable, constructed directly.
#
# These used to be built from `pi_low` with two structural operators, which was
# genuinely broken at the time. It no longer is: drawing held-out pairs only
# from realizable ones, a fixed share per shape class, removes both original
# failure modes from every config we ship -- even the two-operator one, though
# that is a side effect of how the class quotas break ties and not a guarantee
# anyone should lean on. So the situation has to be made rather than found, or
# the test quietly stops exercising anything.
#
# The mechanism is still worth pinning: it guards against a future change to
# how pairs are drawn, to the grammar's shapes, or to a preset's depths
# silently going back to returning a short test set.
TINY = PRESETS["pi_mid"].with_(n_iid=40, n_comp=20, n_depth=20, n_demos=30)


def _realizable(cfg):
    shape = dict(enabled_ops(cfg))
    return {(a, b) for a, _ in enabled_ops(cfg) for b, _ in enabled_ops(cfg)
            if SHAPE_RESULT[shape[b]] == "LIST"}


def test_holding_out_every_pair_empties_depth_but_not_the_demos(monkeypatch):
    """With every pair held out, only the splits that need nesting die.

    Depth-1 expressions -- one operator over a literal list -- contain no
    adjacent operator pair at all, so they are structurally exempt from the
    held-pair constraint. Since `depth_stop_prob` made those reachable, the
    demos and `iid` can still be filled from them; `depth`, which requires
    nesting past `demo_max_depth`, cannot."""
    monkeypatch.setattr(inst_mod, "_STALL_LIMIT", 40)
    monkeypatch.setattr(inst_mod, "_draw_held_pairs",
                        lambda cfg, rng: _realizable(cfg))
    with pytest.raises(GenerationFailed) as e:
        generate(1001, TINY)
    assert e.value.split == "depth"
    assert e.value.made < e.value.want


def test_holding_out_nothing_empties_comp(monkeypatch):
    """`comp` requires a held-out pair to appear, so an empty held set leaves
    it with nothing to ask for."""
    monkeypatch.setattr(inst_mod, "_STALL_LIMIT", 40)
    monkeypatch.setattr(inst_mod, "_draw_held_pairs", lambda cfg, rng: set())
    with pytest.raises(GenerationFailed) as e:
        generate(1001, TINY)
    assert e.value.split == "comp"
    assert e.value.diag["missing_required_pair"] == e.value.diag["sampled"]


def test_a_depth_budget_that_cannot_exceed_the_demos_fails(monkeypatch):
    """The `depth` split asks for expressions deeper than the demos, drawn at a
    budget of `max_expr_depth`. A config where that budget does not exceed
    `demo_max_depth` asks for something the sampler cannot produce -- a plain
    misconfiguration, and one that used to yield a test set missing a third of
    its splits rather than an error."""
    monkeypatch.setattr(inst_mod, "_STALL_LIMIT", 40)
    cfg = TINY.with_(max_expr_depth=TINY.demo_max_depth)
    with pytest.raises(GenerationFailed) as e:
        generate(1001, cfg)
    assert e.value.split == "depth"
    assert e.value.diag["reached_min_depth"] == 0


def test_the_failure_names_its_own_cause(monkeypatch):
    """The diagnosis has to separate "too shallow" from "hit a held-out pair",
    because with early stopping both are now happening at once."""
    monkeypatch.setattr(inst_mod, "_STALL_LIMIT", 40)
    monkeypatch.setattr(inst_mod, "_draw_held_pairs",
                        lambda cfg, rng: _realizable(cfg))
    with pytest.raises(GenerationFailed) as e:
        generate(1001, TINY)
    diag = e.value.diag
    # Every sample that contains any pair at all hits a forbidden one; the
    # remainder are the depth-1 samples, which contain no pair to forbid.
    pairless = diag["depth_hist"].get(1, 0)
    assert diag["hit_forbidden_pair"] == diag["sampled"] - pairless
    # and depth was reachable, so shallowness is not what emptied the split
    assert diag["reached_min_depth"] > 0


@pytest.mark.parametrize("preset", ["pi_low", "pi_mid", "pi_high"])
@pytest.mark.parametrize("seed", [1001, 1002, 1003, 1004])
def test_every_shipped_preset_generates(preset, seed):
    """The presets we actually run must not be able to fail this way.

    Scanned at full size over 20 seeds each: 60/60. This guards a handful of
    them cheaply, so a future config change that reintroduces an unfillable
    split is caught here rather than in a grid run.
    """
    cfg = PRESETS[preset].scaled(300)
    inst = generate(seed, cfg)
    by_split = collections.Counter(t.split for t in inst.test)
    assert set(by_split) == {"iid", "comp", "depth"}
    assert len(inst.test) == cfg.n_test


def test_demo_sampling_terminates_rather_than_spinning(monkeypatch):
    # `_make_demos` had no stall counter at all: a config whose demo
    # constraints cannot be met looped forever with no output.
    monkeypatch.setattr(inst_mod, "_STALL_LIMIT", 20)
    monkeypatch.setattr(inst_mod, "_sample_constrained",
                        lambda *a, **k: None)
    with pytest.raises(GenerationFailed) as e:
        generate(7, FAST)
    assert e.value.split == "demos"


def test_depth_one_expressions_carry_no_operator_pair():
    """The exemption the two tests above turn on, stated directly.

    `op_pairs` reports adjacent (outer, inner) structural operators, so an
    expression with a single operator has none -- and every constraint built on
    held-out pairs, in either direction, passes over it silently. Before
    `depth_stop_prob` these did not exist: depth equalled the split's budget,
    which is at least 2 everywhere."""
    from glyph.data.grammar import depth, op_pairs, parse
    inst = generate(1001, PRESETS["pi_mid"].scaled(2000))
    shallow = [t for t in inst.test
               if depth(parse(t.expr_src, inst.cfg)) == 1]
    assert shallow, "early stopping should produce depth-1 items"
    for t in shallow:
        assert not op_pairs(parse(t.expr_src, inst.cfg))

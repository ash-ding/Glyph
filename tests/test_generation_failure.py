"""Generation must fail loudly rather than return a short test set.

`pi_low` seed 1002 silently produced 8800 items with no `depth` split at all,
and was then compared against 10000-item instances as though the two were the
same measurement.  These tests pin the failure to an exception.
"""
import collections

import pytest

from glyph import instance as inst_mod
from glyph.config import PRESETS
from glyph.instance import GenerationFailed, generate

FAST = PRESETS["pi_mid"].with_(n_iid=300, n_comp=120, n_depth=80)


def test_healthy_config_still_generates_exactly_what_it_asked_for():
    inst = generate(7, FAST)
    by_split = collections.Counter(t.split for t in inst.test)
    assert len(inst.test) == FAST.n_test
    assert by_split == {"iid": 300, "comp": 120, "depth": 80}


# Two structural operators is the shape that cannot work, and it is built here
# rather than read from a preset: `pi_low` used to have it and no longer does,
# and a regression test that depends on a preset staying broken tests the
# preset instead of the mechanism.
#
# With two operators the grammar offers four Cartesian (outer, inner) pairs but
# can only build two of them -- s1 returns VAL and nothing takes a VAL argument
# -- and `n_hold` removes one of the four at random. Half the time it removes a
# pair that cannot occur, so `comp`, which requires it, is empty; a quarter of
# the time it removes (s0, s0), the only route to depth 3, so `depth` is empty.
UNFILLABLE = PRESETS["pi_low"].with_(n_structural=2, n_iid=40, n_comp=20,
                                     n_depth=20)


def test_an_unfillable_split_raises_instead_of_returning_a_short_set(monkeypatch):
    monkeypatch.setattr(inst_mod, "_STALL_LIMIT", 40)
    with pytest.raises(GenerationFailed) as e:
        generate(1002, UNFILLABLE)          # holds out (s0, s0) -> no depth 3
    assert e.value.split == "depth"
    assert e.value.made < e.value.want


def test_the_failure_names_its_own_cause(monkeypatch):
    monkeypatch.setattr(inst_mod, "_STALL_LIMIT", 40)
    with pytest.raises(GenerationFailed) as e:
        generate(1002, UNFILLABLE)
    diag = e.value.diag
    # the depth was reachable; the held-out-pair constraint is what rejected
    assert diag["reached_min_depth"] == diag["sampled"]
    assert diag["hit_forbidden_pair"] == diag["sampled"]


def test_a_phantom_held_out_pair_empties_comp(monkeypatch):
    # The other failure mode: the held-out pair is one the grammar cannot
    # build, so `comp`, which requires it to appear, can never be filled.
    monkeypatch.setattr(inst_mod, "_STALL_LIMIT", 40)
    with pytest.raises(GenerationFailed) as e:
        generate(1004, UNFILLABLE)
    assert e.value.split == "comp"
    assert e.value.diag["missing_required_pair"] == e.value.diag["sampled"]


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

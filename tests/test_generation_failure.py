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


def test_an_unfillable_split_raises_instead_of_returning_a_short_set(monkeypatch):
    # pi_low's `depth` split forbids held-out pairs, but with two structural
    # operators every depth-3 expression carries one.  No amount of resampling
    # fixes that, so it must raise.
    monkeypatch.setattr(inst_mod, "_STALL_LIMIT", 40)
    cfg = PRESETS["pi_low"].with_(n_iid=40, n_comp=20, n_depth=20)
    with pytest.raises(GenerationFailed) as e:
        generate(1002, cfg)
    assert e.value.split == "depth"
    assert e.value.made < e.value.want


def test_the_failure_names_its_own_cause(monkeypatch):
    monkeypatch.setattr(inst_mod, "_STALL_LIMIT", 40)
    cfg = PRESETS["pi_low"].with_(n_iid=40, n_comp=20, n_depth=20)
    with pytest.raises(GenerationFailed) as e:
        generate(1002, cfg)
    diag = e.value.diag
    # the depth was reachable; the held-out-pair constraint is what rejected
    assert diag["reached_min_depth"] == diag["sampled"]
    assert diag["hit_forbidden_pair"] == diag["sampled"]


def test_demo_sampling_terminates_rather_than_spinning(monkeypatch):
    # `_make_demos` had no stall counter at all: a config whose demo
    # constraints cannot be met looped forever with no output.
    monkeypatch.setattr(inst_mod, "_STALL_LIMIT", 20)
    monkeypatch.setattr(inst_mod, "_sample_constrained",
                        lambda *a, **k: None)
    with pytest.raises(GenerationFailed) as e:
        generate(7, FAST)
    assert e.value.split == "demos"

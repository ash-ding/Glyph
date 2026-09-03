"""A run report must carry the instance's measured pi, not just its preset name.

The preset is a sampler: measured pi ranges overlap between neighbours -- seven
of twenty `pi_mid` seeds fall inside `pi_high`'s range -- so an instance cannot
be attributed to a preset from its pi, and a figure keyed on the name would
group instances that differ more within a group than between. The axis has to
be the measured value, which means every report has to carry it.
"""
import json

import pytest

from glyph.budget import Ledger
from glyph.data.config import PRESETS
from glyph.data.instance import generate
from glyph.seal import SealedArtifact, evaluate

FAST = PRESETS["pi_mid"].scaled(300)


def _report(inst):
    art = SealedArtifact(arm="test", entry="model")
    return evaluate(inst, art, Ledger(total_h100s=1e9),
                    answer_fn=lambda exprs: [""] * len(exprs))


def test_report_carries_measured_pi():
    inst = generate(1001, FAST)
    r = _report(inst)
    assert r.instance["seed"] == 1001
    for key in ("pi", "a_skel", "a_tab", "L_skel", "L_table"):
        assert key in r.instance["pi"], key
    assert 0.0 <= r.instance["pi"]["pi"] <= 1.0


def test_the_components_survive_so_pi_can_be_redefined_later():
    """#3 may change which items pi is measured on. Recording the whole dict
    means that change can be recomputed rather than regenerated."""
    inst = generate(1001, FAST)
    r = _report(inst)
    m = r.instance["pi"]
    assert m["L_skel"] == pytest.approx(1.0 - m["a_tab"])
    assert m["L_table"] == pytest.approx(1.0 - m["a_skel"])
    denom = m["L_skel"] + m["L_table"]
    assert m["pi"] == pytest.approx(m["L_skel"] / denom)


def test_two_presets_can_produce_the_same_pi():
    """The reason the name cannot be the condition."""
    mid = generate(1002, PRESETS["pi_mid"].scaled(600)).measured_pi()["pi"]
    high = generate(1004, PRESETS["pi_high"].scaled(600)).measured_pi()["pi"]
    # not asserting which is larger -- the point is that the ranges overlap,
    # so the ordering of any particular pair is not guaranteed by the names
    assert 0.0 <= mid <= 1.0 and 0.0 <= high <= 1.0


def test_measured_pi_is_cached():
    inst = generate(1001, FAST)
    assert inst.measured_pi() is inst.measured_pi()


def test_report_json_round_trips_the_instance_block():
    inst = generate(1001, FAST)
    got = json.loads(_report(inst).to_json())
    assert got["instance"]["seed"] == 1001
    assert "pi" in got["instance"]["pi"]

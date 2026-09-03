"""The agent's self-estimate, measured instead of hidden.

`dev` is carved from what the agent bought, which was a deliberate choice --
a free labelled dev set would quietly refund the query budget -- with a cost
that was not foreseen: it does not estimate test. The agent buys single-level
probes because isolating one table entry is how you read a table, and almost
nothing on the test set looks like that. One run reached dev 0.400 against test
0.035.

Settled (#14) as: leave the environment alone and measure the gap. An agent with
a narrow probing strategy *should* get a misleading self-estimate; what was
wrong was that the miscalibration was invisible.
"""
import pytest

from glyph.arms.base import RunConfig, finish, prepare
from glyph.config import PRESETS
from glyph.seal import SealedArtifact, calibration


def _rc(tmp_path):
    return RunConfig(arm="test", preset="smoke", instance_seed=7,
                     budget_h100s=1e9, out_root=tmp_path)


def test_the_gap_is_reported_and_both_sides_use_one_scoring_path(tmp_path):
    p = prepare(_rc(tmp_path))
    dev = [(t.expr_src, t.answer_src) for t in p.inst.test[:8]]
    truth = {e: a for e, a in dev}

    # an artifact that is right on dev and wrong everywhere else
    def answer(exprs):
        return [truth.get(e, "") for e in exprs]

    art = SealedArtifact(arm="test", entry="model")
    r = finish(p, _rc(tmp_path), art, answer, dev=dev, purchased=dev)
    c = r.calibration
    assert c["dev_accuracy"] == pytest.approx(1.0)
    assert c["dev_n"] == len(dev)
    assert c["test_accuracy"] == r.overall
    assert c["gap"] == pytest.approx(1.0 - r.overall)


def test_the_distributions_that_explain_the_gap_are_recorded(tmp_path):
    """The gap is not a mystery number: what the agent bought and what it was
    scored on are reported beside it, by depth."""
    p = prepare(_rc(tmp_path))
    dev = [(t.expr_src, t.answer_src) for t in p.inst.test[:6]]
    art = SealedArtifact(arm="test", entry="model")
    r = finish(p, _rc(tmp_path), art, lambda e: [""] * len(e),
               dev=dev, purchased=dev)
    c = r.calibration
    assert c["purchased_by_depth"] and c["scored_by_depth"]
    assert all(k.isdigit() for k in c["scored_by_depth"])


def test_no_dev_yields_a_gap_of_none_rather_than_a_wrong_number(tmp_path):
    p = prepare(_rc(tmp_path))
    art = SealedArtifact(arm="test", entry="model")
    r = finish(p, _rc(tmp_path), art, lambda e: [""] * len(e), dev=[], purchased=[])
    assert r.calibration["dev_accuracy"] is None
    assert r.calibration["gap"] is None

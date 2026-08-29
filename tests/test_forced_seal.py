"""A run must always end with something scoreable.

The arms most likely to end up with nothing are the ones whose preparation
is most expensive -- so refusing to seal an empty artifact deletes data
preferentially from the arm the comparison is about.
"""
import pytest

from glyph.agent.schema import Container
from glyph.agent.tools import ToolBox
from glyph.budget import Ledger
from glyph.config import PRESETS
from glyph.instance import generate
from glyph.trace import TraceWriter


@pytest.fixture
def box(tmp_path):
    return ToolBox(inst=generate(7, PRESETS["smoke"]), ledger=Ledger(),
                   trace=TraceWriter(tmp_path / "t.jsonl"),
                   allowed={Container.WEIGHTS}, work_dir=tmp_path, arm="test")


def test_an_ordinary_seal_still_needs_something_to_seal(box):
    assert "error" in box.dispatch("seal", {"entry": "model",
                                            "checkpoint_id": "", "summary": "s"})
    assert box.sealed is None


def test_a_forced_seal_cannot_be_refused(box):
    """Training that all failed still has to produce a data point."""
    out = box.dispatch("seal", {"entry": "model", "checkpoint_id": "",
                                "summary": "harness", "forced": True})
    assert out.get("sealed")
    assert box.sealed is not None
    assert box.sealed.adapter_path is None and box.sealed.context is None


def test_a_forced_program_seal_without_a_program_falls_back_to_the_student(box):
    box.dispatch("seal", {"entry": "program", "checkpoint_id": "",
                          "summary": "harness", "forced": True})
    assert box.sealed is not None
    assert box.sealed.entry == "model", "nothing to run, so the student answers"

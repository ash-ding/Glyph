"""Unit tests for the harness phase state machine (Task 17).

NO claude_agent_sdk import, NO API, NO torch. A FakeClient drives the real
phase state machine; each scripted turn invokes the REAL glyph.v2.tools
handlers so the Session mutates exactly as it would in production.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from glyph.data import PRESETS, generate
from glyph.v2.session import Session
from glyph.v2.ledger import Ledger
from glyph.v2.workspace import build_workspace, write_test_file
from glyph.v2 import tools as T
from glyph.v2.report import build_report
from glyph.v2.harness import drive_practice, drive_final, iter_tool_calls

CFG = PRESETS["smoke"]


# ---------------------------------------------------------------------
# fake client + fake events
# ---------------------------------------------------------------------
class FakeAssistant:
    """An assistant event carrying the tool names called this turn."""
    def __init__(self, tool_names):
        self.tool_names = list(tool_names)


class FakeTerminal:
    """A terminal (ResultMessage-like) event: no tool names, no content."""
    pass


class FakeClient:
    """Scripted client satisfying ClientProto.

    turns: list of turns; each turn is a list of (tool_name, kwargs).  On each
    receive_response() the next turn is played: every scripted tool call
    invokes the REAL t_<name>(session, **kwargs), then an assistant event
    carrying the names is yielded, then a terminal event.  An empty turn
    yields an idle assistant event (no tools) + terminal.
    """
    def __init__(self, session, turns):
        self.session = session
        self._turns = list(turns)
        self._i = 0
        self.interrupted = 0
        self.queries = []

    async def query(self, text: str) -> None:
        self.queries.append(text)

    async def interrupt(self) -> None:
        self.interrupted += 1

    def receive_response(self):
        turn = self._turns[self._i] if self._i < len(self._turns) else []
        self._i += 1
        return self._play(turn)

    async def _play(self, turn):
        names = []
        for name, kwargs in turn:
            handler = getattr(T, "t_" + name)
            handler(self.session, **kwargs)
            names.append(name)
        yield FakeAssistant(names)
        yield FakeTerminal()


# ---------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------
@pytest.fixture
def inst():
    return generate(1001, CFG)


def make_session(inst, tmp_path, arm="no_train", **kw):
    paths, val_id_of, test_id_of = build_workspace(inst, tmp_path)
    s = Session(inst=inst, ledger=Ledger(), run_dir=tmp_path, arm=arm, **kw)
    s.val_id_of = val_id_of
    s.test_id_of = test_id_of
    s.preset = "smoke"
    return s, paths, val_id_of, test_id_of


def write_answers(path, items, id_of, correct=True):
    rows = [{"id": id_of(t), "answer": t.answer_src if correct else "v_a_a"}
            for t in items]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


# ---------------------------------------------------------------------
# iter_tool_calls helper
# ---------------------------------------------------------------------
def test_iter_tool_calls_fake_and_terminal():
    assert iter_tool_calls(FakeAssistant(["submit", "query"])) == ["submit", "query"]
    assert iter_tool_calls(FakeAssistant([])) == []
    assert iter_tool_calls(FakeTerminal()) == []


# ---------------------------------------------------------------------
# practice: idle
# ---------------------------------------------------------------------
def test_three_idle_turns_end_practice(inst, tmp_path):
    s, paths, *_ = make_session(inst, tmp_path)
    client = FakeClient(s, [[], [], []])
    reason = asyncio.run(drive_practice(s, client, opener="go"))
    assert reason == "idle"
    assert s.turns == 3            # every assistant turn counts, idle included
    assert s.should_switch is True  # ended as finish_practice
    # a continuation was sent after each of the first two idle turns
    assert client.queries.count("Continue, or call finish_practice if you are done.") >= 2


# ---------------------------------------------------------------------
# practice: submit cap -> should_switch, S4 interrupt+drain, then switch
# ---------------------------------------------------------------------
def test_submit_cap_switches_and_final_phase_gates(inst, tmp_path):
    s, paths, val_id_of, test_id_of = make_session(inst, tmp_path, submit_cap=2)
    f = tmp_path / "val_ans.jsonl"
    write_answers(f, inst.val, val_id_of, correct=True)
    turns = [[("submit", {"path": str(f)})], [("submit", {"path": str(f)})]]
    client = FakeClient(s, turns)
    reason = asyncio.run(drive_practice(s, client, opener="go"))
    assert reason == "should_switch"
    assert s.submissions == 2
    assert s.should_switch is True
    assert s.turns == 2
    # S4: submit detected mid-stream -> interrupt() was called
    assert client.interrupted >= 1

    # switch to final + materialise test.jsonl
    s.switch_to_final()
    assert s.phase == "final"
    test_path = write_test_file(paths, inst, test_id_of)
    assert test_path.exists()

    # a phase-1 tool called after the switch is refused by the handler
    out = T.t_submit(s, path=str(f))
    assert "error" in out
    assert s.submissions == 2  # unchanged


# ---------------------------------------------------------------------
# final: auto-commit from a passing check_answers(set="test")
# ---------------------------------------------------------------------
def test_final_auto_commit_from_checked_path(inst, tmp_path):
    s, paths, val_id_of, test_id_of = make_session(inst, tmp_path)
    s.switch_to_final()
    write_test_file(paths, inst, test_id_of)
    tf_ans = tmp_path / "test_ans.jsonl"
    write_answers(tf_ans, inst.test, test_id_of, correct=True)
    # turn 0: check_answers on a passing test file; then idle out
    turns = [[("check_answers", {"path": str(tf_ans), "set": "test"})], [], [], []]
    client = FakeClient(s, turns)
    reason = asyncio.run(drive_final(s, client, opener="final"))
    assert reason == "idle"
    assert s.last_checked_test_path == str(tf_ans)
    assert s.final_commit == "auto_checked_path"
    assert s.final_commit_path == str(tf_ans)


def test_final_no_commit_when_never_checked(inst, tmp_path):
    s, paths, val_id_of, test_id_of = make_session(inst, tmp_path)
    s.switch_to_final()
    write_test_file(paths, inst, test_id_of)
    client = FakeClient(s, [[], [], []])
    reason = asyncio.run(drive_final(s, client, opener="final"))
    assert reason == "idle"
    assert s.final_commit == "none"
    assert getattr(s, "final_commit_path", None) is None


# ---------------------------------------------------------------------
# final: a legal final_answer ends with final_commit == "agent"
# ---------------------------------------------------------------------
def test_legal_final_answer_ends_run(inst, tmp_path):
    s, paths, val_id_of, test_id_of = make_session(inst, tmp_path)
    s.switch_to_final()
    write_test_file(paths, inst, test_id_of)
    tf_ans = tmp_path / "final.jsonl"
    write_answers(tf_ans, inst.test, test_id_of, correct=True)
    turns = [[("final_answer", {"path": str(tf_ans)})]]
    client = FakeClient(s, turns)
    reason = asyncio.run(drive_final(s, client, opener="final"))
    assert reason == "agent"
    assert s.final_commit == "agent"
    assert s.final_commit_path == str(tf_ans)

    # end-to-end: build_report over the committed path
    report = build_report(s, s.final_commit_path, test_id_of)
    assert "overall" in report
    assert report["overall"] == 1.0

"""Tests for glyph.v2.tools — the handler core."""
import json
import hashlib
import pathlib
import numpy as np
import pytest

from glyph.data import PRESETS, generate
from glyph.data.instance import _sample_constrained
from glyph.data.grammar import render
from glyph.v2.session import Session
from glyph.v2.ledger import Ledger
from glyph.v2 import tools as T

CFG = PRESETS["smoke"]


@pytest.fixture
def inst():
    return generate(1001, CFG)


def make_session(inst, tmp_path, arm="train", phase="practice", **kw):
    s = Session(inst=inst, ledger=Ledger(), run_dir=tmp_path, arm=arm, **kw)
    s.phase = phase
    s.val_id_of = T.default_id_of("val", inst.val)
    s.test_id_of = T.default_id_of("test", inst.test)
    return s


def write_val(path, inst, id_of, correct=True):
    rows = []
    for t in inst.val:
        rows.append({"id": id_of(t), "answer": t.answer_src if correct else "v_a_a"})
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def write_test(path, inst, id_of, correct=True):
    rows = []
    for t in inst.test:
        rows.append({"id": id_of(t), "answer": t.answer_src if correct else "v_a_a"})
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def novel_expr(inst, cfg, seed=7):
    """A well-formed expr that is not a val/test item and not query-refused."""
    rng = np.random.default_rng(seed)
    for _ in range(50):
        e = _sample_constrained(rng, cfg, cfg.demo_max_depth, forbid=inst.held_pairs,
                                 require=None, min_depth=1)
        if e is None:
            continue
        src = render(e, cfg)
        if inst.query_violation(src, "strict") is None:
            return src
    raise RuntimeError("could not build a novel well-formed expr")


# ---------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------

def test_gate_train_tool_on_no_train_session(inst, tmp_path):
    s = make_session(inst, tmp_path, arm="no_train")
    out = T.t_build_dataset(s, path="x")
    assert "error" in out
    assert s.turns == 0


def test_gate_query_on_final_phase(inst, tmp_path):
    s = make_session(inst, tmp_path, phase="final")
    out = T.t_query(s, exprs=["v_a_a"], why="probe")
    assert "error" in out
    assert s.q_used == 0


# ---------------------------------------------------------------------
# t_query charging
# ---------------------------------------------------------------------

def test_query_charging_mix(inst, tmp_path):
    s = make_session(inst, tmp_path)
    val_expr = inst.val[0].expr_src
    good_expr = novel_expr(inst, CFG, seed=11)
    exprs = [val_expr, good_expr, "s0(oops"]
    out = T.t_query(s, exprs=exprs, why="testing mix")
    assert s.q_used == 2  # only the two non-refused charged
    results = out["results"]
    assert results[0]["expr"] == val_expr
    assert results[0]["refused"] == "is_validation_item"
    assert "error" not in results[0] and "out" not in results[0]
    assert results[1]["expr"] == good_expr
    assert "out" in results[1]
    assert results[2]["expr"] == "s0(oops"
    assert results[2]["error"] == "malformed"
    qfile = tmp_path / "task" / "queries.jsonl"
    lines = qfile.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["expr"] == good_expr and rec["out"] == results[1]["out"]


def test_query_exhausted(inst, tmp_path):
    s = make_session(inst, tmp_path, q_cap=1)
    e1 = novel_expr(inst, CFG, seed=21)
    e2 = novel_expr(inst, CFG, seed=22)
    assert e1 != e2
    out = T.t_query(s, exprs=[e1, e2], why="w")
    assert s.q_used == 1
    assert "out" in out["results"][0]
    assert out["results"][1]["error"] == "q_exhausted"


def test_query_success_echoes_remaining(inst, tmp_path):
    s = make_session(inst, tmp_path)
    e1 = novel_expr(inst, CFG, seed=31)
    out = T.t_query(s, exprs=[e1], why="w")
    assert out["q_remaining"] == s.q_remaining()
    assert out["submissions_remaining"] == s.submissions_remaining()
    assert out["turns_remaining"] == s.turns_remaining()


# ---------------------------------------------------------------------
# t_submit
# ---------------------------------------------------------------------

def test_submit_legal_perfect(inst, tmp_path):
    s = make_session(inst, tmp_path)
    f = tmp_path / "ans.jsonl"
    write_val(f, inst, s.val_id_of, correct=True)
    out = T.t_submit(s, path=str(f))
    assert s.submissions == 1
    assert out["overall"] == 1.0
    assert "by_depth" in out
    assert "by_split" not in out
    assert "error" not in out


def test_submit_hits_cap_sets_should_switch(inst, tmp_path):
    s = make_session(inst, tmp_path, submit_cap=1)
    f = tmp_path / "ans.jsonl"
    write_val(f, inst, s.val_id_of, correct=True)
    out = T.t_submit(s, path=str(f))
    assert s.submissions == 1
    assert s.should_switch is True


def test_submit_illegal_missing_ids_not_counted(inst, tmp_path):
    s = make_session(inst, tmp_path)
    f = tmp_path / "ans.jsonl"
    # only write the first item -> rest are missing
    rows = [{"id": s.val_id_of(inst.val[0]), "answer": inst.val[0].answer_src}]
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = T.t_submit(s, path=str(f))
    assert "error" in out
    assert out["violations"]["missing_ids"] > 0
    assert s.submissions == 0


# ---------------------------------------------------------------------
# t_finish_practice
# ---------------------------------------------------------------------

def test_finish_practice_sets_should_switch(inst, tmp_path):
    s = make_session(inst, tmp_path)
    out = T.t_finish_practice(s, reason="done")
    assert s.should_switch is True
    assert out["ok"] is True


# ---------------------------------------------------------------------
# t_check_answers
# ---------------------------------------------------------------------

def test_check_answers_test_set_in_practice_is_error(inst, tmp_path):
    s = make_session(inst, tmp_path, phase="practice")
    out = T.t_check_answers(s, path="whatever", set="test")
    assert "error" in out


def test_check_answers_test_set_final_passing(inst, tmp_path):
    s = make_session(inst, tmp_path, phase="final")
    f = tmp_path / "final.jsonl"
    write_test(f, inst, s.test_id_of, correct=True)
    out = T.t_check_answers(s, path=str(f), set="test")
    assert out["ok"] is True
    assert "overall" not in out and "score" not in out
    assert s.last_checked_test_path == str(f)


def test_check_answers_validation_practice(inst, tmp_path):
    s = make_session(inst, tmp_path, phase="practice")
    f = tmp_path / "val.jsonl"
    write_val(f, inst, s.val_id_of, correct=True)
    out = T.t_check_answers(s, path=str(f), set="validation")
    assert out["ok"] is True


# ---------------------------------------------------------------------
# t_final_answer
# ---------------------------------------------------------------------

def test_final_answer_legal(inst, tmp_path):
    s = make_session(inst, tmp_path, phase="final")
    f = tmp_path / "final.jsonl"
    write_test(f, inst, s.test_id_of, correct=True)
    out = T.t_final_answer(s, path=str(f))
    assert out["committed"] is True
    assert "digest" in out and len(out["digest"]) == 16
    expect = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
    assert out["digest"] == expect
    assert s.final_commit_path == str(f)
    assert s.final_commit == "agent"
    assert "overall" not in out and "score" not in out


def test_final_answer_illegal_not_committed(inst, tmp_path):
    s = make_session(inst, tmp_path, phase="final")
    f = tmp_path / "final.jsonl"
    rows = [{"id": s.test_id_of(inst.test[0]), "answer": inst.test[0].answer_src}]
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = T.t_final_answer(s, path=str(f))
    assert "error" in out
    assert not hasattr(s, "final_commit_path") or s.final_commit_path is None


# ---------------------------------------------------------------------
# train-arm tools with fake student
# ---------------------------------------------------------------------

class FakeStudent:
    def build_dataset(self, path, inst):
        return {"dataset_id": "ds1", "size": 10, "provenance": "fake"}

    def train(self, dataset_id, epochs, lr):
        return {"checkpoint_id": "ck1", "final_loss": 0.1, "gpu_seconds": 5.0,
                "gpu_seconds_remaining": 95.0, "stopped_at_cap": False}

    def infer(self, checkpoint, input_path, output_path, prefix_path):
        return {"rows": 3, "truncated": False, "gpu_seconds": 1.0}


def test_build_dataset_with_fake_student(inst, tmp_path):
    s = make_session(inst, tmp_path)
    s.student = FakeStudent()
    out = T.t_build_dataset(s, path="foo.jsonl")
    assert out["dataset_id"] == "ds1" and out["size"] == 10
    assert "q_remaining" in out


def test_train_with_fake_student(inst, tmp_path):
    s = make_session(inst, tmp_path)
    s.student = FakeStudent()
    out = T.t_train(s, dataset_id="ds1", epochs=2, lr=0.01)
    assert out["checkpoint_id"] == "ck1"
    assert out["gpu_seconds_remaining"] == 95.0


def test_student_infer_with_fake_student(inst, tmp_path):
    s = make_session(inst, tmp_path)
    s.student = FakeStudent()
    out = T.t_student_infer(s, checkpoint="ck1", input_path="i", output_path="o", prefix_path=None)
    assert out["rows"] == 3


def test_train_tools_no_student_error(inst, tmp_path):
    s = make_session(inst, tmp_path)
    assert getattr(s, "student", None) is None
    out = T.t_build_dataset(s, path="foo.jsonl")
    assert "error" in out
    out2 = T.t_train(s, dataset_id="ds1", epochs=1, lr=0.1)
    assert "error" in out2
    out3 = T.t_student_infer(s, checkpoint="c", input_path="i", output_path="o")
    assert "error" in out3


# ---------------------------------------------------------------------
# default_id_of
# ---------------------------------------------------------------------

def test_default_id_of(inst):
    id_of = T.default_id_of("val", inst.val)
    assert id_of(inst.val[0]) == "val_00000"
    assert id_of(inst.val[1]) == "val_00001"

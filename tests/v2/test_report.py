import json, pathlib, pytest
from glyph.data import PRESETS, generate
from glyph.v2.session import Session
from glyph.v2.ledger import Ledger
from glyph.v2.report import build_report

CFG = PRESETS["smoke"]

def _idof(items):
    m = {id(t): f"test_{i:05d}" for i, t in enumerate(items)}
    return lambda t: m[id(t)]

def test_perfect_commit_scores_1_and_has_all_blocks(tmp_path):
    inst = generate(1001, CFG)
    sess = Session(inst=inst, ledger=Ledger(), run_dir=tmp_path, arm="no_train")
    idof = _idof(inst.test)
    f = tmp_path/"final.jsonl"
    f.write_text("\n".join(json.dumps({"id": idof(t), "answer": t.answer_src}) for t in inst.test) + "\n")
    r = build_report(sess, f, idof)
    assert r["overall"] == 1.0
    assert set(r["by_split"]) <= {"iid","comp","depth"}
    for k in ("overall","iid","comp","depth","tail"):
        assert k in r["headroom"]
    instb = r["instance"]
    assert instb["seed"] == 1001
    assert isinstance(instb["uses_binary_tables"], bool)
    for comp in ("pi","a_skel","a_tab","L_skel","L_table"):
        assert comp in instb["pi"]
    assert "usd_by_kind" in r["ledger"]
    assert "val_lookup_solvable" in r["covariates"] and "test_covered" in r["covariates"]

def test_no_commit_scores_zero(tmp_path):
    inst = generate(1001, CFG)
    sess = Session(inst=inst, ledger=Ledger(), run_dir=tmp_path, arm="train")
    r = build_report(sess, None, _idof(inst.test))
    assert r["overall"] == 0.0
    assert r["covariates"]["final_commit"] == "none"

def test_validation_history_summarised(tmp_path):
    inst = generate(1001, CFG)
    sess = Session(inst=inst, ledger=Ledger(), run_dir=tmp_path, arm="no_train")
    sess.note(kind="validation_submit", submission=1, overall=0.2, by_depth={"1":0.3})
    sess.note(kind="validation_submit", submission=2, overall=0.5, by_depth={"1":0.6})
    idof = _idof(inst.test)
    f = tmp_path/"f.jsonl"; f.write_text("\n".join(json.dumps({"id": idof(t),"answer": t.answer_src}) for t in inst.test)+"\n")
    r = build_report(sess, f, idof)
    assert r["validation"]["best"] == 0.5 and r["validation"]["last"] == 0.5
    assert len(r["validation"]["history"]) == 2

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

def test_report_json_round_trips_instance_block(tmp_path):
    """The report is written to disk as JSON; its instance block (seed +
    measured pi components) must survive the round-trip unchanged. Restores the
    coverage v1's ScoreReport.to_json test gave before v1 was removed --
    exercises real build_report output, not a hand-built dict."""
    inst = generate(1001, CFG)
    sess = Session(inst=inst, ledger=Ledger(), run_dir=tmp_path, arm="no_train")
    idof = _idof(inst.test)
    f = tmp_path / "final.jsonl"
    f.write_text("\n".join(json.dumps({"id": idof(t), "answer": t.answer_src}) for t in inst.test) + "\n")
    r = build_report(sess, f, idof)
    got = json.loads(json.dumps(r))  # raises if the report is not JSON-serializable
    assert got["instance"] == r["instance"]
    assert got["instance"]["seed"] == 1001
    for comp in ("pi", "a_skel", "a_tab", "L_skel", "L_table"):
        assert comp in got["instance"]["pi"]


def test_report_probe_block_scores_by_op_and_seen_unseen(tmp_path):
    import json
    from glyph.data import PRESETS, generate, parse_value, render_value
    from glyph.data.probe import probe_set
    from glyph.v2.ledger import Ledger
    from glyph.v2.report import build_report
    from glyph.v2.session import Session
    from glyph.v2 import tools as T

    inst = generate(1001, PRESETS["smoke"])
    cfg = inst.cfg
    s = Session(inst=inst, ledger=Ledger(), run_dir=tmp_path, arm="no_train")
    s.phase = "final"
    test_id_of = T.default_id_of("test", inst.test)
    s.test_id_of = test_id_of
    s.probes = probe_set(inst, n_per_op=6)
    s.probe_id_of = T.default_id_of("probe", s.probes)

    # mark the first probe's cell as purchased -> it is "seen"
    first = s.probes[0]
    inst.query(first.expr_src)

    # answer everything correctly except the second probe
    def wrong(t):
        return render_value((parse_value(t.answer_src, cfg) + 1) % cfg.n_values, cfg)

    rows = [{"id": test_id_of(t), "answer": t.answer_src} for t in inst.test]
    for k, t in enumerate(s.probes):
        rows.append({"id": s.probe_id_of(t),
                     "answer": t.answer_src if k != 1 else wrong(t)})
    f = tmp_path / "final.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    rep = build_report(s, str(f), test_id_of)
    probe = rep["probe"]
    assert set(probe["by_op"]) == {t.split for t in s.probes}
    op0 = first.split
    assert probe["by_op"][op0]["seen"]["n"] >= 1        # the purchased cell
    total_n = sum(v["n"] for v in probe["by_op"].values())
    assert total_n == len(s.probes)
    # the one wrong answer shows up in exactly one op's accuracy
    wrong_op = s.probes[1].split
    assert probe["by_op"][wrong_op]["overall"] < 1.0
    # and the main scores are untouched by probes
    assert rep["overall"] == 1.0


def test_report_probe_block_absent_without_probes(tmp_path):
    import json
    from glyph.data import PRESETS, generate
    from glyph.v2.ledger import Ledger
    from glyph.v2.report import build_report
    from glyph.v2.session import Session
    from glyph.v2 import tools as T

    inst = generate(1001, PRESETS["smoke"])
    s = Session(inst=inst, ledger=Ledger(), run_dir=tmp_path, arm="no_train")
    test_id_of = T.default_id_of("test", inst.test)
    rows = [{"id": test_id_of(t), "answer": t.answer_src} for t in inst.test]
    f = tmp_path / "final.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert build_report(s, str(f), test_id_of)["probe"] is None

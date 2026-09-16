from glyph.v2.export import normalize_transcript, build_run_json


def _asst(*blocks):
    return {"type": "assistant", "message": {"content": list(blocks)}}

def _think():
    return {"type": "thinking", "thinking": "", "signature": "sig"}

def _text(s):
    return {"type": "text", "text": s}

def _tu(name, id, inp):
    return {"type": "tool_use", "name": name, "id": id, "input": inp}

def _results(*pairs):
    return {"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": i, "content": c, "is_error": e}
        for (i, c, e) in pairs]}}

def _opener(s):
    return {"type": "user", "message": {"content": s}}


def test_pairs_tool_use_with_result_and_captures_text():
    rows = [
        _opener("You are ready to begin the practice phase."),
        _asst(_think()),
        _asst(_text("Let me query.")),
        _asst(_tu("query", "a1", {"exprs": ["s0(u0)"]})),
        _results(("a1", "ok-result", False)),
    ]
    turns = normalize_transcript(rows)
    assert len(turns) == 1
    t = turns[0]
    assert t["text"] == ["Let me query."]
    assert t["thinking_redacted"] == 1
    assert len(t["actions"]) == 1
    a = t["actions"][0]
    assert a["name"] == "query" and a["input"] == {"exprs": ["s0(u0)"]}
    assert a["result"] == "ok-result" and a["is_error"] is False
    assert t["phase"] == "practice"


def test_is_error_flag_flows_through():
    rows = [_asst(_tu("Read", "r1", {"file_path": "/x"})),
            _results(("r1", "File does not exist", True))]
    turns = normalize_transcript(rows)
    assert turns[0]["actions"][0]["is_error"] is True


def test_phase_flips_at_final_opener():
    fo = "The oracle is gone. Here is the held-out test."
    rows = [
        _opener("You are ready to begin the practice phase."),
        _asst(_tu("query", "a1", {})), _results(("a1", "r", False)),
        _opener(fo),
        _asst(_tu("final_answer", "f1", {"path": "/out"})), _results(("f1", "committed", False)),
    ]
    turns = normalize_transcript(rows, final_opener=fo)
    assert [t["phase"] for t in turns] == ["practice", "final"]
    assert turns[1]["actions"][0]["name"] == "final_answer"


def test_multiple_tool_uses_in_one_turn():
    rows = [
        _asst(_tu("Read", "r1", {})),
        _asst(_tu("Read", "r2", {})),
        _results(("r1", "a", False), ("r2", "b", False)),
    ]
    turns = normalize_transcript(rows)
    assert len(turns) == 1 and len(turns[0]["actions"]) == 2
    assert turns[0]["actions"][0]["result"] == "a"
    assert turns[0]["actions"][1]["result"] == "b"


def test_build_run_json_shape(tmp_path):
    (tmp_path / "agent_home" / ".claude" / "projects" / "-work").mkdir(parents=True)
    import json
    tr = tmp_path / "agent_home" / ".claude" / "projects" / "-work" / "s.jsonl"
    tr.write_text("\n".join(json.dumps(r) for r in [
        _asst(_tu("query", "a1", {"exprs": ["x"]})), _results(("a1", "r", False))]) + "\n")
    rj = build_run_json(
        run_id="pi_mid_train_1001_x", created="2026-09-16T00:00:00Z",
        config={"arm": "train", "preset": "pi_mid", "seed": 1001},
        task_readme="TASK", prompts={"system": "SYS", "final_opener": "FO"},
        report={"overall": 0.42, "instance": {"pi": {"pi": 0.44}},
                "ledger": {"spent_usd": 25.7}, "covariates": {"run_status": "completed"}},
        run_dir=tmp_path)
    assert rj["id"] == "pi_mid_train_1001_x"
    assert rj["config"]["arm"] == "train"
    assert rj["task"]["readme"] == "TASK" and rj["prompts"]["system"] == "SYS"
    assert rj["summary"]["overall"] == 0.42 and rj["summary"]["pi"] == 0.44
    assert rj["summary"]["arm"] == "train" and rj["summary"]["n_turns"] == 1
    assert len(rj["transcript"]) == 1

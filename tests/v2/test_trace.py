import json
from glyph.v2.trace import RunTrace

def test_emits_jsonl_lines(tmp_path):
    with RunTrace(tmp_path) as tr:
        tr.emit("run_start", arm="train", seed=1001)
        tr.phase_event("final", reason="submit_cap", turns=12)
        tr.emit("tool", tool="query", n=3)
    lines = [json.loads(l) for l in (tmp_path/"trace.jsonl").read_text().splitlines() if l.strip()]
    assert len(lines) == 3
    assert lines[0]["kind"] == "run_start" and lines[0]["arm"] == "train"
    assert lines[1]["kind"] == "phase" and lines[1]["phase"] == "final" and lines[1]["reason"] == "submit_cap"
    assert lines[2]["kind"] == "tool" and lines[2]["tool"] == "query"
    # keys are sorted within each line (TraceWriter guarantee)
    raw = (tmp_path/"trace.jsonl").read_text().splitlines()[0]
    assert raw.index('"arm"') < raw.index('"kind"') < raw.index('"seed"')

def test_run_dir_created(tmp_path):
    d = tmp_path/"nested"/"run"
    tr = RunTrace(d); tr.emit("x", v=1); tr.close()
    assert (d/"trace.jsonl").exists()

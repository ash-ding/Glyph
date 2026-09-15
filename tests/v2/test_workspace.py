import json, pathlib
from glyph.data import PRESETS, generate
from glyph.v2.workspace import build_workspace, write_test_file

CFG = PRESETS["smoke"]

def test_workspace_layout_and_validation_ids(tmp_path):
    inst = generate(1001, CFG)
    paths, val_id_of, test_id_of = build_workspace(inst, tmp_path)
    assert paths.task.is_dir() and paths.final_dir.is_dir()
    assert paths.queries.exists() and paths.queries.read_text() == ""  # empty log
    # validation.jsonl: id+expr, no answer, ids in order val_00000..
    lines = [json.loads(l) for l in paths.validation.read_text().splitlines() if l.strip()]
    assert len(lines) == len(inst.val)
    assert set(lines[0]) == {"id","expr"}   # NO 'answer'
    assert lines[0]["id"] == "val_00000"
    assert [x["id"] for x in lines] == [f"val_{i:05d}" for i in range(len(inst.val))]
    # val_id_of maps the actual items
    assert val_id_of(inst.val[0]) == "val_00000"
    # demos.jsonl has answers
    d = [json.loads(l) for l in paths.demos.read_text().splitlines() if l.strip()]
    assert len(d) == len(inst.demos) and set(d[0]) == {"expr","answer"}
    # syntax present
    assert paths.syntax.read_text().strip() != ""

def test_test_file_is_shuffled_and_labelless(tmp_path):
    inst = generate(1001, CFG)
    paths, _, test_id_of = build_workspace(inst, tmp_path)
    assert not any(paths.final_dir.iterdir())          # final/ empty until now
    tf = write_test_file(paths, inst, test_id_of)
    rows = [json.loads(l) for l in tf.read_text().splitlines() if l.strip()]
    assert len(rows) == len(inst.test)
    assert set(rows[0]) == {"id","expr"}               # no 'split', no 'answer'
    ids = [r["id"] for r in rows]
    assert ids == sorted(ids)                           # ids are test_00000.. in file order
    # file order is a SHUFFLE of inst.test order (not identical) for a non-trivial test set
    exprs_in_file = [r["expr"] for r in rows]
    exprs_natural = [t.expr_src for t in inst.test]
    assert sorted(exprs_in_file) == sorted(exprs_natural)
    assert exprs_in_file != exprs_natural               # shuffled
    # test_id_of maps each item to the id its expr got in the file
    by_expr = {r["expr"]: r["id"] for r in rows}
    assert test_id_of(inst.test[0]) == by_expr[inst.test[0].expr_src]

def test_deterministic_per_seed(tmp_path):
    a = generate(1001, CFG); b = generate(1001, CFG)
    pa, _, ta = build_workspace(a, tmp_path/"a"); write_test_file(pa, a, ta)
    pb, _, tb = build_workspace(b, tmp_path/"b"); write_test_file(pb, b, tb)
    assert (pa.final_dir/"test.jsonl").read_text() == (pb.final_dir/"test.jsonl").read_text()

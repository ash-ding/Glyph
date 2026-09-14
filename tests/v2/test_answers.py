import json, pathlib, pytest
from glyph.data import PRESETS, generate
from glyph.v2.answers import check_file, score_file, parse_answer, Violations

CFG = PRESETS["smoke"]

@pytest.fixture
def inst():
    return generate(1001, CFG)

def _idof(items):
    m = {id(t): f"val_{i:05d}" for i, t in enumerate(items)}
    return lambda t: m[id(t)]

def _write(p, rows):
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

def test_correct_file_is_legal_and_scores_1(tmp_path, inst):
    items = inst.val; idof = _idof(items)
    work = tmp_path; f = work / "ans.jsonl"
    _write(f, [{"id": idof(t), "answer": t.answer_src} for t in items])
    v = check_file(f, items, inst.cfg, work, idof)
    assert v.ok, v.counts
    s = score_file(f, items, inst.cfg, idof)
    assert s["overall"] == 1.0

def test_type_mismatch_rejected(tmp_path, inst):
    from glyph.data import result_type, parse
    items = inst.val; idof = _idof(items)
    # flip each answer's type: VAL->a list, LIST->a single value
    rows = []
    for t in items:
        want = result_type(parse(t.expr_src, inst.cfg))
        ans = "[v_a_a, v_b_b]" if want == "VAL" else "v_a_a"
        rows.append({"id": idof(t), "answer": ans})
    f = tmp_path/"a.jsonl"; _write(f, rows)
    v = check_file(f, items, inst.cfg, tmp_path, idof)
    assert not v.ok and v.counts["type_mismatch"] > 0

def test_list_length_is_not_checked(tmp_path, inst):
    from glyph.data import result_type, parse
    items = [t for t in inst.val if result_type(parse(t.expr_src, inst.cfg)) == "LIST"][:3]
    assert items, "need some LIST-typed val items"
    idof = _idof(items)
    f = tmp_path/"a.jsonl"
    _write(f, [{"id": idof(t), "answer": "[v_a_a]"} for t in items])   # length 1, surely wrong
    v = check_file(f, items, inst.cfg, tmp_path, idof)
    assert v.ok, v.counts   # legal despite wrong length

def test_missing_unknown_duplicate_ids(tmp_path, inst):
    items = inst.val[:5]; idof = _idof(items)
    rows = [{"id": idof(items[0]), "answer": items[0].answer_src},
            {"id": idof(items[0]), "answer": items[0].answer_src},   # duplicate
            {"id": "val_99999", "answer": "v_a_a"}]                   # unknown; items[1..4] missing
    f = tmp_path/"a.jsonl"; _write(f, rows)
    v = check_file(f, items, inst.cfg, tmp_path, idof)
    assert v.counts["duplicate_ids"] >= 1
    assert v.counts["unknown_ids"] >= 1
    assert v.counts["missing_ids"] >= 1

def test_malformed_json_and_bad_shape(tmp_path, inst):
    items = inst.val[:2]; idof = _idof(items)
    f = tmp_path/"a.jsonl"
    f.write_text('{"id": "%s", "answer": "v_a_a"}\nnot json\n{"id":"x"}\n' % idof(items[0]), encoding="utf-8")
    v = check_file(f, items, inst.cfg, tmp_path, idof)
    assert v.counts["malformed_json"] >= 1
    assert v.counts["bad_shape"] >= 1     # {"id":"x"} missing "answer"

def test_bad_answer_value(tmp_path, inst):
    items = inst.val[:2]; idof = _idof(items)
    f = tmp_path/"a.jsonl"
    _write(f, [{"id": idof(items[0]), "answer": "v_z_z_z"},   # bad digit for smoke base
               {"id": idof(items[1]), "answer": "garbage"}])
    v = check_file(f, items, inst.cfg, tmp_path, idof)
    assert v.counts["bad_answer"] >= 1

def test_path_escape(tmp_path, inst):
    items = inst.val[:1]; idof = _idof(items)
    work = tmp_path/"work"; work.mkdir()
    task = work/"task"; task.mkdir()
    f = task/"sneak.jsonl"; _write(f, [{"id": idof(items[0]), "answer": "v_a_a"}])
    v = check_file(f, items, inst.cfg, work, idof)
    assert v.counts["path_escape"] >= 1 and not v.ok

def test_parse_answer():
    assert parse_answer("v_a_a", CFG)[0] == "VAL"
    assert parse_answer("[v_a_a, v_b_b]", CFG)[0] == "LIST"
    with pytest.raises(Exception): parse_answer("[]", CFG)   # empty list not allowed

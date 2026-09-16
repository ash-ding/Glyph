import json, sys
from pathlib import Path
sys.path.insert(0, "tools")
from collect_runs import collect


def _run(root, rid, arm, overall, pi):
    d = root / rid / "sub"
    d.mkdir(parents=True)
    (d / "run.json").write_text(json.dumps({
        "id": rid, "created": "2026-09-16T0%d:00:00Z" % (overall * 10),
        "config": {"arm": arm}, "summary": {"arm": arm, "overall": overall, "pi": pi},
        "transcript": [{"turn": 1}]}))


def test_collect_copies_and_indexes(tmp_path):
    root = tmp_path / "runs"
    _run(root, "a_train_1", "train", 0.42, 0.44)
    _run(root, "b_notrain_2", "no_train", 0.40, 0.25)
    out = tmp_path / "viewer_runs"
    index = collect([str(root)], str(out))
    assert len(index) == 2
    assert (out / "a_train_1.json").exists() and (out / "b_notrain_2.json").exists()
    idx = json.loads((out / "index.json").read_text())
    # newest-first by created; both summaries carry id + fields
    ids = {r["id"] for r in idx}
    assert ids == {"a_train_1", "b_notrain_2"}
    assert all("arm" in r and "overall" in r for r in idx)


def test_collect_dedupes_by_id(tmp_path):
    root1 = tmp_path / "r1"; root2 = tmp_path / "r2"
    _run(root1, "dup_1", "train", 0.42, 0.44)
    _run(root2, "dup_1", "train", 0.42, 0.44)
    index = collect([str(root1), str(root2)], str(tmp_path / "out"))
    assert len(index) == 1

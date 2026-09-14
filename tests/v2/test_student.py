import json, pathlib
from glyph.data import PRESETS, generate
from glyph.v2.ledger import Ledger
from glyph.v2.student import StudentPool

CFG = PRESETS["smoke"]

class FakeBackend:
    def __init__(self): self.train_calls = 0
    def train_fn(self, examples, hp, base_model, out_dir, ledger):
        self.train_calls += 1
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        return {"examples": len(examples), "final_loss": 0.001, "gpu_seconds": 100.0}
    def make_student(self, base_model, adapter_path, prefix):
        class S:
            def answer(self, exprs): return ["v_a_a" for _ in exprs]
            def close(self): pass
        return S()

def _pool(tmp_path, inst, queries=None, **kw):
    qp = None
    if queries is not None:
        qp = tmp_path/"queries.jsonl"; qp.write_text("\n".join(json.dumps({"expr":e,"out":"x"}) for e in queries)+"\n")
    return StudentPool("Qwen/Qwen3-1.7B", Ledger(), work_dir=tmp_path/"ck", queries_path=qp, backend=FakeBackend(), **kw)

def test_build_dataset_provenance(tmp_path):
    inst = generate(1001, CFG)
    demo_expr = inst.demos[0][0]; val_expr = inst.val[0].expr_src
    purchased_expr = "s0(u0, [v_a_a, v_b_b])"   # a plausible legal query; adjust if it fails check
    pool = _pool(tmp_path, inst, queries=[purchased_expr])
    ds = tmp_path/"ds.jsonl"
    ds.write_text("\n".join(json.dumps(r) for r in [
        {"expr": purchased_expr, "answer": "v_a_a"},
        {"expr": demo_expr, "answer": "v_a_a"},
        {"expr": val_expr, "answer": "v_a_a"},
    ])+"\n")
    rec = pool.build_dataset(ds, inst)
    p = rec["provenance"]
    assert p["purchased"] >= 1 and p["demo"] >= 1 and p["validation"] >= 1
    assert rec["size"] >= 3 and rec["dataset_id"].startswith("ds")

def test_train_delegates_and_meters(tmp_path):
    inst = generate(1001, CFG); pool = _pool(tmp_path, inst)
    ds = tmp_path/"d.jsonl"; ds.write_text(json.dumps({"expr": inst.demos[0][0], "answer":"v_a_a"})+"\n")
    dsid = pool.build_dataset(ds, inst)["dataset_id"]
    r = pool.train(dsid, epochs=1, lr=1e-5)
    assert r["checkpoint_id"] and abs(r["final_loss"]-0.001) < 1e-9 and r["gpu_seconds"] == 100.0
    assert r["stopped_at_cap"] is False
    assert pool.ledger.summary()["gpu_seconds"]["train"] == 100.0

def test_cumulative_cap_refuses_without_calling_backend(tmp_path):
    inst = generate(1001, CFG)
    pool = _pool(tmp_path, inst, gpu_cap_total_s=50.0)   # below one call's 100s
    ds = tmp_path/"d.jsonl"; ds.write_text(json.dumps({"expr": inst.demos[0][0], "answer":"v_a_a"})+"\n")
    dsid = pool.build_dataset(ds, inst)["dataset_id"]
    r1 = pool.train(dsid, 1, 1e-5)   # first call: 100s used, exceeds 50 total afterwards
    r2 = pool.train(dsid, 1, 1e-5)   # second: pre-check sees used>=cap -> refuse without backend
    assert r2["stopped_at_cap"] is True and r2.get("checkpoint_id") is None
    assert pool.backend.train_calls == 1    # backend NOT called the second time

def test_infer_writes_answers(tmp_path):
    inst = generate(1001, CFG); pool = _pool(tmp_path, inst)
    inp = tmp_path/"in.jsonl"; out = tmp_path/"out.jsonl"
    inp.write_text("\n".join(json.dumps({"id":f"t{i}","expr":inst.test[i].expr_src}) for i in range(3))+"\n")
    r = pool.infer("base", inp, out, None)
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert r["rows"] == 3 and len(rows) == 3 and set(rows[0]) == {"id","answer"}

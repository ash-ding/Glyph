# PR B — `student_id` + continue-training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `train` takes a `student_id` — a new id fine-tunes from the base model, an existing id continues from that student's latest checkpoint; students coexist; `student_infer` also accepts a student_id (meaning its latest checkpoint). The prompt describes the mechanics and never the strategy.

**Architecture:** `StudentPool` grows a `students: {student_id: [ck…]}` lineage map; `train` resolves the init weights and passes `init_checkpoint` through the backend to `sft.train`, which loads the model from it instead of the base. Checkpoints stay globally numbered (`ckN`) and remain valid `student_infer` targets. GPU accounting unchanged (shared caps, training metered, inference not).

**Tech Stack:** Python 3.11, pytest; torch/transformers only inside `train/` (unchanged rule).

**Spec:** `docs/superpowers/specs/2026-09-29-bare-atomics-multistudent-probes-design.md` (§B)

## Global Constraints

- Host lumen1, repo `~/code/Glyph`, branch `feat-student-pool` off `main`. Interpreter `~/miniforge3/envs/glyph/bin/python`.
- The prompt paragraph states MECHANICS ONLY — no hint that per-op specialists are a good idea; the delegate-or-not decision is the quantity under test. Mark this with a comment.
- No torch import outside `train/`; the pool tests use a fake backend.
- `pytest -q -m "not slow"` green before the merge. Commits end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: `StudentPool` lineages — `train(student_id)` + `infer` by student id

**Files:**
- Modify: `src/glyph/v2/student.py` (`StudentPool.__init__`, `train`, `infer`)
- Test: `tests/v2/test_student.py`

**Interfaces:**
- Produces: `StudentPool.train(dataset_id, epochs, lr, student_id="s1") -> dict` — result gains `"student_id"` and `"continued_from"` (previous ck id or `None`); pool attribute `students: dict[str, list[str]]`. `StudentPool.infer(checkpoint, ...)` — `checkpoint` may be `"base"`, a ck id, or a student id (resolves to that student's latest ck).
- Consumes: backend `train_fn(examples, hp, base_model, out_dir, ledger, init_checkpoint=None)` (Task 2 adds the real one; the fake here defines the contract).

- [ ] **Step 1: Write the failing tests** (append to `tests/v2/test_student.py`, reusing its existing fake-backend/fixture style — if its fake backend lacks `init_checkpoint`, extend that fake in place)

```python
class RecordingBackend:
    """Fake backend that records every train_fn call's init_checkpoint."""
    def __init__(self):
        self.calls = []

    def train_fn(self, examples, hp, base_model, out_dir, ledger,
                 init_checkpoint=None):
        self.calls.append({"init_checkpoint": init_checkpoint,
                           "out_dir": str(out_dir)})
        return {"gpu_seconds": 1.0, "final_loss": 0.5}

    def make_student(self, base_model, adapter_path, prefix):
        class _S:
            def answer(self, exprs):
                return ["v_a_a"] * len(exprs)
            def close(self):
                pass
        return _S()


def _pool_with(tmp_path, backend):
    from glyph.v2.ledger import Ledger
    from glyph.v2.student import StudentPool
    pool = StudentPool("base-model", Ledger(), work_dir=tmp_path,
                       backend=backend)
    pool.datasets["ds1"] = []
    return pool


def test_new_student_id_trains_from_base(tmp_path):
    be = RecordingBackend()
    pool = _pool_with(tmp_path, be)
    rec = pool.train("ds1", epochs=1, lr=1e-4, student_id="alpha")
    assert be.calls[0]["init_checkpoint"] is None
    assert rec["student_id"] == "alpha" and rec["continued_from"] is None
    assert pool.students["alpha"] == [rec["checkpoint_id"]]


def test_existing_student_id_continues_from_latest(tmp_path):
    be = RecordingBackend()
    pool = _pool_with(tmp_path, be)
    r1 = pool.train("ds1", epochs=1, lr=1e-4, student_id="alpha")
    r2 = pool.train("ds1", epochs=1, lr=1e-4, student_id="alpha")
    assert be.calls[1]["init_checkpoint"] == str(pool.checkpoints[r1["checkpoint_id"]])
    assert r2["continued_from"] == r1["checkpoint_id"]
    assert pool.students["alpha"] == [r1["checkpoint_id"], r2["checkpoint_id"]]


def test_two_student_ids_are_independent(tmp_path):
    be = RecordingBackend()
    pool = _pool_with(tmp_path, be)
    pool.train("ds1", epochs=1, lr=1e-4, student_id="alpha")
    pool.train("ds1", epochs=1, lr=1e-4, student_id="beta")
    assert be.calls[1]["init_checkpoint"] is None
    assert set(pool.students) == {"alpha", "beta"}


def test_infer_accepts_student_id_as_latest_checkpoint(tmp_path):
    import json
    be = RecordingBackend()
    pool = _pool_with(tmp_path, be)
    r1 = pool.train("ds1", epochs=1, lr=1e-4, student_id="alpha")
    r2 = pool.train("ds1", epochs=1, lr=1e-4, student_id="alpha")
    inp = tmp_path / "in.jsonl"
    inp.write_text(json.dumps({"id": "x", "expr": "e"}) + "\n")
    out = tmp_path / "out.jsonl"
    # student id resolves; a fresh pool dir has no train_record.json so the
    # checkpoint resolves as a full fine-tune (served AS the model)
    pool.checkpoints[r2["checkpoint_id"]].mkdir(parents=True, exist_ok=True)
    rec = pool.infer("alpha", str(inp), str(out))
    assert rec["rows"] == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `~/miniforge3/envs/glyph/bin/python -m pytest tests/v2/test_student.py -q -k "student_id or continues or independent"`
Expected: FAIL — `TypeError: train() got an unexpected keyword argument 'student_id'`

- [ ] **Step 3: Implement.** In `student.py`:

`__init__` gains, next to `self.checkpoints`:

```python
        self.students: dict[str, list[str]] = {}
```

`train` signature becomes `def train(self, dataset_id, epochs, lr, student_id="s1") -> dict:` and, replacing the body between the cap check and the `backend.train_fn` call:

```python
        examples = self.datasets[dataset_id]
        hp = HParams(full_finetune=True, epochs=epochs, lr=lr)

        # Lineage: a known student continues from its latest checkpoint; a new
        # one starts from the base model.  Full fine-tunes only, so the
        # checkpoint dir IS a loadable model (see _resolve_checkpoint).
        lineage = self.students.setdefault(student_id, [])
        continued_from = lineage[-1] if lineage else None
        init_checkpoint = (str(self.checkpoints[continued_from])
                           if continued_from else None)

        self._ck_n += 1
        checkpoint_id = f"ck{self._ck_n}"
        out_dir = self.work_dir / checkpoint_id

        rec = self.backend.train_fn(examples, hp, base_model=self.base_model,
                                     out_dir=out_dir, ledger=self.ledger,
                                     init_checkpoint=init_checkpoint)
```

after `self.checkpoints[checkpoint_id] = out_dir` add:

```python
        lineage.append(checkpoint_id)
```

and add to the returned dict: `"student_id": student_id, "continued_from": continued_from,`.

In `infer`, replace the checkpoint resolution with:

```python
        if checkpoint == "base":
            base_model, adapter_path = self.base_model, None
        else:
            ck = checkpoint
            if ck not in self.checkpoints and ck in self.students and self.students[ck]:
                ck = self.students[ck][-1]      # a student id means its latest ck
            base_model, adapter_path = self._resolve_checkpoint(self.checkpoints[ck])
```

- [ ] **Step 4: Run to verify pass** (whole file — the pre-existing pool tests must not regress; if an existing fake backend's `train_fn` lacks `init_checkpoint`, add the keyword there with a default)

Run: `~/miniforge3/envs/glyph/bin/python -m pytest tests/v2/test_student.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/glyph/v2/student.py tests/v2/test_student.py
git commit -m "feat(student): per-student lineages — new id from base, known id continues"
```

---

### Task 2: `sft.train` accepts `init_checkpoint`; real backend passes it

**Files:**
- Modify: `src/glyph/train/sft.py` (`train` signature + model load + record)
- Modify: `src/glyph/v2/student.py` (`_RealBackend.train_fn`)

**Interfaces:**
- Produces: `sft.train(..., init_checkpoint: str | None = None)` — loads the model from `init_checkpoint` when given, else from `base_model`; tokenizer always from `base_model`; `train_record.json` gains `"init_checkpoint"`.

- [ ] **Step 1: Implement** (no new GPU test — the contract is pinned by Task 1's fake; the real path is covered by the `slow`-marked end-to-end tests and the deferred weights runs). In `sft.py`:

signature: `def train(examples, hp, *, base_model, out_dir, ledger=None, device="cuda", log_every=50, init_checkpoint=None) -> dict:`

model load becomes:

```python
    model = AutoModelForCausalLM.from_pretrained(
        init_checkpoint if init_checkpoint else base_model,
        dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
```

record gains `"init_checkpoint": init_checkpoint,` next to `"base_model"`.

In `student.py`'s `_RealBackend`:

```python
        def train_fn(self, examples, hp, base_model, out_dir, ledger,
                     init_checkpoint=None):
```

and pass `init_checkpoint=init_checkpoint` into the `sft.train(...)` call.

- [ ] **Step 2: Sanity-run the torch-free import gate and the student tests**

Run: `~/miniforge3/envs/glyph/bin/python -m pytest tests/v2/test_student.py tests/test_data_boundary.py -q`
Expected: PASS (sft is only imported lazily inside `_RealBackend`)

- [ ] **Step 3: Commit**

```bash
git add src/glyph/train/sft.py src/glyph/v2/student.py
git commit -m "feat(train): sft accepts init_checkpoint for continued fine-tuning"
```

---

### Task 3: Tool + MCP schema

**Files:**
- Modify: `src/glyph/v2/tools.py` (`t_train`), `src/glyph/v2/mcp.py` (schema + description)
- Test: `tests/v2/test_tools.py` (update `FakeStudent`, add pass-through test)

**Interfaces:**
- Produces: `t_train(session, dataset_id, epochs, lr, student_id)` (required arg); MCP `train` schema `{"dataset_id": str, "epochs": int, "lr": float, "student_id": str}`.

- [ ] **Step 1: Write the failing test** (append to `tests/v2/test_tools.py`; also extend the existing `FakeStudent.train` signature to `def train(self, dataset_id, epochs, lr, student_id="s1"):` and have it echo `"student_id": student_id` in its return)

```python
def test_train_passes_student_id_through(inst, tmp_path):
    s = make_session(inst, tmp_path)
    s.student = FakeStudent()
    out = T.t_train(s, dataset_id="ds1", epochs=1, lr=1e-4, student_id="alpha")
    assert out["student_id"] == "alpha"
```

- [ ] **Step 2: Run to verify failure**

Run: `~/miniforge3/envs/glyph/bin/python -m pytest tests/v2/test_tools.py -q -k student_id`
Expected: FAIL — `TypeError: t_train() got an unexpected keyword argument 'student_id'`

- [ ] **Step 3: Implement.** `tools.py`:

```python
def t_train(session, dataset_id, epochs, lr, student_id) -> dict:
```

and pass `student_id=student_id` into `student.train(...)`. `mcp.py`:

```python
    "train": (_T.t_train, {"dataset_id": str, "epochs": int, "lr": float,
                            "student_id": str}),
```

and the description:

```python
    "train": ("Fine-tune a student model on a dataset (train arm). A new "
              "student_id starts a fresh student from the base model; an "
              "existing student_id continues training that student from its "
              "latest checkpoint. Several students may coexist."),
```

Also update `student_infer`'s description to note `checkpoint` accepts a checkpoint id, `"base"`, or a student_id (its latest checkpoint).

- [ ] **Step 4: Run to verify pass**

Run: `~/miniforge3/envs/glyph/bin/python -m pytest tests/v2/test_tools.py tests/v2/test_harness_logic.py -q`
Expected: PASS (harness-logic tests exercise the tool table; fix any fixture calling `t_train` without `student_id` by adding one)

- [ ] **Step 5: Commit**

```bash
git add src/glyph/v2/tools.py src/glyph/v2/mcp.py tests/v2/test_tools.py
git commit -m "feat(tools): train takes a required student_id; schema + descriptions"
```

---

### Task 4: Prompt mechanics + docs

**Files:**
- Modify: `src/glyph/v2/prompts.py` (`_STUDENT_PARAGRAPH`), `docs/tools.md`
- Test: `tests/v2/test_prompts.py` (run; adapt any assertion pinned to the old paragraph)

- [ ] **Step 1: Replace `_STUDENT_PARAGRAPH` with** (mechanics only — the comment is load-bearing):

```python
# Student paragraph: only for the train arm.
# MECHANICS ONLY. Never hint at a strategy (e.g. "train one specialist per
# operator") -- whether and how to delegate to students is the quantity this
# benchmark measures, and a hint here would contaminate it.
_STUDENT_PARAGRAPH = """
**Building Student Models (train arm only)**

During practice, you may also build student models:
- build_dataset: collect training data from queries you make
- train: fine-tune a student on a dataset. You choose a student_id: a new id
  starts a fresh student from the base model; an existing id continues
  training that student from its latest checkpoint. Several students may
  coexist.
- student_infer: generate answers with a trained checkpoint, or with a
  student_id (meaning that student's latest checkpoint)

These tools let you put what you discover into a small model's weights.
Training and inference consume turns but not queries."""
```

- [ ] **Step 2: Run the prompt tests, adapt pinned strings forward**

Run: `~/miniforge3/envs/glyph/bin/python -m pytest tests/v2/test_prompts.py -q`
Expected: PASS after updating any assertion that quoted the old paragraph verbatim.

- [ ] **Step 3: `docs/tools.md`** — update the `train` entry signature to `train(dataset_id, epochs, lr, student_id)` with the new-vs-existing id semantics, and the `student_infer` entry to note the student_id form of `checkpoint`.

- [ ] **Step 4: Commit**

```bash
git add src/glyph/v2/prompts.py tests/v2/test_prompts.py docs/tools.md
git commit -m "feat(prompts): student-pool mechanics in the train-arm paragraph"
```

---

### Task 5: Full suite, progress entry, PR

- [ ] **Step 1: Full fast suite**

Run: `~/miniforge3/envs/glyph/bin/python -m pytest -q -m "not slow"`
Expected: all pass.

- [ ] **Step 2: `docs/progress.md`** — append a PR B entry: what changed (lineages, init_checkpoint, required student_id, prompt mechanics), the command and pass count, and the sentence that GPU accounting is unchanged (shared caps; training metered, inference not — the unmetered-inference issue stays open).

- [ ] **Step 3: Commit docs, push, open PR; stop for the user's merge approval**

```bash
git add docs/progress.md
git commit -m "docs: progress entry for the student-pool change"
git push -u origin feat-student-pool
gh pr create --title "feat: student_id lineages — continue-training and coexisting students" --body-file <body>
```

PR body: §B summary, proof block, "Generated with Claude Code" line. **Stop: the user approves the merge.**

"""Tool-handler core for protocol v2.

Plain-Python handler functions, one per protocol tool: `handler(session,
**args) -> dict`.  All state work (charging, counters, phase gating,
notes/events) happens here; the MCP decoration around these is a separate
layer (Task 17). No SDK import, no GPU/train import -- the `student` object
is duck-typed and injected onto `session.student` by the harness (or by a
test, as a fake).

Cross-cutting rules (see the brief, spec Sec 6.1):
  - every handler's FIRST line is the phase/arm availability gate. This is
    the authoritative enforcement of the tool matrix -- tools cannot be
    removed from the model's visible request set within a session, so the
    gate is what actually stops a disallowed call from doing anything.
  - every success return includes q_remaining/submissions_remaining/
    turns_remaining. Error returns need not, but may.
  - refused and q_exhausted queries are NOT charged; malformed IS charged
    (the agent must not be able to probe the grammar for free).
  - validation feedback (t_submit) is aggregate-only: overall + by_depth.
    Never per-item, never by_split.
  - illegal submit / illegal final_answer do not count / do not commit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from glyph.v2.answers import check_file, score_file


def default_id_of(prefix: str, items: list) -> Callable[[Any], str]:
    """Fallback id map: identity-keyed, f'{prefix}_{i:05d}' in list order."""
    index = {id(t): i for i, t in enumerate(items)}

    def id_of(t: Any) -> str:
        return f"{prefix}_{index[id(t)]:05d}"

    return id_of


def _remaining(session) -> dict:
    return {
        "q_remaining": session.q_remaining(),
        "submissions_remaining": session.submissions_remaining(),
        "turns_remaining": session.turns_remaining(),
    }


def _val_id_of(session) -> Callable[[Any], str]:
    id_of = getattr(session, "val_id_of", None)
    if id_of is None:
        id_of = default_id_of("val", session.inst.val)
    return id_of


def _test_id_of(session) -> Callable[[Any], str]:
    id_of = getattr(session, "test_id_of", None)
    if id_of is None:
        id_of = default_id_of("test", session.inst.test)
    return id_of


def _examples(v) -> dict:
    return {k: getattr(v, k) for k in v.counts if v.counts[k]}


def _gate_error(session, name: str) -> dict:
    return {"error": f"tool '{name}' not available in phase {session.phase} "
                     f"for arm {session.arm}"}


# ---------------------------------------------------------------------
# query
# ---------------------------------------------------------------------

def t_query(session, exprs: list, why: str) -> dict:
    if not session.tool_available("query"):
        return _gate_error(session, "query")

    results = []
    for expr in exprs:
        reason = session.inst.query_violation(expr, session.policy)
        if reason:
            results.append({"expr": expr, "refused": reason})
            continue
        if session.q_remaining() <= 0:
            results.append({"expr": expr, "error": "q_exhausted"})
            continue
        # Charge first -- malformed expressions are charged too.
        session.q_used += 1
        try:
            out = session.inst.query(expr)
        except Exception:
            results.append({"expr": expr, "error": "malformed"})
            continue
        results.append({"expr": expr, "out": out})
        task_dir = Path(session.run_dir) / "task"
        task_dir.mkdir(parents=True, exist_ok=True)
        with open(task_dir / "queries.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"expr": expr, "out": out}) + "\n")

    session.note(kind="query", why=why[:200], n=len(exprs))
    return {"results": results, "q_used": session.q_used, **_remaining(session)}


# ---------------------------------------------------------------------
# submit (validation)
# ---------------------------------------------------------------------

def t_submit(session, path) -> dict:
    if not session.tool_available("submit"):
        return _gate_error(session, "submit")

    id_of = _val_id_of(session)
    v = check_file(path, session.inst.val, session.inst.cfg, session.run_dir, id_of)
    if not v.ok:
        return {"error": "illegal submission (not counted)",
                "violations": v.counts, "examples": _examples(v),
                **_remaining(session)}

    s = score_file(path, session.inst.val, session.inst.cfg, id_of)
    session.submissions += 1
    fb = {"submission": session.submissions, "overall": s["overall"],
          "by_depth": s["by_depth"]}
    session.note(kind="validation_submit", **fb)
    if session.submissions >= session.submit_cap:
        session.should_switch = True
    return {**fb, **_remaining(session)}


# ---------------------------------------------------------------------
# check_answers (both phases)
# ---------------------------------------------------------------------

def t_check_answers(session, path, set) -> dict:
    if not session.tool_available("check_answers"):
        return _gate_error(session, "check_answers")

    if set not in ("validation", "test"):
        return {"error": f"unknown set {set!r}", **_remaining(session)}
    if set == "test" and session.phase != "final":
        return {"error": "set='test' only valid in the final phase",
                **_remaining(session)}

    if set == "validation":
        items, id_of = session.inst.val, _val_id_of(session)
    else:
        items, id_of = session.inst.test, _test_id_of(session)

    v = check_file(path, items, session.inst.cfg, session.run_dir, id_of)
    out = {"ok": v.ok, "violations": v.counts, "examples": _examples(v),
           **_remaining(session)}

    if set == "test" and v.ok and session.phase == "final":
        session.last_checked_test_path = str(path)

    return out


# ---------------------------------------------------------------------
# finish_practice
# ---------------------------------------------------------------------

def t_finish_practice(session, reason: str) -> dict:
    if not session.tool_available("finish_practice"):
        return _gate_error(session, "finish_practice")

    session.should_switch = True
    session.note(kind="finish_practice", reason=reason[:200])
    return {"ok": True, **_remaining(session)}


# ---------------------------------------------------------------------
# train-arm tools (student is duck-typed and injected)
# ---------------------------------------------------------------------

def _student_or_error(session):
    student = getattr(session, "student", None)
    if student is None:
        return None, {"error": "no student available for this session",
                      **_remaining(session)}
    return student, None


def t_build_dataset(session, path) -> dict:
    if not session.tool_available("build_dataset"):
        return _gate_error(session, "build_dataset")

    student, err = _student_or_error(session)
    if err is not None:
        return err
    rec = student.build_dataset(path, session.inst)
    return {**rec, **_remaining(session)}


def t_train(session, dataset_id, epochs, lr) -> dict:
    if not session.tool_available("train"):
        return _gate_error(session, "train")

    student, err = _student_or_error(session)
    if err is not None:
        return err
    rec = student.train(dataset_id, epochs, lr)
    return {**rec, **_remaining(session)}


def t_student_infer(session, checkpoint, input_path, output_path,
                    prefix_path=None) -> dict:
    if not session.tool_available("student_infer"):
        return _gate_error(session, "student_infer")

    student, err = _student_or_error(session)
    if err is not None:
        return err
    rec = student.infer(checkpoint, input_path, output_path, prefix_path)
    return {**rec, **_remaining(session)}


# ---------------------------------------------------------------------
# final_answer
# ---------------------------------------------------------------------

def t_final_answer(session, path) -> dict:
    if not session.tool_available("final_answer"):
        return _gate_error(session, "final_answer")

    id_of = _test_id_of(session)
    v = check_file(path, session.inst.test, session.inst.cfg, session.run_dir, id_of)
    if not v.ok:
        return {"error": "illegal final answer (not committed)",
                "violations": v.counts, "examples": _examples(v),
                **_remaining(session)}

    session.final_commit_path = str(path)
    session.final_commit = "agent"
    session.note(kind="final_answer", committed=True)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    return {"committed": True, "digest": digest, **_remaining(session)}

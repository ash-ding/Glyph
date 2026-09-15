"""The single answer-file legality checker and scorer.

Used by three tools (submit, check_answers, final_answer) to decide whether a
candidate answer file is *legal*, and by the report to score a file that has
already passed that check.  Pure: imports only `glyph.data` (+ stdlib) -- no
SDK, no GPU, no network, no other v2 module.

Two things this deliberately does NOT do, both load-bearing:
  - it never runs `glyph.data.grammar.check()` on an answer.  `check` polices
    *expressions* (e.g. list_len_range for literals inside a query); answers
    are just values, and an output list is allowed to be longer or shorter
    than list_len_range.  Running check() on an answer would wrongly reject
    correct long/short lists.
  - it never compares list *lengths* -- only that the parsed answer's TYPE
    (VAL vs LIST) matches the expression's result_type.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from glyph.data import GlyphConfig, TestItem, depth, parse, parse_value, result_type

_VIOLATION_FIELDS = (
    "malformed_json", "bad_shape", "unknown_ids", "missing_ids",
    "duplicate_ids", "bad_answer", "type_mismatch", "path_escape",
)
_MAX_EXAMPLES = 5


@dataclass
class Violations:
    malformed_json: list = field(default_factory=list)
    bad_shape: list = field(default_factory=list)
    unknown_ids: list = field(default_factory=list)
    missing_ids: list = field(default_factory=list)
    duplicate_ids: list = field(default_factory=list)
    bad_answer: list = field(default_factory=list)
    type_mismatch: list = field(default_factory=list)
    path_escape: list = field(default_factory=list)
    counts: dict = field(default_factory=lambda: {k: 0 for k in _VIOLATION_FIELDS})

    @property
    def ok(self) -> bool:
        return all(self.counts.get(k, 0) == 0 for k in _VIOLATION_FIELDS)

    def add(self, category: str, msg: str) -> None:
        self.counts[category] = self.counts.get(category, 0) + 1
        bucket = getattr(self, category)
        if len(bucket) < _MAX_EXAMPLES:
            bucket.append(msg)


def parse_answer(s: str, cfg: GlyphConfig) -> tuple:
    """Parse a raw answer string into ("VAL", idx) or ("LIST", [idx, ...]).

    Deliberately does NOT use the expression parser or grammar.check: output
    lists may be longer than list_len_range, which check() would reject.
    """
    t = s.strip()
    if t.startswith("["):
        if not t.endswith("]"):
            raise SyntaxError(f"unterminated list {s!r}")
        inner = t[1:-1].strip()
        if not inner:
            raise SyntaxError("empty list is not a legal answer")
        toks = [tok.strip() for tok in inner.split(",")]
        vals = [parse_value(tok, cfg) for tok in toks]
        return ("LIST", vals)
    idx = parse_value(t, cfg)
    return ("VAL", idx)


def check_file(path, items: list, cfg: GlyphConfig, work_root,
               id_of: Callable) -> Violations:
    v = Violations()

    work_root = Path(work_root).resolve()
    resolved = Path(path).resolve()
    task_root = work_root / "task"
    is_inside_work = resolved == work_root or work_root in resolved.parents
    is_under_task = resolved == task_root or task_root in resolved.parents
    if not is_inside_work or is_under_task:
        v.add("path_escape", f"{resolved} is not inside {work_root} (or is under task/)")
        return v

    # Precompute expected result type per item, O(n).
    target = {id_of(t): t for t in items}
    expected_type = {tid: result_type(parse(t.expr_src, cfg)) for tid, t in target.items()}

    seen: dict[str, int] = {}
    text = resolved.read_text(encoding="utf-8")
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception as e:
            v.add("malformed_json", f"line {lineno}: {e}")
            continue
        if (not isinstance(obj, dict) or set(obj.keys()) != {"id", "answer"}
                or not isinstance(obj.get("id"), str)
                or not isinstance(obj.get("answer"), str)):
            v.add("bad_shape", f"line {lineno}: expected {{'id': str, 'answer': str}}, got {obj!r}")
            continue

        tid = obj["id"]
        count = seen.get(tid, 0)
        seen[tid] = count + 1
        if count >= 1:
            v.add("duplicate_ids", f"line {lineno}: id {tid!r} appears more than once")
            continue  # only the first occurrence is answer/type-checked

        if tid not in target:
            v.add("unknown_ids", f"line {lineno}: id {tid!r} not in target set")
            continue

        answer = obj["answer"]
        try:
            kind, _ = parse_answer(answer, cfg)
        except Exception as e:
            v.add("bad_answer", f"id {tid!r} (line {lineno}): {e}")
            continue

        if kind != expected_type[tid]:
            v.add("type_mismatch",
                  f"id {tid!r} (line {lineno}): answer is {kind}, expected {expected_type[tid]}")

    for tid in target:
        if tid not in seen:
            v.add("missing_ids", f"id {tid!r} missing from file")

    return v


def score_file(path, items: list, cfg: GlyphConfig, id_of: Callable) -> dict:
    """Assumes `path` already passed check_file. Exact match after .strip()."""
    answers: dict[str, str] = {}
    text = Path(path).read_text(encoding="utf-8")
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("id"), str):
            if obj["id"] not in answers:
                answers[obj["id"]] = obj.get("answer", "")

    scores: list = []
    by_split: dict[str, list] = {}
    by_depth: dict[str, list] = {}
    for t in items:
        tid = id_of(t)
        ans = answers.get(tid, None)
        s = 1 if (ans is not None and ans.strip() == t.answer_src.strip()) else 0
        scores.append(s)
        by_split.setdefault(t.split, []).append(s)
        d = str(depth(parse(t.expr_src, cfg)))
        by_depth.setdefault(d, []).append(s)

    def mean(xs: list) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "overall": mean(scores),
        "by_split": {k: mean(v) for k, v in by_split.items()},
        "by_depth": {k: mean(v) for k, v in by_depth.items()},
    }

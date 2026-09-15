"""Per-run workspace: the read-only task/ files the sandbox sees, and item id
assignment for validation and (shuffled) test items.

Built once per run (Task 17 calls this at prepare-phase start, and again --
via write_test_file -- at the phase switch into final). Nothing here writes
test.jsonl at build time: the held-out test file is written only when the
harness flips to the final phase, so a running agent can never find it early
by walking the filesystem.

Id scheme matches Task 10's default_id_of (glyph.v2.tools): identity-keyed
(id(item)), f"{prefix}_{i:05d}" in the order given. Validation ids follow
inst.val order. Test ids are assigned in a SHUFFLED order (numpy
default_rng(inst.seed ^ 0x7e57)) so that neither the id number nor the
on-disk row order tells an agent anything about which split (iid/comp/depth)
a test item belongs to.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

# Fixed XOR mask for deriving the test-shuffle seed from the instance seed.
# Arbitrary but fixed constant -- see the brief for task 11. Checked against
# the "smoke" preset (100 test items): this constant does not reproduce the
# identity order, so the shuffled-ness assertion is not flaky for it.
_SHUFFLE_XOR = 0x7e57


@dataclass
class Paths:
    root: Path
    task: Path
    validation: Path
    queries: Path
    final_dir: Path
    readme: Path
    syntax: Path
    demos: Path


def shuffle_for(seed: int) -> np.random.Generator:
    """The fixed rng used to shuffle test-item order for a given instance seed."""
    return np.random.default_rng(seed ^ _SHUFFLE_XOR)


def assign_ids(prefix: str, items: list) -> tuple[Callable[[Any], str], list[tuple[str, Any]]]:
    """Identity-keyed id map over `items`, in the given order.

    Matches glyph.v2.tools.default_id_of exactly: f"{prefix}_{i:05d}" by
    position in `items`. Returns the id_of closure plus the ordered
    (id, item) pairs, since most callers immediately want both.
    """
    index = {id(t): i for i, t in enumerate(items)}

    def id_of(t: Any) -> str:
        return f"{prefix}_{index[id(t)]:05d}"

    ordered = [(id_of(t), t) for t in items]
    return id_of, ordered


_README = """\
# Glyph task workspace

Files in this directory:
  syntax.md          the expression grammar you may write
  demos.jsonl         worked examples: {"expr", "answer"} per line
  validation.jsonl    {"id", "expr"} per line -- practice items with a scored
                      oracle; no answers are included
  queries.jsonl       a growing log of every query you have made this run
  final/              empty until the run switches to the final phase

Rules:
  - Every call to the query tool spends from a fixed query budget.
  - During practice you may submit answers to the validation items up to 20
    times; each submission is scored against the held validation answers and
    the aggregate score is fed back to you.
  - Turns are capped separately for the practice phase (T_p) and the final
    phase (T_f).
  - After practice, the run switches to a final phase with a held-out test
    set and no oracle: your last submission there is scored, but nothing is
    fed back to you.
  - To submit answers, write a JSONL file of {"id", "answer"} rows that
    covers every id in the relevant file (validation.jsonl now, later
    final/test.jsonl) exactly once. An answer is either a value of the form
    "v_x_y_z" or a non-empty list [...].
"""


def build_workspace(inst, run_dir) -> tuple[Paths, Callable[[Any], str], Callable[[Any], str]]:
    """Create the per-run workspace under `run_dir/work/` and return its
    Paths plus the val/test id_of closures.

    Does NOT write test.jsonl -- see write_test_file, called separately at
    the phase switch (Task 17).
    """
    root = Path(run_dir) / "work"
    task = root / "task"
    final_dir = task / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    paths = Paths(
        root=root,
        task=task,
        validation=task / "validation.jsonl",
        queries=task / "queries.jsonl",
        final_dir=final_dir,
        readme=task / "README.md",
        syntax=task / "syntax.md",
        demos=task / "demos.jsonl",
    )

    paths.readme.write_text(_README)
    paths.syntax.write_text(inst.syntax_spec())

    demos_lines = [json.dumps({"expr": e, "answer": a}) for e, a in inst.demos]
    paths.demos.write_text("\n".join(demos_lines) + ("\n" if demos_lines else ""))

    val_id_of, val_ordered = assign_ids("val", inst.val)
    val_lines = [json.dumps({"id": i, "expr": t.expr_src}) for i, t in val_ordered]
    paths.validation.write_text("\n".join(val_lines) + ("\n" if val_lines else ""))

    paths.queries.write_text("")

    shuffled_test = list(inst.test)
    shuffle_for(inst.seed).shuffle(shuffled_test)
    test_id_of, _ = assign_ids("test", shuffled_test)

    return paths, val_id_of, test_id_of


def write_test_file(paths: Paths, inst, test_id_of: Callable[[Any], str]) -> Path:
    """Write final/test.jsonl: one {"id","expr"} per test item, in the
    SHUFFLED order test_id_of was built from (ids sort back into that order
    since they were assigned 0.. in it). No answer, no split label.
    """
    items = sorted(inst.test, key=test_id_of)
    lines = [json.dumps({"id": test_id_of(t), "expr": t.expr_src}) for t in items]
    out = paths.final_dir / "test.jsonl"
    out.write_text("\n".join(lines) + ("\n" if lines else ""))
    return out

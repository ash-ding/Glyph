"""CPU-only tests for the reference-battery runner (#task 8).

Only `merge_reference` and `run_cheap` are exercised here -- the CLI's
weights (GPU) and a0prime (API) branches are wired but never executed by
this suite; that happens in a later step.
"""

import json
import sys

sys.path.insert(0, "tools")
from run_reference import (_align_answers, _run_a0prime_scan,  # noqa: E402
                            merge_reference, run_cheap)

from glyph.data import PRESETS, generate  # noqa: E402
from glyph.reference.subset import paired_subset  # noqa: E402


def test_merge_per_oracle_idempotent(tmp_path):
    p = tmp_path / "reference_ceilings.json"

    merge_reference(p, "low_1", "weights", {"a": 1})
    merge_reference(p, "low_1", "a0prime", {"b": 2})

    with p.open() as f:
        data = json.load(f)
    assert data["low_1"]["weights"] == {"a": 1}
    assert data["low_1"]["a0prime"] == {"b": 2}

    merge_reference(p, "low_1", "weights", {"a": 9})

    with p.open() as f:
        data = json.load(f)
    assert data["low_1"]["weights"] == {"a": 9}
    assert data["low_1"]["a0prime"] == {"b": 2}

    merge_reference(p, "mid_1", "skeleton", {"c": 3})

    with p.open() as f:
        data = json.load(f)
    assert data["low_1"]["weights"] == {"a": 9}
    assert data["low_1"]["a0prime"] == {"b": 2}
    assert data["mid_1"]["skeleton"] == {"c": 3}


def test_run_cheap_shape():
    inst = generate(1001, PRESETS["smoke"])
    items = paired_subset(inst, 40)

    out = run_cheap(inst, items)

    assert set(out.keys()) == {"skeleton", "table", "perfect"}
    for oracle in ("skeleton", "table", "perfect"):
        assert set(out[oracle].keys()) == {"overall", "by_split", "tail", "headroom"}


def test_align_answers_ignores_preamble():
    reply = "Here are the answers:\n1. v_a\n2. v_b\n3. v_c"
    assert _align_answers(reply, 3) == ["v_a", "v_b", "v_c"]

    reply_missing = "1. v_a\n3. v_c"
    assert _align_answers(reply_missing, 3) == ["v_a", "", "v_c"]


def test_a0prime_scan_evidence_is_cumulative():
    """B1: evidence at a later frac must include every earlier frac's facts.

    `buy_evidence` only returns the facts bought *in that call*, but the
    scan reuses the same `inst` across fracs -- so a naive per-frac evidence
    block would silently drop everything revealed at earlier fracs. This
    records the `evidence` string `_run_a0prime_scan` actually hands to
    `answer_fn` at each frac and asserts each later frac's evidence lines
    are a strict superset of the previous frac's.
    """
    inst = generate(1001, PRESETS["smoke"])
    items = paired_subset(inst, 20)

    seen_evidence: list[str] = []

    def recording_fn(evidence: str, expr_srcs: list[str]) -> list[str]:
        seen_evidence.append(evidence)
        return ["" for _ in expr_srcs]

    fracs = [0.1, 0.4, 0.8]
    _run_a0prime_scan(inst, items, recording_fn, fracs)

    assert len(seen_evidence) == len(fracs)
    lines_by_frac = [set(e.splitlines()) for e in seen_evidence]

    assert lines_by_frac[0], "first frac should have bought some evidence"
    for earlier, later in zip(lines_by_frac, lines_by_frac[1:]):
        assert earlier <= later, "a later frac must retain every earlier fact"
    assert len(lines_by_frac[-1]) > len(lines_by_frac[0]), (
        "coverage should strictly grow across the scan")

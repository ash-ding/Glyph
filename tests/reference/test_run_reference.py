"""CPU-only tests for the reference-battery runner (#task 8).

Only `merge_reference` and `run_cheap` are exercised here -- the CLI's
weights (GPU) and a0prime (API) branches are wired but never executed by
this suite; that happens in a later step.
"""

import json
import sys

sys.path.insert(0, "tools")
from run_reference import merge_reference, run_cheap  # noqa: E402

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

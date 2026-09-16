"""Fingerprinting and loading for the frozen benchmark instance set.

`fingerprint` extends `tests/test_backcompat.py::_fp`'s construction (demos,
test, sorted held pairs) with a fourth section over `inst.val`, so a frozen
instance's identity also pins its validation split.
"""

import hashlib
import json
from pathlib import Path


def fingerprint(inst) -> str:
    h = hashlib.sha256()
    for e, a in inst.demos:
        h.update(f"D|{e}|{a}\n".encode())
    for t in inst.test:
        h.update(f"T|{t.split}|{t.expr_src}|{t.answer_src}\n".encode())
    for p in sorted(inst.held_pairs):
        h.update(f"H|{p[0]}|{p[1]}\n".encode())
    for v in inst.val:
        h.update(f"V|{v.expr_src}|{v.answer_src}\n".encode())
    return h.hexdigest()


DEFAULT_CUTOFFS = {"low": (0.20, 0.35), "mid": (0.40, 0.55), "high": (0.60, 0.80)}


def band_of(pi, cutoffs=DEFAULT_CUTOFFS) -> str | None:
    for name, (lo, hi) in cutoffs.items():
        if lo <= pi < hi:
            return name
    return None


def load_frozen(path="docs/benchmark/frozen_instances.json") -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open() as f:
        manifest = json.load(f)
    return manifest["instances"]

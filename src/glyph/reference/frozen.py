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


def frozen_entry(instance_id, *, manifest_path="docs/benchmark/frozen_instances.json") -> dict:
    """Return the manifest metadata dict for a frozen instance id (band/preset/seed/
    measured_pi/uses_binary_tables/fingerprint_sha256).

    Raises KeyError (listing the available ids) if `instance_id` is not in the manifest.
    """
    entries = load_frozen(manifest_path)
    for e in entries:
        if e["id"] == instance_id:
            return e
    raise KeyError(
        f"no frozen instance {instance_id!r}; available: {sorted(e['id'] for e in entries)}"
    )


def load_instance(instance_id, *, manifest_path="docs/benchmark/frozen_instances.json",
                   verify=True, entry=None):
    """Regenerate the frozen instance for `instance_id` from the manifest.

    By default (`verify=True`) asserts the regenerated instance's fingerprint still matches
    the frozen `fingerprint_sha256`, so callers get exactly the frozen data or a clear error
    on drift.

    `entry`: an already-fetched manifest entry (e.g. from a prior `frozen_entry(instance_id,
    ...)` call) to reuse instead of reloading/reparsing the manifest here -- for callers, like
    `resolve_run_instance`, that also need the entry's own fields. Must be the entry for
    `instance_id`; when omitted (the default), it's looked up the normal way.
    """
    if entry is None:
        entry = frozen_entry(instance_id, manifest_path=manifest_path)
    from glyph.data import PRESETS, generate

    inst = generate(entry["seed"], PRESETS[entry["preset"]])
    if verify:
        actual = fingerprint(inst)
        expected = entry["fingerprint_sha256"]
        if actual != expected:
            raise ValueError(
                f"fingerprint drift for frozen instance {instance_id!r}: "
                f"expected {expected}, got {actual}"
            )
    return inst

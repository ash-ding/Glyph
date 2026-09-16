"""Integrity test for the frozen benchmark instance manifest (#task 5).

Confirms the committed `docs/benchmark/frozen_instances.json` manifest is
reproducible from its seeds and internally consistent with its own band
assignment.
"""

import json

from glyph.data import PRESETS, generate
from glyph.reference.frozen import band_of, fingerprint, load_frozen

MANIFEST_PATH = "docs/benchmark/frozen_instances.json"


def test_manifest_shape():
    with open(MANIFEST_PATH) as f:
        raw = json.load(f)
    n_per_band = raw["n_per_band"]
    instances = raw["instances"]

    bands = {inst["band"] for inst in instances}
    assert bands == {"low", "mid", "high"}

    for band in ("low", "mid", "high"):
        count = sum(1 for inst in instances if inst["band"] == band)
        assert count == n_per_band

    assert len(instances) == 3 * n_per_band


def test_manifest_reproduces_and_in_band():
    instances = load_frozen(MANIFEST_PATH)
    assert instances

    for entry in instances:
        inst = generate(entry["seed"], PRESETS[entry["preset"]])

        assert fingerprint(inst) == entry["fingerprint_sha256"]
        assert abs(inst.measured_pi()["pi"] - entry["measured_pi"]) < 1e-9
        assert band_of(entry["measured_pi"]) == entry["band"]
        assert any(t.needs_b for t in inst.test) is True

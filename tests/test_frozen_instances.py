"""Integrity test for the frozen benchmark instance manifest (#task 5).

Confirms the committed `docs/benchmark/frozen_instances.json` manifest is
reproducible from its seeds and internally consistent with its own band
assignment, and that it matches the current selection policy: each band is
preset-pure and every instance's measured pi sits inside that band's
archetype window (see `tools/select_instances.DEFAULT_SELECTION`).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from glyph.data import PRESETS, generate
from glyph.reference.frozen import band_of, fingerprint, load_frozen
from select_instances import DEFAULT_SELECTION  # noqa: E402

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


def test_manifest_is_preset_pure_and_in_window():
    """Selection policy (preset-pure archetype window): every instance in a
    band must come from that band's required preset, and its measured pi
    must lie inside that band's narrow selection window (a strict subset of
    the band's regime cutoff checked by `test_manifest_reproduces_and_in_band`
    above)."""
    instances = load_frozen(MANIFEST_PATH)
    assert instances

    for entry in instances:
        spec = DEFAULT_SELECTION[entry["band"]]
        assert entry["preset"] == spec["preset"], (
            f"{entry['id']}: expected preset {spec['preset']!r} for band "
            f"{entry['band']!r}, got {entry['preset']!r}"
        )
        win_lo, win_hi = spec["window"]
        assert win_lo <= entry["measured_pi"] < win_hi, (
            f"{entry['id']}: measured_pi {entry['measured_pi']} outside "
            f"selection window [{win_lo}, {win_hi}) for band {entry['band']!r}"
        )

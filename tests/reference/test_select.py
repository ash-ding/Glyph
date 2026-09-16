import json
import sys

sys.path.insert(0, "tools")
from select_instances import select, write_manifest  # noqa: E402

from glyph.reference.frozen import DEFAULT_CUTOFFS, band_of


def _mk(seed, preset, pi, uses_binary_tables=True):
    return {
        "seed": seed,
        "preset": preset,
        "pi": pi,
        "uses_binary_tables": uses_binary_tables,
        "fingerprint": "fp%04d" % seed,
    }


def _synthetic_candidates():
    presets = ["pi_low", "pi_mid", "pi_high"]
    cands = []
    seed = 1001

    def add(pi, uses_binary_tables=True):
        nonlocal seed
        preset = presets[(seed - 1001) % len(presets)]
        cands.append(_mk(seed, preset, pi, uses_binary_tables))
        seed += 1

    # below the low band entirely (out of range)
    for pi in (0.02, 0.05, 0.08, 0.10, 0.15):
        add(pi)

    # low band [0.20, 0.35)
    for pi in (0.20, 0.22, 0.25, 0.28, 0.30, 0.33):
        add(pi)
    # a couple of low-band candidates that fail the binary-tables filter
    add(0.23, uses_binary_tables=False)
    add(0.31, uses_binary_tables=False)

    # gap between low and mid
    for pi in (0.37, 0.38):
        add(pi)

    # mid band [0.40, 0.55)
    for pi in (0.40, 0.43, 0.46, 0.49, 0.52):
        add(pi)
    add(0.44, uses_binary_tables=False)

    # gap between mid and high
    for pi in (0.57, 0.58):
        add(pi)

    # high band [0.60, 0.80)
    for pi in (0.60, 0.65, 0.70, 0.75, 0.78):
        add(pi)
    add(0.72, uses_binary_tables=False)

    # above the high band entirely (out of range)
    for pi in (0.85, 0.90, 0.95):
        add(pi)

    return cands


def test_select_counts_and_bands():
    cands = _synthetic_candidates()
    chosen = select(cands, n_per_band=2)

    assert len(chosen) == 6
    by_band = {}
    for inst in chosen:
        by_band.setdefault(inst["band"], []).append(inst)
    assert set(by_band.keys()) == {"low", "mid", "high"}
    for band, insts in by_band.items():
        assert len(insts) == 2
        lo, hi = DEFAULT_CUTOFFS[band]
        for inst in insts:
            assert lo <= inst["measured_pi"] < hi
            assert band_of(inst["measured_pi"]) == band
            assert inst["uses_binary_tables"] is True

    # none of the uses_binary_tables=False candidates' fingerprints leaked through
    excluded_fps = {"fp1012", "fp1013", "fp1021", "fp1029"}
    chosen_fps = {inst["fingerprint_sha256"] for inst in chosen}
    assert not (chosen_fps & excluded_fps)


def test_select_spread():
    cands = _synthetic_candidates()
    chosen = select(cands, n_per_band=2)

    by_band = {}
    for inst in chosen:
        by_band.setdefault(inst["band"], []).append(inst["measured_pi"])
    for band, pis in by_band.items():
        assert len(set(pis)) == len(pis), "band %s picks were not spread: %r" % (band, pis)


def test_write_manifest_shape(tmp_path):
    cands = _synthetic_candidates()
    chosen = select(cands, n_per_band=2)
    out = tmp_path / "m.json"

    write_manifest(chosen, DEFAULT_CUTOFFS, 2, out)

    with open(out) as f:
        manifest = json.load(f)

    assert isinstance(manifest, dict)
    assert "band_cutoffs" in manifest
    assert manifest["n_per_band"] == 2
    assert isinstance(manifest["instances"], list)
    assert len(manifest["instances"]) == 6

    expected_keys = {
        "id", "band", "preset", "seed", "measured_pi",
        "uses_binary_tables", "fingerprint_sha256",
    }
    for inst in manifest["instances"]:
        assert set(inst.keys()) == expected_keys

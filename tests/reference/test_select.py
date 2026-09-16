import json
import sys

sys.path.insert(0, "tools")
from select_instances import DEFAULT_SELECTION, select, write_manifest  # noqa: E402

from glyph.reference.frozen import DEFAULT_CUTOFFS, band_of

# A compact selection spec for the synthetic pool below: same shape as
# DEFAULT_SELECTION but with windows sized to the synthetic data so the
# tests stay self-contained and CPU-only.
_TEST_SELECTION = {
    "low": {"preset": "pi_low", "window": (0.20, 0.35)},
    "mid": {"preset": "pi_mid", "window": (0.40, 0.55)},
    "high": {"preset": "pi_high", "window": (0.60, 0.80)},
}


def _mk(seed, preset, pi, uses_binary_tables=True):
    return {
        "seed": seed,
        "preset": preset,
        "pi": pi,
        "uses_binary_tables": uses_binary_tables,
        "fingerprint": "fp%04d" % seed,
    }


def _synthetic_candidates():
    """A pool with enough per-band, per-preset pure candidates to pick from,
    plus decoys: wrong preset inside the window, out-of-window same-preset,
    and uses_binary_tables=False, all of which must NOT be selected."""
    cands = []
    seed = 1001

    def add(preset, pi, uses_binary_tables=True):
        nonlocal seed
        cands.append(_mk(seed, preset, pi, uses_binary_tables))
        seed += 1

    # low band window [0.20, 0.35): 6 pure pi_low candidates (more than n=2)
    for pi in (0.20, 0.23, 0.26, 0.29, 0.32, 0.34):
        add("pi_low", pi)
    # decoys in the low window: wrong preset, and a failed binary-tables pick
    add("pi_mid", 0.24)          # wrong preset -- must be excluded
    add("pi_low", 0.25, uses_binary_tables=False)  # fails filter -- excluded
    # out-of-window pi_low candidates (outside [0.20, 0.35)) -- excluded
    add("pi_low", 0.10)
    add("pi_low", 0.37)

    # mid band window [0.40, 0.55): 6 pure pi_mid candidates
    for pi in (0.40, 0.43, 0.46, 0.49, 0.52, 0.54):
        add("pi_mid", pi)
    add("pi_high", 0.45)          # wrong preset -- excluded
    add("pi_mid", 0.44, uses_binary_tables=False)  # excluded

    # high band window [0.60, 0.80): 6 pure pi_high candidates
    for pi in (0.60, 0.65, 0.68, 0.72, 0.76, 0.79):
        add("pi_high", pi)
    add("pi_low", 0.70)           # wrong preset -- excluded
    add("pi_high", 0.62, uses_binary_tables=False)  # excluded

    return cands


def test_select_preset_pure_and_in_window():
    cands = _synthetic_candidates()
    chosen = select(cands, n_per_band=2, selection=_TEST_SELECTION)

    assert len(chosen) == 6
    by_band = {}
    for inst in chosen:
        by_band.setdefault(inst["band"], []).append(inst)
    assert set(by_band.keys()) == {"low", "mid", "high"}

    for band, insts in by_band.items():
        assert len(insts) == 2
        spec = _TEST_SELECTION[band]
        win_lo, win_hi = spec["window"]
        for inst in insts:
            # (a) preset purity
            assert inst["preset"] == spec["preset"]
            # (b) inside the band's archetype window
            assert win_lo <= inst["measured_pi"] < win_hi
            # still consistent with the regime cutoffs
            assert band_of(inst["measured_pi"], DEFAULT_CUTOFFS) == band
            assert inst["uses_binary_tables"] is True

    # decoys never leak through: wrong-preset and uses_binary_tables=False
    # candidates, derived directly from the synthetic pool so the check
    # doesn't depend on hand-counted seeds.
    excluded_fps = {
        c["fingerprint"] for c in cands
        if not c["uses_binary_tables"]
        or all(c["preset"] != spec["preset"] for spec in _TEST_SELECTION.values())
        or (
            # wrong preset for the window it numerically falls in
            any(
                spec["window"][0] <= c["pi"] < spec["window"][1] and c["preset"] != spec["preset"]
                for spec in _TEST_SELECTION.values()
            )
        )
    }
    chosen_fps = {inst["fingerprint_sha256"] for inst in chosen}
    assert not (chosen_fps & excluded_fps)
    # sanity: the derived exclusion set is non-empty, i.e. the decoys exist
    assert len(excluded_fps) >= 6


def test_select_five_per_band():
    cands = _synthetic_candidates()
    chosen = select(cands, n_per_band=5, selection=_TEST_SELECTION)

    by_band = {}
    for inst in chosen:
        by_band.setdefault(inst["band"], []).append(inst)
    for band, insts in by_band.items():
        assert len(insts) == 5


def test_select_spread():
    cands = _synthetic_candidates()
    chosen = select(cands, n_per_band=2, selection=_TEST_SELECTION)

    by_band = {}
    for inst in chosen:
        by_band.setdefault(inst["band"], []).append(inst["measured_pi"])
    for band, pis in by_band.items():
        assert len(set(pis)) == len(pis), "band %s picks were not spread: %r" % (band, pis)


def test_select_no_candidate_reused():
    cands = _synthetic_candidates()
    chosen = select(cands, n_per_band=5, selection=_TEST_SELECTION)

    seeds = [inst["seed"] for inst in chosen]
    assert len(seeds) == len(set(seeds)), "a candidate was reused across picks"


def test_select_is_deterministic():
    cands = _synthetic_candidates()
    chosen_a = select(cands, n_per_band=5, selection=_TEST_SELECTION)
    chosen_b = select(cands, n_per_band=5, selection=_TEST_SELECTION)

    assert [c["id"] for c in chosen_a] == [c["id"] for c in chosen_b]
    assert [c["seed"] for c in chosen_a] == [c["seed"] for c in chosen_b]

    # also stable across candidate list re-ordering (input order must not matter)
    shuffled = list(reversed(cands))
    chosen_c = select(shuffled, n_per_band=5, selection=_TEST_SELECTION)
    assert [c["seed"] for c in chosen_a] == [c["seed"] for c in chosen_c]


def test_select_uses_default_selection_spec():
    # DEFAULT_SELECTION should be the documented preset-pure archetype windows.
    assert DEFAULT_SELECTION["low"] == {"preset": "pi_low", "window": (0.20, 0.30)}
    assert DEFAULT_SELECTION["mid"] == {"preset": "pi_mid", "window": (0.45, 0.53)}
    assert DEFAULT_SELECTION["high"] == {"preset": "pi_high", "window": (0.70, 0.80)}
    # every default window is a strict subset of its band's regime cutoff
    for band, spec in DEFAULT_SELECTION.items():
        cutoff_lo, cutoff_hi = DEFAULT_CUTOFFS[band]
        win_lo, win_hi = spec["window"]
        assert cutoff_lo <= win_lo and win_hi <= cutoff_hi


def test_write_manifest_shape(tmp_path):
    cands = _synthetic_candidates()
    chosen = select(cands, n_per_band=2, selection=_TEST_SELECTION)
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

"""Tests for reuse-by-id: `frozen_entry`/`load_instance` (Task R1-A) and the harness's
`resolve_run_instance` resolver (Task R1-B).

CPU-only, offline: `generate` and the resolver never touch an agent/API.
"""
from __future__ import annotations

import pytest

from glyph.reference import frozen as frozen_mod
from glyph.reference.frozen import fingerprint, frozen_entry, load_frozen, load_instance
from glyph.v2.harness import RunConfig, resolve_run_instance


@pytest.mark.parametrize("instance_id", ["low_1", "mid_1", "high_1"])
def test_load_instance_matches_frozen_fingerprint(instance_id):
    inst = load_instance(instance_id)
    entry = frozen_entry(instance_id)
    assert fingerprint(inst) == entry["fingerprint_sha256"]


def test_frozen_entry_unknown_id_raises_keyerror_listing_ids():
    with pytest.raises(KeyError) as exc_info:
        frozen_entry("does_not_exist")
    msg = str(exc_info.value)
    assert "does_not_exist" in msg
    all_ids = sorted(e["id"] for e in load_frozen())
    for i in all_ids:
        assert i in msg


def test_load_instance_unknown_id_raises_keyerror():
    with pytest.raises(KeyError):
        load_instance("does_not_exist")


def test_load_instance_verify_catches_fingerprint_drift(monkeypatch):
    monkeypatch.setattr(frozen_mod, "fingerprint", lambda inst: "0" * 64)
    with pytest.raises(ValueError, match="fingerprint drift"):
        load_instance("low_1")


def test_load_instance_verify_false_skips_check(monkeypatch):
    monkeypatch.setattr(frozen_mod, "fingerprint", lambda inst: "0" * 64)
    # should not raise even though the (patched) fingerprint would not match
    inst = load_instance("low_1", verify=False)
    assert inst is not None


def test_resolve_run_instance_with_instance_id_overrides_preset_seed():
    rc = RunConfig(arm="no_train", instance_id="high_3")
    inst, preset, seed, instance_id = resolve_run_instance(rc)
    entry = frozen_entry("high_3")
    assert preset == entry["preset"]
    assert seed == entry["seed"]
    assert instance_id == "high_3"
    assert fingerprint(inst) == entry["fingerprint_sha256"]


def test_resolve_run_instance_without_instance_id_uses_raw_preset_seed():
    # n_val=None so RunConfig's own n_val default (5000) doesn't override the
    # "smoke" preset's n_val (40) -- that override is pre-existing run()
    # behavior (unchanged by this resolver), orthogonal to what's under test here.
    rc = RunConfig(arm="no_train", preset="smoke", instance_seed=1001, n_val=None)
    inst, preset, seed, instance_id = resolve_run_instance(rc)
    assert preset == "smoke"
    assert seed == 1001
    assert instance_id is None

    from glyph.data import PRESETS, generate
    expected = generate(1001, PRESETS["smoke"])
    assert fingerprint(inst) == fingerprint(expected)

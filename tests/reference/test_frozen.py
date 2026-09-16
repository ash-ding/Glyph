import hashlib

from glyph.data import PRESETS, generate
from glyph.reference.frozen import band_of, fingerprint, load_frozen


def test_fingerprint_stable():
    inst = generate(1001, PRESETS["smoke"])
    fp1 = fingerprint(inst)
    fp2 = fingerprint(inst)
    assert fp1 == fp2
    assert len(fp1) == 64
    assert all(c in "0123456789abcdef" for c in fp1)


def test_fingerprint_prefix_matches_backcompat():
    inst = generate(1001, PRESETS["smoke"])
    h = hashlib.sha256()
    for e, a in inst.demos:
        h.update(f"D|{e}|{a}\n".encode())
    for t in inst.test:
        h.update(f"T|{t.split}|{t.expr_src}|{t.answer_src}\n".encode())
    for p in sorted(inst.held_pairs):
        h.update(f"H|{p[0]}|{p[1]}\n".encode())
    assert h.hexdigest() == "ac4e73ba0b9822d5d2fab38ef3a2f565c06f8d5f29dab23f6f6584e777ffc1d6"


def test_band_of():
    assert band_of(0.27) == "low"
    assert band_of(0.50) == "mid"
    assert band_of(0.70) == "high"
    assert band_of(0.37) is None
    assert band_of(0.95) is None


def test_load_frozen_missing_returns_empty():
    assert load_frozen("/tmp/does_not_exist_xyz.json") == []

import hashlib
from glyph.data import PRESETS, generate

def _fp(inst):
    h = hashlib.sha256()
    for e, a in inst.demos: h.update(f"D|{e}|{a}\n".encode())
    for t in inst.test: h.update(f"T|{t.split}|{t.expr_src}|{t.answer_src}\n".encode())
    for p in sorted(inst.held_pairs): h.update(f"H|{p[0]}|{p[1]}\n".encode())
    return h.hexdigest()

def test_smoke_1001_unchanged():
    assert _fp(generate(1001, PRESETS["smoke"])) == \
        "ac4e73ba0b9822d5d2fab38ef3a2f565c06f8d5f29dab23f6f6584e777ffc1d6"

def test_pi_mid_1001_unchanged():
    assert _fp(generate(1001, PRESETS["pi_mid"])) == \
        "efaf4694b0ece951ebaa24f3ebec0f8cae6e048fc92e6562ad89bf53be80255f"

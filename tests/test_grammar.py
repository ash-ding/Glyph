"""Self-check #1: the interpreter's front end round-trips."""
import numpy as np
import pytest

from glyph.data.config import PRESETS, GlyphConfig
from glyph.data.grammar import (check, digits, parse, parse_value, render,
                           render_value, syntax_spec, undigits)
from glyph.data.instance import _sample

CFGS = [PRESETS["smoke"], PRESETS["pi_mid"], PRESETS["pi_high"]]


@pytest.mark.parametrize("cfg", CFGS)
def test_value_codec_roundtrip(cfg):
    rng = np.random.default_rng(0)
    for _ in range(200):
        i = int(rng.integers(cfg.n_values))
        assert undigits(digits(i, cfg), cfg) == i
        assert parse_value(render_value(i, cfg), cfg) == i


@pytest.mark.parametrize("form", ["underscore", "bracket", "flat", "letter_sep"])
def test_every_value_form_roundtrips(form):
    cfg = PRESETS["smoke"].with_(value_form=form)
    for i in range(cfg.n_values):
        assert parse_value(render_value(i, cfg), cfg) == i


@pytest.mark.parametrize("cfg", CFGS)
def test_expression_roundtrip(cfg):
    """render -> parse -> render must be a fixed point."""
    rng = np.random.default_rng(7)
    for _ in range(300):
        want = "VAL" if rng.random() < 0.5 else "LIST"
        e = _sample(rng, cfg, want, cfg.max_expr_depth)
        src = render(e, cfg)
        back = parse(src, cfg)
        assert back == e
        assert render(back, cfg) == src


@pytest.mark.parametrize("cfg", CFGS)
def test_check_accepts_sampled(cfg):
    rng = np.random.default_rng(11)
    for _ in range(200):
        check(_sample(rng, cfg, "LIST", cfg.demo_max_depth), cfg)


def test_check_rejects_malformed():
    cfg = PRESETS["smoke"]
    bad = ["s0(u0)", "s0(nope, [v_0_0, v_1_1])", "s99([v_0_0, v_1_1])",
           "s3(99, [v_0_0, v_1_1])", "s0(u0, u1)"]
    for src in bad:
        with pytest.raises(Exception):
            check(parse(src, cfg), cfg)


def test_syntax_spec_leaks_no_semantics():
    """The public spec must not name a familiar operation -- naming priors
    would hand the skeleton to a frontier model for free."""
    spec = syntax_spec(PRESETS["pi_mid"]).lower()
    for word in ("map", "fold", "reverse", "rotate", "dedup", "filter",
                 "sort", "guard", "even"):
        assert word not in spec, f"{word!r} leaks into the public syntax spec"

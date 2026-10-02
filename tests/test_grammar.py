"""Self-check #1: the interpreter's front end round-trips."""
import numpy as np
import pytest

from glyph.data.config import PRESETS, GlyphConfig
from glyph.data.grammar import (STRUCT_SHAPES, check, digits, enabled_ops, parse,
                           parse_value, render, render_value, syntax_spec, undigits)
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


@pytest.mark.parametrize("n", [0, len(STRUCT_SHAPES) + 1])
def test_n_structural_out_of_range_raises(n):
    with pytest.raises(ValueError, match="n_structural"):
        enabled_ops(PRESETS["pi_mid"].with_(n_structural=n))


def test_every_block_of_four_shapes_holds_each_shape_once():
    shapes = [s for _, s in STRUCT_SHAPES]
    assert len(shapes) % 4 == 0
    for i in range(0, len(shapes), 4):
        assert sorted(shapes[i:i + 4]) == ["KL", "L", "LB", "UL"]


def test_syntax_spec_leaks_no_semantics():
    """The public spec must not name a familiar operation -- naming priors
    would hand the skeleton to a frontier model for free."""
    spec = syntax_spec(PRESETS["pi_mid"]).lower()
    for word in ("map", "fold", "reverse", "rotate", "dedup", "filter",
                 "sort", "guard", "even"):
        assert word not in spec, f"{word!r} leaks into the public syntax spec"


@pytest.mark.parametrize("cfg", CFGS)
def test_atomapp_roundtrip(cfg):
    """render -> parse -> render is a fixed point for bare atomic applications."""
    from glyph.data.grammar import AtomApp, depth, result_type
    rng = np.random.default_rng(23)
    for _ in range(100):
        i, j = int(rng.integers(cfg.n_values)), int(rng.integers(cfg.n_values))
        u = AtomApp(f"u{int(rng.integers(cfg.n_unary))}", (i,))
        b = AtomApp(f"b{int(rng.integers(cfg.n_binary))}", (i, j))
        for e in (u, b):
            src = render(e, cfg)
            back = parse(src, cfg)
            assert back == e
            assert render(back, cfg) == src
            assert result_type(back) == "VAL"
            assert depth(back) == 0   # no structural nesting: the depth-0 stratum


def test_atomapp_parse_from_source():
    """The written form parses to the node, not to a Val fallback."""
    from glyph.data.grammar import AtomApp
    cfg = PRESETS["smoke"]
    v0, v1 = render_value(0, cfg), render_value(1, cfg)
    assert parse(f"u0({v0})", cfg) == AtomApp("u0", (0,))
    assert parse(f"b0({v0}, {v1})", cfg) == AtomApp("b0", (0, 1))


def test_atomapp_check():
    from glyph.data.grammar import AtomApp
    cfg = PRESETS["smoke"]
    v0, v1 = render_value(0, cfg), render_value(1, cfg)
    # legal
    check(parse(f"u0({v0})", cfg), cfg)
    check(parse(f"b0({v0}, {v1})", cfg), cfg)
    # wrong arity, disabled op, value out of range
    for e in (AtomApp("u0", (0, 1)), AtomApp("b0", (0,)),
              AtomApp(f"u{cfg.n_unary}", (0,)), AtomApp(f"b{cfg.n_binary}", (0, 1)),
              AtomApp("u0", (cfg.n_values,))):
        with pytest.raises(SyntaxError):
            check(e, cfg)
    # nesting and non-literal args never reach check: the parser rejects them
    for src in (f"u0(u1({v0}))", f"u0([{v0}, {v1}])", f"u0(s2([{v0}, {v1}]))", "u0()"):
        with pytest.raises(SyntaxError):
            parse(src, cfg)


def test_syntax_spec_states_depth_zero():
    spec = syntax_spec(PRESETS["smoke"])
    assert "nesting depth 0" in spec

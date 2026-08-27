"""Self-check #2: an oracle holding the true skeleton and the true table
scores 100%.  Anything less is an interpreter bug, and it must be fixed
before any arm is written."""
import pytest

from glyph.config import PRESETS
from glyph.grammar import parse, render_list, render_value
from glyph.instance import generate
from glyph.interp import Interpreter
from glyph.semantics import trivial_skeleton
from glyph.tables import IdentityTables

FAST = {k: v.with_(n_iid=200, n_comp=70, n_depth=30) for k, v in PRESETS.items()}


def _score(interp, inst):
    ok = 0
    for t in inst.test:
        out = interp.eval(parse(t.expr_src, inst.cfg))
        src = (render_value(out, inst.cfg) if isinstance(out, int)
               else render_list(out, inst.cfg))
        ok += src == t.answer_src
    return ok / len(inst.test)


@pytest.mark.parametrize("preset", ["smoke", "pi_low", "pi_mid", "pi_high"])
def test_oracle_is_exact(preset):
    inst = generate(2024, FAST[preset])
    assert _score(inst.P, inst) == 1.0


def test_tables_are_deterministic():
    inst = generate(3, FAST["smoke"])
    for i in (0, 5, 17):
        assert inst.tables.apply_unary("u0", i) == inst.tables.apply_unary("u0", i)
        assert inst.tables.apply_binary("b0", i, 2) == inst.tables.apply_binary("b0", i, 2)


def test_digit_neighbours_are_correlated():
    """The tables must be *structured*, not random: perturbing one digit has
    to leave the output far more often unchanged than chance would."""
    import numpy as np
    from glyph.grammar import digits
    inst = generate(5, FAST["pi_mid"])
    cfg, rng = inst.cfg, np.random.default_rng(0)
    hits = 0
    for _ in range(300):
        i = int(rng.integers(cfg.n_values))
        d = list(digits(i, cfg))
        d[0] = (d[0] + 1) % cfg.base
        j = sum(x * cfg.base ** k for k, x in enumerate(d))
        hits += inst.tables.apply_unary("u0", i) == inst.tables.apply_unary("u0", j)
    assert hits / 300 > 20 * (1 / cfg.n_values)


def test_trivial_skeleton_still_uses_the_table():
    """The degenerate skeleton baseline must not silently ablate the table
    as well -- if s0 stopped mapping, L_skel would swallow L_table and pi
    would be biased with no visible symptom."""
    inst = generate(9, FAST["pi_mid"])
    B = Interpreter(inst.cfg, trivial_skeleton(inst.cfg), inst.tables)
    touched = 0
    for t in inst.test[:200]:
        _, log = B.eval_logged(parse(t.expr_src, inst.cfg))
        touched += len(log)
    assert touched > 0


def test_identity_table_still_exercises_the_skeleton():
    inst = generate(9, FAST["pi_mid"])
    A = Interpreter(inst.cfg, inst.skeleton, IdentityTables())
    diffs = sum(A.eval(parse(t.expr_src, inst.cfg)) !=
                Interpreter(inst.cfg, trivial_skeleton(inst.cfg),
                            IdentityTables()).eval(parse(t.expr_src, inst.cfg))
                for t in inst.test[:200])
    assert diffs > 0

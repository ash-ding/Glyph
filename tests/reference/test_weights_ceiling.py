"""CPU-only tests for the weights ceiling (#19): true skeleton over a
student's tables. No torch/transformers involved -- `train_student` and
`generate_answers` (the only functions that import them) are never called
here; the "perfect student" path exercises `_student_tables_from` and the
true-skeleton scoring `score_ceiling` uses, with callables built straight
from `inst.tables`.
"""

from glyph.data import PRESETS, generate
from glyph.reference.subset import paired_subset
from glyph.reference.weights_ceiling import (_score_on, _student_tables_from,
                                             seen_u)


def test_seen_frac_hash_stable():
    frac = 0.1
    # deterministic
    for k in (0, 1, 17, 4912):
        assert seen_u(k, frac) == seen_u(k, frac)

    n = 4913
    seen_count = sum(1 for k in range(n) if seen_u(k, frac))
    rate = seen_count / n
    assert abs(rate - frac) <= 0.03


def test_score_with_perfect_student_is_perfect():
    inst = generate(1001, PRESETS["smoke"])
    items = paired_subset(inst, 40)

    need_u, need_b = set(), set()
    for t in items:
        need_u |= set(t.needs_u)
        need_b |= set(t.needs_b)

    student_tables = _student_tables_from(
        inst.tables.apply_unary, inst.tables.apply_binary,
        need_u, need_b, inst.cfg)

    result = _score_on(inst, student_tables, items)

    assert abs(result["overall"] - 1.0) < 1e-9
    assert result["tail"] == 1.0 or result["tail"] is None


def test_oracle_prompts_are_public_syntax():
    """The oracle must train/answer in the same surface form the agent uses
    (CLAUDE.md rule 7). The prompt minus ' =' must parse under the public
    grammar as a bare atomic application."""
    from glyph.data import parse
    from glyph.data.grammar import AtomApp
    from glyph.reference.weights_ceiling import prompt_binary, prompt_unary
    cfg = PRESETS["smoke"]
    pu = prompt_unary("u0", 3, cfg)
    pb = prompt_binary("b0", 3, 4, cfg)
    for p, want in ((pu, AtomApp("u0", (3,))), (pb, AtomApp("b0", (3, 4)))):
        assert p.endswith(" =")
        assert parse(p[:-2], cfg) == want


# ---------------------------------------------------------------------
# per-op oracle v2: eligibility, holdout split, exposure-exact probe scoring
# ---------------------------------------------------------------------

def test_eligible_cells_n_mode_exact_and_deterministic():
    import pytest
    from glyph.reference.weights_ceiling import eligible_cells
    inst = generate(1001, PRESETS["smoke"])
    n_vals = inst.cfg.n_values
    for op, width in (("u0", 1), ("b0", 2)):
        a = eligible_cells(inst, op, n_seen=12)
        b = eligible_cells(inst, op, n_seen=12)
        assert a == b and len(a) == 12 == len(set(a))
        assert all(len(c) == width and all(0 <= x < n_vals for x in c) for c in a)
    assert eligible_cells(inst, "u0", n_seen=12) != eligible_cells(inst, "u1", n_seen=12)
    with pytest.raises(ValueError):
        eligible_cells(inst, "u9", n_seen=5)
    with pytest.raises(ValueError):
        eligible_cells(inst, "u0")                        # neither knob
    with pytest.raises(ValueError):
        eligible_cells(inst, "u0", seen_frac=0.1, n_seen=5)  # both knobs


def test_eligible_cells_frac_mode_matches_frozen_hash():
    from glyph.reference.weights_ceiling import eligible_cells, seen_b
    inst = generate(1001, PRESETS["smoke"])
    n_vals = inst.cfg.n_values
    assert eligible_cells(inst, "u0", seen_frac=0.3) ==         [(i,) for i in range(n_vals) if seen_u(i, 0.3)]
    assert eligible_cells(inst, "b1", seen_frac=0.3) ==         [(i, j) for i in range(n_vals) for j in range(n_vals) if seen_b(i, j, 0.3)]


def test_split_holdout_rules():
    from glyph.reference.weights_ceiling import split_holdout
    cells = [(i,) for i in range(100)]
    tr, ho = split_holdout(cells, seed=7)
    tr2, ho2 = split_holdout(cells, seed=7)
    assert (tr, ho) == (tr2, ho2)                      # deterministic
    assert not (set(tr) & set(ho))
    assert sorted(tr + ho) == cells
    assert len(ho) == 10                               # round(0.1 * 100)
    assert len(split_holdout([(i,) for i in range(20)], seed=7)[1]) == 8   # floor 8
    assert len(split_holdout([(i,) for i in range(50000)], seed=7)[1]) == 1024  # cap
    assert len(split_holdout([(i,) for i in range(5)], seed=7)[1]) == 4   # at most n-1


def test_score_probes_with_explicit_seen_sets():
    from glyph.data.probe import probe_set
    from glyph.reference.weights_ceiling import score_probes
    inst = generate(1001, PRESETS["smoke"])
    probes = [t for t in probe_set(inst, n_per_op=10) if t.split == "u0"]
    seen_cells = {next(iter(t.needs_u)) for t in probes[:3]}   # 3 full-keyed cells
    out = score_probes(
        inst, probes, seen_cells,
        answer_unary=lambda name, i: inst.tables.apply_unary(name, i),
        answer_binary=lambda name, i, j: inst.tables.apply_binary(name, i, j))
    assert out["overall"] == 1.0 and out["n"] == len(probes)
    assert out["seen"]["n"] == 3 and out["unseen"]["n"] == len(probes) - 3

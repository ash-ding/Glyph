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


def test_score_probes_perfect_student_and_seen_split():
    """A perfect student scores 1.0 on its op's probes; seen/unseen n's are
    decided by the training hash, not by any query log."""
    from glyph.data.probe import probe_set
    from glyph.reference.weights_ceiling import score_probes
    inst = generate(1001, PRESETS["smoke"])
    frac = 0.5
    probes = [t for t in probe_set(inst, n_per_op=10) if t.split == "u0"]
    out = score_probes(
        inst, probes, frac,
        answer_unary=lambda name, i: inst.tables.apply_unary(name, i),
        answer_binary=lambda name, i, j: inst.tables.apply_binary(name, i, j))
    assert out["overall"] == 1.0 and out["n"] == len(probes)
    n_seen = sum(1 for t in probes for (_, i) in t.needs_u if seen_u(i, frac))
    assert out["seen"]["n"] == n_seen
    assert out["seen"]["n"] + out["unseen"]["n"] == out["n"]


def test_train_student_ops_restriction_signature():
    """CPU-only: the ops filter must reject unknown op names before any GPU
    work starts (fail fast on a typo, not 6000 steps in)."""
    import pytest
    from glyph.reference.weights_ceiling import train_student
    inst = generate(1001, PRESETS["smoke"])
    with pytest.raises(ValueError):
        train_student(inst, 0.1, ops=["u9"])

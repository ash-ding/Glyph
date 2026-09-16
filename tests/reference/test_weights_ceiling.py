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

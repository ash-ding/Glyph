"""CPU-only tests for A0' (#20): the retrieval upper bound.

`run_a0prime`'s `answer_fn` seam is exercised here with an oracle that
returns the TRUE answer for every expression, which validates the parse+score
wiring end to end with no real frontier call ever made.
"""

from glyph.data import PRESETS, generate
from glyph.reference.a0prime import buy_evidence, retrieval_split, run_a0prime
from glyph.reference.subset import paired_subset


def test_buy_reveals_entries():
    inst = generate(1001, PRESETS["smoke"])
    facts, revealed = buy_evidence(inst, 20, seed=1)

    assert facts
    for src, answer in facts:
        assert isinstance(src, str) and src
        assert isinstance(answer, str) and answer

    u, b = revealed
    assert u
    assert u == frozenset(inst.query_log.unary)
    assert b == frozenset(inst.query_log.binary)


def test_retrieval_split_partitions():
    inst = generate(1001, PRESETS["smoke"])
    items = paired_subset(inst, 30)
    _, revealed = buy_evidence(inst, 20, seed=1)

    d = retrieval_split(items, revealed)

    assert d["covered_n"] + d["uncovered_n"] == len(items)
    assert d["covered_n"] == len(d["covered"])
    assert d["uncovered_n"] == len(d["uncovered"])

    u, b = revealed
    for t in d["covered"]:
        assert t.needs_u <= u and t.needs_b <= b
    for t in d["uncovered"]:
        assert not (t.needs_u <= u and t.needs_b <= b)


def test_run_a0prime_with_oracle_answer_fn_scores_high():
    inst = generate(1001, PRESETS["smoke"])
    items = paired_subset(inst, 30)
    by_src = {t.expr_src: t.answer_src for t in items}

    def oracle_fn(evidence, expr_srcs):
        return [by_src[s] for s in expr_srcs]

    result = run_a0prime(inst, items, evidence="", answer_fn=oracle_fn)

    assert result["overall"] == 1.0
    assert set(result.keys()) == {"overall", "by_split", "tail", "headroom"}

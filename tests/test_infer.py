"""The prefix cache has to be a speed-up, not a different computation.

A2's cost story depends on prefilling the shared prompt once, so the cache is
mandatory rather than optional -- which makes "does it change the answers?" a
question about the arm's validity, not about performance.

Checked in fp32: bf16's accumulation order differs between a prefix computed
on its own and the same prefix computed inside a longer sequence, and on an
item near a decision boundary that is enough to flip a token.  That is
numerical noise every prefix-caching implementation has, vLLM included, and
it is not what this test is for.
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

pytestmark = pytest.mark.slow

BASE = "Qwen/Qwen3-1.7B"
CONTEXT = ("Here are some worked examples of a small language.\n"
           "s0(u1, [v_1_2_3]) = [v_4_5_6]\n"
           "s2([v_1_1_1, v_2_2_2]) = [v_2_2_2, v_1_1_1]\n"
           "Answer the next one in the same form.\n\n")
EXPRS = ["s0(u1, [v_3_3_3])", "s2([v_0_1_2, v_5_5_5])",
         "s1([v_1_1_1], b0)", "s0(u2, [v_7_7_7])"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_prefilled_prefix_matches_inline_context():
    from glyph.train import Student

    cached = Student(BASE, context=CONTEXT, max_new_tokens=16, dtype=torch.float32)
    inline = Student(BASE, max_new_tokens=16, dtype=torch.float32)

    got = cached.answer(EXPRS, batch_size=4)
    want = inline.answer(EXPRS, batch_size=4,
                         prompt_of=lambda e: f"{CONTEXT}{e} =")

    assert got.prefilled_once and got.prefix_tokens > 0
    assert want.prefix_tokens == 0
    assert got.answers == want.answers


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_batching_does_not_change_answers():
    """Left padding plus a shared cache is where position ids go wrong."""
    from glyph.train import Student

    s = Student(BASE, context=CONTEXT, max_new_tokens=16, dtype=torch.float32)
    assert s.answer(EXPRS, batch_size=1).answers == s.answer(EXPRS, batch_size=4).answers

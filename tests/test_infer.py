"""What the student's generation path has to guarantee.

The hand-rolled prefix cache these tests were originally written for is gone;
vLLM owns the cache now, so there is no equivalence of ours left to prove.
What still has to hold is that A2's prefix is actually cached (its whole cost
story depends on it), that the answers come back in the order they were
asked, and that decoding is deterministic -- test-time sampling is not one of
the variables under study, and noise there lands straight in the crossover.
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

pytestmark = pytest.mark.slow

BASE = "Qwen/Qwen3-1.7B"
CONTEXT = ("Worked examples of a small language:\n"
           "s0(u1, [v_1_2_3]) = [v_4_5_6]\n"
           "s2([v_1_1_1, v_2_2_2]) = [v_2_2_2, v_1_1_1]\n\n")
EXPRS = ["s0(u1, [v_3_3_3])", "s2([v_0_1_2, v_5_5_5])",
         "s1([v_1_1_1], b0)", "s0(u2, [v_7_7_7])"]


@pytest.fixture(scope="module")
def student():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    from glyph.train import Student
    s = Student(BASE, context=CONTEXT, max_new_tokens=16,
                gpu_memory_utilization=0.55)
    yield s
    s.close()


def test_the_context_prefix_is_cached_and_counted(student):
    got = student.answer(EXPRS)
    assert got.prefix_caching, "A2's cost story depends on this being on"
    assert got.prefix_tokens > 0
    assert len(got.answers) == len(EXPRS)


def test_answers_come_back_in_the_order_they_were_asked(student):
    """vLLM reorders internally; a silent shuffle would misgrade every item."""
    repeated = EXPRS * 4
    got = student.answer(repeated)
    assert len(got.answers) == len(repeated)
    first = got.answers[:len(EXPRS)]
    for k in range(1, 4):
        assert got.answers[k * len(EXPRS):(k + 1) * len(EXPRS)] == first


def test_decoding_is_deterministic(student):
    assert student.answer(EXPRS).answers == student.answer(EXPRS).answers

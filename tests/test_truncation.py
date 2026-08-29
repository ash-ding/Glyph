"""A truncated answer must never be mistaken for a wrong one.

The generation cap silently cost A2 and A6 four tenths of the test set: a
list cut off mid-way fell through to the value pattern and came back looking
like a wrong single value. It scores wrong either way -- but invisibly, so
the arm looks incapable when the budget was short.
"""
from glyph.train.infer import _clean


def test_a_complete_list_survives():
    assert _clean(" [v_1_2_3, v_4_5_6]") == "[v_1_2_3, v_4_5_6]"


def test_a_bare_value_survives():
    assert _clean(" v_1_2_3") == "v_1_2_3"


def test_prose_around_a_value_is_stripped():
    assert _clean("the answer is v_9_9_9.") == "v_9_9_9"


def test_a_truncated_list_does_not_become_its_first_element():
    got = _clean(" [v_1_2_3, v_4_5")
    assert got != "v_1_2_3", "truncation must not look like a wrong answer"
    assert got.startswith("["), "it should still look like the broken list it is"


def test_the_default_cap_clears_the_longest_real_answer():
    """Measured: the longest answer across presets is 42 tokens."""
    import inspect

    from glyph.train.infer import Student
    default = inspect.signature(Student.__init__).parameters["max_new_tokens"].default
    assert default >= 64, f"{default} is too close to the 42-token maximum"

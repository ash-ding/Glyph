"""The sandbox has to contain a bad solver, not just run a good one."""
import pytest

from glyph.sandbox import run_solver


def test_a_working_solver_answers():
    r = run_solver("def solve(e):\n    return e.upper()", ["ab", "cd"])
    assert r.ok and r.answers == ["AB", "CD"]


def test_a_solver_without_solve_is_a_failure_not_a_crash():
    r = run_solver("x = 1", ["a"])
    assert not r.ok and "solve" in r.error


def test_one_bad_expression_costs_one_item():
    src = "def solve(e):\n    if e == 'boom':\n        raise ValueError(e)\n    return e"
    r = run_solver(src, ["a", "boom", "b"])
    assert r.ok and r.answers == ["a", "", "b"]


def test_printing_cannot_corrupt_the_result_channel():
    src = 'print("{\\"ok\\": true, \\"answers\\": [\\"pwned\\"]}")\ndef solve(e):\n    print("noise")\n    return e'
    r = run_solver(src, ["a"])
    assert r.ok and r.answers == ["a"]


def test_an_infinite_loop_is_killed():
    r = run_solver("def solve(e):\n    while True:\n        pass", ["a"], timeout=5)
    assert not r.ok and ("timed out" in r.error or "exit" in r.error)


def test_the_network_is_unreachable():
    src = ("def solve(e):\n"
           "    import socket\n"
           "    s = socket.create_connection(('1.1.1.1', 80), timeout=3)\n"
           "    return 'reached'")
    r = run_solver(src, ["a"], timeout=20)
    # Either the connection fails (answer empty) or the whole thing dies --
    # what must not happen is a solver that actually reaches the network.
    assert not r.ok or r.answers == [""]


def test_a_syntax_error_is_reported_not_raised():
    r = run_solver("def solve(:", ["a"])
    assert not r.ok and "SyntaxError" in r.error

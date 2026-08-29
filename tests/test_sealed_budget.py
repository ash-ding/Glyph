"""The prepare budget must not be able to stop a run from being scored."""
import pytest

from glyph.budget import BudgetExhausted, Ledger


def test_the_prepare_budget_still_bites_before_sealing():
    led = Ledger(total_h100s=1.0)
    with pytest.raises(BudgetExhausted):
        for _ in range(2000):
            led.charge("gpu_second", 1.0)


def test_the_test_phase_records_without_being_capped():
    """A run that spent its budget should still produce a score.

    Otherwise an agent that overspends leaves a hole in the paired grid
    rather than a bad-but-comparable data point -- and worse, the arm most
    likely to overspend is the one whose deployment costs most, which is
    exactly the arm the comparison exists to measure.
    """
    led = Ledger(total_h100s=10.0)
    led.charge("gpu_second", 9.0)
    with led.sealed_mode():
        led.charge("gpu_second", 500.0)        # would have raised before
    assert led.spent_h100s > led.total
    assert any(r.n == 500.0 for r in led.records), "still recorded"


def test_enforcement_comes_back_after_the_sealed_block():
    led = Ledger(total_h100s=1.0)
    with led.sealed_mode():
        led.charge("gpu_second", 100.0)
    with pytest.raises(BudgetExhausted):
        led.charge("gpu_second", 1.0)

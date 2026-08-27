"""The ledger is the only way to advance the clock."""
import pytest

from glyph.budget import BudgetExhausted, CostModel, Ledger, meters


def test_charges_accumulate():
    led = Ledger()
    led.charge("oracle_query", 1000)
    led.charge("frontier_in", 50_000)
    led.charge("frontier_out", 2_000)
    assert led.spent_usd > 0
    assert set(led.breakdown()) == {"oracle_query", "frontier_in", "frontier_out"}


def test_cache_and_batch_discounts():
    led = Ledger()
    plain = led.charge("frontier_in", 10_000).usd
    cached = led.charge("frontier_in", 10_000, cached=True).usd
    batched = led.charge("frontier_in", 10_000, batch=True).usd
    assert cached == pytest.approx(plain * CostModel().cache_read_mult)
    assert batched == pytest.approx(plain * CostModel().batch_mult)


def test_exhaustion_raises():
    led = Ledger(total_h100s=0.001)
    with pytest.raises(BudgetExhausted):
        for _ in range(10_000):
            led.charge("oracle_query", 100)


def test_unknown_kind_is_rejected():
    with pytest.raises(KeyError):
        Ledger().charge("free_lunch", 1)


def test_decorator_meters():
    led = Ledger()

    @meters("oracle_query")
    def f(x):
        return x * 2

    assert f(3, ledger=led) == 6
    assert len(led.records) == 1

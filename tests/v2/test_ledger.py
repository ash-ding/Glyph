from glyph.v2.ledger import Ledger

def test_usd_by_kind_multipliers():
    l = Ledger()
    l.add_model_usage(input=1000, cache_read=1000, cache_write=1000, output=1000)
    s = l.summary()
    k = s["usd_by_kind"]
    assert abs(k["input"] - 1000 * 5.0e-6) < 1e-12
    assert abs(k["cache_read"] - 1000 * 5.0e-6 * 0.1) < 1e-12
    assert abs(k["cache_write"] - 1000 * 5.0e-6 * 1.25) < 1e-12
    assert abs(k["output"] - 1000 * 25.0e-6) < 1e-12
    assert k["cache_read"] < k["input"] < k["cache_write"]
    assert abs(s["spent_usd"] - sum(k.values())) < 1e-12

def test_accumulates_across_calls():
    l = Ledger()
    l.add_model_usage(input=10, cache_read=0, cache_write=0, output=0)
    l.add_model_usage(input=10, cache_read=0, cache_write=0, output=0)
    assert abs(l.spent_usd - 20 * 5.0e-6) < 1e-12

def test_safety_line():
    l = Ledger()
    l.add_model_usage(input=1000, cache_read=0, cache_write=0, output=1000)
    assert not l.over_safety_line(1e9)
    assert l.over_safety_line(0.0)

def test_gpu_and_cpu_seconds():
    l = Ledger()
    l.add_gpu_seconds("train", 12.5)
    l.add_gpu_seconds("infer", 2.5)
    l.add_gpu_seconds("train", 1.0)
    l.add_cpu_seconds(3.0)
    s = l.summary()
    assert s["gpu_seconds"]["train"] == 13.5
    assert s["gpu_seconds"]["infer"] == 2.5
    assert s["gpu_seconds"]["_total"] == 16.0
    assert s["cpu_seconds"] == 3.0
    assert l.spent_usd == 0.0   # gpu/cpu seconds are NOT model USD

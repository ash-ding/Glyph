"""Self-check #3: measured pi orders the presets the way the design intends."""
import statistics

import pytest

from glyph.config import PRESETS
from glyph.instance import generate

SEEDS = (1001, 1002, 1003, 1004)


def _fast(name):
    return PRESETS[name].with_(n_iid=400, n_comp=140, n_depth=60)


@pytest.mark.slow
def test_pi_is_ordered_across_presets():
    means = {}
    for name in ("pi_low", "pi_mid", "pi_high"):
        means[name] = statistics.mean(
            generate(s, _fast(name)).measured_pi()["pi"] for s in SEEDS)
    assert means["pi_low"] < means["pi_mid"] < means["pi_high"], means


def test_pi_is_in_range_and_full_is_exact():
    m = generate(1001, _fast("pi_mid")).measured_pi()
    assert m["full"] == pytest.approx(1.0)
    assert 0.0 <= m["pi"] <= 1.0

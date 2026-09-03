"""Self-check #3: measured pi orders the presets the way the design intends."""
import statistics

import pytest

from glyph.data.config import PRESETS
from glyph.data.instance import GenerationFailed, generate

SEEDS = (1001, 1002, 1003, 1004)


def _fast(name):
    return PRESETS[name].with_(n_iid=400, n_comp=140, n_depth=60)


@pytest.mark.slow
def test_pi_is_ordered_across_presets():
    """Ordering holds on the mean, over instances that actually generated.

    This check used to pass partly by accident: `pi_low` seed 1002 silently
    produced a short test set with no `depth` split and no binary operators at
    all, and its pi was averaged in like any other.  `GenerationFailed` now
    makes that visible, so the exclusion is explicit and counted rather than
    invisible.

    The exclusion is not incidental -- roughly half of `pi_low`'s seeds cannot
    fill a split, because with two structural operators the held-out third of
    operator pairs is unavoidable.  That is a live design question (how many
    pairs to hold out at small `n_structural`), not a flaw in this test.
    """
    means, spread, unfillable = {}, {}, {}
    for name in ("pi_low", "pi_mid", "pi_high"):
        vals, bad = [], []
        for s in SEEDS:
            try:
                vals.append(generate(s, _fast(name)).measured_pi()["pi"])
            except GenerationFailed as e:
                bad.append((s, e.split))
        assert vals, f"{name}: no seed in {SEEDS} produced a complete instance"
        means[name] = statistics.mean(vals)
        spread[name] = (round(min(vals), 3), round(max(vals), 3))
        unfillable[name] = bad
    assert means["pi_low"] < means["pi_mid"] < means["pi_high"], (means, unfillable)
    # Per-seed ranges overlap between adjacent presets; the phase-diagram axis
    # uses each instance's measured pi, never the preset name, so that is
    # tolerated here and recorded rather than asserted away.
    print("pi per preset:", {k: (round(means[k], 3), spread[k]) for k in means},
          "unfillable:", unfillable)


def test_pi_is_in_range_and_full_is_exact():
    m = generate(1001, _fast("pi_mid")).measured_pi()
    assert m["full"] == pytest.approx(1.0)
    assert 0.0 <= m["pi"] <= 1.0

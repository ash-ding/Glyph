"""Slow, real-API end-to-end smoke tests for protocol v2 (Task 18, Part E).

These hit the real Claude Agent SDK (real API cost, needs a sandbox host with
the bundled CLI + gateway reachable) and are NOT run in normal CI. Select
explicitly with `-m slow`; the fast test suite (`-m "not slow"`) skips this
whole module's real work.
"""
from __future__ import annotations

import pytest

from glyph.v2.harness import RunConfig


@pytest.mark.slow
def test_no_train_smoke(tmp_path):
    """A TINY no_train run over the real SDK: smoke preset, tiny caps."""
    from glyph.v2 import harness

    rc = RunConfig(
        arm="no_train",
        preset="smoke",
        instance_seed=1001,
        q_cap=8,
        submit_cap=1,
        tp=3,
        tf=2,
        n_val=20,
        out_root=str(tmp_path),
    )
    report = harness.run(rc)
    assert report["covariates"]["run_status"] == "completed"
    assert 0.0 <= report["overall"] <= 1.0
    assert report["covariates"]["final_commit"] in {
        "agent", "auto_checked_path", "none",
    }


@pytest.mark.skip(reason="train arm E2E deferred; needs a host with free GPU")
@pytest.mark.slow
def test_train_smoke(tmp_path):
    """A TINY train run over the real SDK: smoke preset, tiny caps."""
    from glyph.v2 import harness

    rc = RunConfig(
        arm="train",
        preset="smoke",
        instance_seed=1001,
        q_cap=8,
        submit_cap=1,
        tp=3,
        tf=2,
        n_val=20,
        out_root=str(tmp_path),
    )
    report = harness.run(rc)
    assert report["covariates"]["run_status"] == "completed"
    assert 0.0 <= report["overall"] <= 1.0
    assert report["covariates"]["final_commit"] in {
        "agent", "auto_checked_path", "none",
    }

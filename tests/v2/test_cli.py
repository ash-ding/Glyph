"""Fast, SDK-free tests for the protocol v2 CLI (Task 18).

No API calls: only argument parsing and RunConfig construction are exercised.
harness.run() is never invoked here.
"""
from __future__ import annotations

from glyph.v2.cli import build_parser, build_run_config
from glyph.v2.harness import RunConfig


def test_run_args_build_expected_run_config():
    p = build_parser()
    args = p.parse_args([
        "run", "--arm", "no_train", "--preset", "smoke", "--seed", "1001", "--q", "5",
    ])
    rc = build_run_config(args)
    assert isinstance(rc, RunConfig)
    assert rc.arm == "no_train"
    assert rc.preset == "smoke"
    assert rc.instance_seed == 1001
    assert rc.q_cap == 5
    # untouched fields keep their RunConfig defaults
    assert rc.submit_cap == 20
    assert rc.tp == 100
    assert rc.tf == 30
    assert rc.usd_line == 300.0
    assert rc.n_val == 5000
    assert rc.max_turns == 60
    assert rc.out_root is None


def test_run_args_all_overrides():
    p = build_parser()
    args = p.parse_args([
        "run", "--arm", "train", "--preset", "pi_mid", "--seed", "7",
        "--model", "claude-x", "--q", "42", "--submit-cap", "3",
        "--tp", "11", "--tf", "4", "--usd-line", "12.5", "--n-val", "500",
        "--max-turns", "9", "--out-root", "/tmp/somewhere",
    ])
    rc = build_run_config(args)
    assert rc == RunConfig(
        arm="train", preset="pi_mid", instance_seed=7, model="claude-x",
        q_cap=42, submit_cap=3, tp=11, tf=4, usd_line=12.5, n_val=500,
        max_turns=9, out_root="/tmp/somewhere",
    )


def test_run_requires_arm():
    import pytest
    p = build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["run", "--preset", "smoke"])


def test_run_args_instance_id_flows_into_run_config():
    p = build_parser()
    args = p.parse_args(["run", "--instance-id", "high_3", "--arm", "no_train"])
    rc = build_run_config(args)
    assert isinstance(rc, RunConfig)
    assert rc.instance_id == "high_3"
    assert rc.arm == "no_train"


def test_run_args_no_instance_id_is_backward_compatible():
    p = build_parser()
    args = p.parse_args([
        "run", "--arm", "no_train", "--preset", "pi_mid", "--seed", "1001",
    ])
    rc = build_run_config(args)
    assert rc.instance_id is None
    assert rc.preset == "pi_mid"
    assert rc.instance_seed == 1001


def test_grid_instance_ids_builds_one_run_config_per_arm_and_id(monkeypatch):
    """`grid --instance-ids ...` must expand to one RunConfig per (arm, instance_id) pair,
    and must NOT also cross the --presets/--seeds grid -- proven here by passing a multi-value
    --presets/--seeds grid alongside --instance-ids and asserting neither the run count nor the
    captured RunConfigs' preset/seed reflect them (harness.run is never actually invoked: the
    real entrypoint is monkeypatched out so no agent/API is touched)."""
    from glyph.v2 import harness
    from glyph.v2.cli import cmd_grid

    captured = []
    monkeypatch.setattr(harness, "run", lambda rc: captured.append(rc))

    p = build_parser()
    args = p.parse_args([
        "grid", "--instance-ids", "low_1", "high_2", "--arms", "no_train",
        "--presets", "pi_low", "pi_high", "--seeds", "1", "2", "3",
    ])
    cmd_grid(args)

    assert len(captured) == 2  # 1 arm x 2 instance_ids, NOT 1 x 2(presets) x 3(seeds)
    assert {rc.instance_id for rc in captured} == {"low_1", "high_2"}
    assert all(rc.arm == "no_train" for rc in captured)
    # the --presets/--seeds grid was not used to build these configs
    default = RunConfig(arm="no_train")
    assert all(rc.preset == default.preset for rc in captured)
    assert all(rc.instance_seed == default.instance_seed for rc in captured)


def test_grid_parser_defaults():
    p = build_parser()
    args = p.parse_args(["grid"])
    assert args.arms == ["no_train"]
    assert args.presets == ["pi_mid"]
    assert args.seeds == [1001]
    assert args.fn.__name__ == "cmd_grid"


def test_parser_construction_does_not_error():
    # a --help-style parser build/parse round trip
    p = build_parser()
    args = p.parse_args(["run", "--arm", "no_train"])
    assert args.fn.__name__ == "cmd_run"

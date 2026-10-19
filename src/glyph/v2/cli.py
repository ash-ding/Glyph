"""Command line entry points for protocol v2 (`python -m glyph.v2 ...`).

Mirrors the argparse style of v1's src/glyph/cli.py. The SDK import stays
local to glyph.v2.harness.run(); this module (and the RunConfig it builds)
never needs claude_agent_sdk, so `python -m glyph.v2 --help` and
`build_run_config()` work with no SDK installed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from glyph.v2.harness import RunConfig
from glyph.v2.free_sandbox import DEFAULT_IMAGE as FREE_DEFAULT_IMAGE


def build_run_config(args: argparse.Namespace) -> RunConfig:
    """Build a RunConfig from parsed `run` (or `grid`-iteration) args."""
    return RunConfig(
        arm=args.arm,
        preset=args.preset,
        instance_seed=args.seed,
        instance_id=getattr(args, "instance_id", None),
        model=args.model,
        student_model=args.student_model,
        q_cap=args.q,
        submit_cap=args.submit_cap,
        tp=args.tp,
        tf=args.tf,
        usd_line=args.usd_line,
        n_val=args.n_val,
        max_turns=args.max_turns,
        out_root=args.out_root,
        **_free_fields(args),
    )


def _free_fields(args: argparse.Namespace) -> dict:
    """The free protocol's RunConfig fields, defaulted when the flags are absent."""
    g = lambda k, d: getattr(args, k, d)
    return {
        "protocol": g("protocol", "v2"),
        "t_p_s": g("t_p", 4.0) * 3600,
        "t_f_s": g("t_f", 30.0) * 60,
        "gpu_minor": g("gpu_minor", None),
        "cpus": g("cpus", 24),
        "mem_gb": g("mem_gb", 160),
        "disk_gb": g("disk_gb", 35),
        "image": g("image", FREE_DEFAULT_IMAGE),
        "scratch_root": g("scratch_root", "/scratch/glyph_free"),
        "pin_cpus": not g("no_cpu_pin", False),
    }


def cmd_run(args: argparse.Namespace) -> int:
    """One arm, one instance, practice+final over the real SDK."""
    from glyph.v2 import harness

    rc = build_run_config(args)
    report = harness.run(rc)
    print(json.dumps(report, indent=2))
    return 0


def cmd_grid(args: argparse.Namespace) -> int:
    """A simple sweep: loop run() over every (arm, preset, seed) tuple, or, when
    `--instance-ids` is given, over every (arm, instance_id) pair instead (each frozen
    id overrides preset/seed via the harness's `resolve_run_instance`)."""
    from glyph.v2 import harness

    out_root = Path(args.out_root) if args.out_root else Path.cwd() / "runs"
    n = 0
    if args.instance_ids:
        for arm in args.arms:
            for instance_id in args.instance_ids:
                rc = RunConfig(
                    arm=arm, instance_id=instance_id, model=args.model,
                    student_model=args.student_model,
                    q_cap=args.q, submit_cap=args.submit_cap, tp=args.tp, tf=args.tf,
                    usd_line=args.usd_line, n_val=args.n_val, max_turns=args.max_turns,
                    out_root=args.out_root,
                )
                harness.run(rc)
                run_dir = out_root / f"{instance_id}_{arm}"
                print(str(run_dir / "report.json"))
                n += 1
    else:
        for arm in args.arms:
            for preset in args.presets:
                for seed in args.seeds:
                    rc = RunConfig(
                        arm=arm, preset=preset, instance_seed=seed, model=args.model,
                        student_model=args.student_model,
                        q_cap=args.q, submit_cap=args.submit_cap, tp=args.tp, tf=args.tf,
                        usd_line=args.usd_line, n_val=args.n_val, max_turns=args.max_turns,
                        out_root=args.out_root,
                    )
                    harness.run(rc)
                    run_dir = out_root / f"{preset}_{arm}_{seed}"
                    print(str(run_dir / "report.json"))
                    n += 1
    print(f"{n} run(s) complete")
    return 0


def _q_type(s: str):
    """--q accepts an integer cap or none/null/unlimited for no cap."""
    if s.lower() in ("none", "null", "unlimited"):
        return None
    return int(s)


def _add_run_args(s: argparse.ArgumentParser) -> None:
    s.add_argument("--arm", required=True, choices=["train", "no_train", "silent"],
                   help="silent: free protocol only; a GPU like train, no environment description")
    s.add_argument("--preset", default="pi_mid")
    s.add_argument("--model", default="claude-opus-4-8")
    s.add_argument("--student-model", default="Qwen/Qwen3-0.6B",
                   help="the trainable student's base model (train arm); "
                        "NOT the frontier model id")
    s.add_argument("--q", type=_q_type, default=1000,
                   help="query cap (q_cap); none/null/unlimited lifts the cap")
    s.add_argument("--submit-cap", type=int, default=20)
    s.add_argument("--tp", type=int, default=100, help="practice turn cap")
    s.add_argument("--tf", type=int, default=30, help="final turn cap")
    s.add_argument("--usd-line", type=float, default=300.0, help="USD safety line")
    s.add_argument("--n-val", type=int, default=5000)
    s.add_argument("--max-turns", type=int, default=60,
                    help="SDK ClaudeAgentOptions.max_turns (agent-loop cap)")
    s.add_argument("--out-root", default=None)
    s.add_argument("--instance-id", default=None,
                    help="frozen instance id (e.g. high_3); overrides --preset/--seed")
    f = s.add_argument_group("free protocol (--protocol free)")
    f.add_argument("--protocol", default="v2", choices=["v2", "free"])
    f.add_argument("--t-p", type=float, default=4.0, help="practice wall-clock, hours")
    f.add_argument("--t-f", type=float, default=30.0, help="final wall-clock, minutes")
    f.add_argument("--gpu-minor", type=int, default=None,
                   help="/dev/nvidiaN for the train and silent arms")
    f.add_argument("--cpus", type=int, default=24)
    f.add_argument("--mem-gb", type=int, default=160)
    f.add_argument("--disk-gb", type=int, default=35)
    f.add_argument("--image", default=FREE_DEFAULT_IMAGE)
    f.add_argument("--scratch-root", default="/scratch/glyph_free")
    f.add_argument("--no-cpu-pin", action="store_true",
                   help="do not pin the container to a block of --cpus CPUs (quota only)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="glyph.v2")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("run", help="one arm, one instance, practice+final")
    s.add_argument("--seed", type=int, default=1001, help="instance seed")
    _add_run_args(s)
    s.set_defaults(fn=cmd_run)

    g = sub.add_parser("grid", help="loop run() over (arm, preset, seed) tuples")
    g.add_argument("--arms", nargs="+", default=["no_train"], choices=["train", "no_train"])
    g.add_argument("--presets", nargs="+", default=["pi_mid"])
    g.add_argument("--seeds", nargs="+", type=int, default=[1001], dest="seeds")
    g.add_argument("--model", default="claude-opus-4-8")
    g.add_argument("--student-model", default="Qwen/Qwen3-0.6B",
                   help="the trainable student's base model (train arm); "
                        "NOT the frontier model id")
    g.add_argument("--q", type=_q_type, default=1000)
    g.add_argument("--submit-cap", type=int, default=20)
    g.add_argument("--tp", type=int, default=100)
    g.add_argument("--tf", type=int, default=30)
    g.add_argument("--usd-line", type=float, default=300.0)
    g.add_argument("--n-val", type=int, default=5000)
    g.add_argument("--max-turns", type=int, default=60)
    g.add_argument("--out-root", default=None)
    g.add_argument("--instance-ids", nargs="+", default=None,
                    help="frozen instance ids (e.g. high_3 low_1); when given, expands to "
                         "one run per (arm, id), overriding --presets/--seeds")
    g.set_defaults(fn=cmd_grid)

    return p


def main(argv=None) -> int:
    p = build_parser()
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())

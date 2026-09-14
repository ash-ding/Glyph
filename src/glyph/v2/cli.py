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


def build_run_config(args: argparse.Namespace) -> RunConfig:
    """Build a RunConfig from parsed `run` (or `grid`-iteration) args."""
    return RunConfig(
        arm=args.arm,
        preset=args.preset,
        instance_seed=args.seed,
        model=args.model,
        q_cap=args.q,
        submit_cap=args.submit_cap,
        tp=args.tp,
        tf=args.tf,
        usd_line=args.usd_line,
        n_val=args.n_val,
        max_turns=args.max_turns,
        out_root=args.out_root,
    )


def cmd_run(args: argparse.Namespace) -> int:
    """One arm, one instance, practice+final over the real SDK."""
    from glyph.v2 import harness

    rc = build_run_config(args)
    report = harness.run(rc)
    print(json.dumps(report, indent=2))
    return 0


def cmd_grid(args: argparse.Namespace) -> int:
    """A simple sweep: loop run() over every (arm, preset, seed) tuple."""
    from glyph.v2 import harness

    out_root = Path(args.out_root) if args.out_root else Path.cwd() / "runs"
    n = 0
    for arm in args.arms:
        for preset in args.presets:
            for seed in args.seeds:
                rc = RunConfig(
                    arm=arm, preset=preset, instance_seed=seed, model=args.model,
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


def _add_run_args(s: argparse.ArgumentParser) -> None:
    s.add_argument("--arm", required=True, choices=["train", "no_train"])
    s.add_argument("--preset", default="pi_mid")
    s.add_argument("--model", default="claude-opus-4-8")
    s.add_argument("--q", type=int, default=1000, help="query cap (q_cap)")
    s.add_argument("--submit-cap", type=int, default=20)
    s.add_argument("--tp", type=int, default=100, help="practice turn cap")
    s.add_argument("--tf", type=int, default=30, help="final turn cap")
    s.add_argument("--usd-line", type=float, default=300.0, help="USD safety line")
    s.add_argument("--n-val", type=int, default=5000)
    s.add_argument("--max-turns", type=int, default=60,
                    help="SDK ClaudeAgentOptions.max_turns (agent-loop cap)")
    s.add_argument("--out-root", default=None)


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
    g.add_argument("--q", type=int, default=1000)
    g.add_argument("--submit-cap", type=int, default=20)
    g.add_argument("--tp", type=int, default=100)
    g.add_argument("--tf", type=int, default=30)
    g.add_argument("--usd-line", type=float, default=300.0)
    g.add_argument("--n-val", type=int, default=5000)
    g.add_argument("--max-turns", type=int, default=60)
    g.add_argument("--out-root", default=None)
    g.set_defaults(fn=cmd_grid)

    return p


def main(argv=None) -> int:
    p = build_parser()
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())

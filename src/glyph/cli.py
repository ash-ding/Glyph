"""Command line entry points."""

from __future__ import annotations

import argparse
import json

from .data.config import PRESETS
from .data.instance import generate
from .data.semantics import describe


def cmd_show(args) -> int:
    cfg = PRESETS[args.preset]
    inst = generate(args.seed, cfg)
    print(f"=== instance seed={args.seed} preset={args.preset} ===")
    print(f"|V|={cfg.n_values}  structural={cfg.n_structural}  "
          f"atomic_ratio={cfg.atomic_ratio}  binary_coupling={cfg.binary_coupling}")
    print(f"table params: {inst.tables.param_count():,}")
    print("\n--- hidden skeleton (agent never sees this) ---")
    for s in inst.skeleton.values():
        print("  " + describe(s))
    print(f"\n--- syntax spec (public) ---\n{inst.syntax_spec()}")
    print(f"\n--- demos ({len(inst.demos)}) ---")
    for a, b in inst.demos[: args.n]:
        print(f"  {a} -> {b}")
    print(f"\n--- test items ---")
    for sp in ("iid", "comp", "depth"):
        items = inst.test_set(sp)
        print(f"  [{sp}] {len(items)}")
        for t in items[: args.n]:
            print(f"      {t.expr_src} -> {t.answer_src}")
    print("\n--- pi ---")
    print("  " + json.dumps(inst.measured_pi(), indent=2))
    return 0


def cmd_pi(args) -> int:
    rows = []
    for name in (args.presets or list(PRESETS)):
        for seed in range(args.seed, args.seed + args.n_seeds):
            m = generate(seed, PRESETS[name]).measured_pi()
            rows.append((name, seed, m))
            print(f"{name:<9} seed={seed}  full={m['full']:.3f}  "
                  f"a_skel={m['a_skel']:.3f}  a_tab={m['a_tab']:.3f}  "
                  f"pi={m['pi']:.3f}")
    return 0


def cmd_run(args) -> int:
    """One arm, one instance, one budget -- the unit the scheduler dispatches."""
    from pathlib import Path

    from .arms.base import RunConfig

    mods = {"a2": "a2_context", "a4": "a4_code", "a6": "a6_weights"}
    if args.arm not in mods:
        print(f"unknown arm {args.arm!r}; known: {', '.join(mods)}")
        return 2
    import importlib
    mod = importlib.import_module(f".arms.{mods[args.arm]}", package="glyph")

    rc = RunConfig(arm=args.arm, preset=args.preset,
                   instance_seed=args.instance_seed, run_seed=args.run_seed,
                   budget_h100s=args.budget, base_model=args.base_model,
                   out_root=Path(args.out_root), n_test=args.n_test,
                   max_turns=args.max_turns)
    report = mod.run(rc)
    print(report.to_json())
    return 0


def cmd_grid(args) -> int:
    """A whole sweep, grouped by instance so the comparison stays paired."""
    from .worker import grid, run_grid

    jobs = grid(arms=tuple(args.arms), presets=tuple(args.presets),
                instance_seeds=tuple(args.instance_seeds),
                budgets=tuple(args.budgets), run_seeds=tuple(args.run_seeds),
                n_test=args.n_test, out_root=args.out_root)
    print(f"{len(jobs)} runs across {len(set(j.instance_seed for j in jobs))} "
          f"instance(s)")
    if args.dry_run:
        for j in jobs:
            print("  " + " ".join(j.argv()[2:]))
        return 0
    summary = run_grid(jobs, timeout=args.timeout,
                       log_root=f"{args.out_root}/_logs")
    print("")
    print(f"{summary['ok']}/{summary['jobs']} ok in {summary['minutes']}m")
    return 0 if summary["ok"] == summary["jobs"] else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="glyph")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("show", help="print one instance end to end")
    s.add_argument("--preset", default="pi_mid", choices=list(PRESETS))
    s.add_argument("--seed", type=int, default=1001)
    s.add_argument("-n", type=int, default=5)
    s.set_defaults(fn=cmd_show)

    s = sub.add_parser("run", help="one arm on one instance at one budget")
    s.add_argument("--arm", required=True)
    s.add_argument("--preset", default="pi_mid", choices=list(PRESETS))
    s.add_argument("--instance-seed", type=int, default=1001)
    s.add_argument("--run-seed", type=int, default=0)
    s.add_argument("--budget", type=float, default=15000.0,
                   help="prepare-phase budget in H100-seconds")
    s.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    s.add_argument("--out-root", default="runs")
    s.add_argument("--n-test", type=int, default=None)
    s.add_argument("--max-turns", type=int, default=40)
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("grid", help="a sweep, scheduled across GPUs by instance")
    s.add_argument("--arms", nargs="+", default=["a2", "a4", "a6"])
    s.add_argument("--presets", nargs="+", default=["pi_mid"])
    s.add_argument("--instance-seeds", nargs="+", type=int, default=[1001])
    s.add_argument("--budgets", nargs="+", type=float, default=[15000.0])
    s.add_argument("--run-seeds", nargs="+", type=int, default=[0])
    s.add_argument("--n-test", type=int, default=None)
    s.add_argument("--out-root", default="runs")
    s.add_argument("--timeout", type=float, default=7200)
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_grid)

    s = sub.add_parser("pi", help="measure pi across presets")
    s.add_argument("--presets", nargs="*")
    s.add_argument("--seed", type=int, default=1001)
    s.add_argument("--n-seeds", type=int, default=3)
    s.set_defaults(fn=cmd_pi)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())

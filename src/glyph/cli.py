"""Command line entry points."""

from __future__ import annotations

import argparse
import json

from .config import PRESETS
from .instance import generate
from .semantics import describe


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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="glyph")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("show", help="print one instance end to end")
    s.add_argument("--preset", default="pi_mid", choices=list(PRESETS))
    s.add_argument("--seed", type=int, default=1001)
    s.add_argument("-n", type=int, default=5)
    s.set_defaults(fn=cmd_show)

    s = sub.add_parser("pi", help="measure pi across presets")
    s.add_argument("--presets", nargs="*")
    s.add_argument("--seed", type=int, default=1001)
    s.add_argument("--n-seeds", type=int, default=3)
    s.set_defaults(fn=cmd_pi)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())

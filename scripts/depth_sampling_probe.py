"""E-5: expression depth is not sampled, it is written into each split.

`_sample`'s recursion stops only when the budget runs out, so an expression's
depth *equals* its budget and `min_depth` is close to a no-op. On pi_mid the
whole test set is two points: 8800 items at depth 2 (iid + comp) and 1200 at
depth 4 (depth), with nothing at 3 and zero variance inside iid.

Two consequences worth measuring before the data layer is frozen. The `depth`
split is a 2 -> 4 jump across a gap rather than an extrapolation along a
distribution. And the agent's own probes are shallow -- it buys single-level
expressions to isolate table entries, which is the right way to read a table --
so its purchases and the test set do not overlap in depth at all.

This probes a sampler that stops early with some probability, without touching
`src/`: same grammar, same constraints, only the recursion's stopping rule.
What it costs is measured too -- a stopping rule that makes deep expressions
rare makes the `depth` split expensive or impossible to fill.

Usage:  python scripts/depth_sampling_probe.py --preset pi_mid --seeds 3
"""
from __future__ import annotations

import argparse
import collections
import json
import time

import numpy as np

from glyph.data import instance as inst_mod
from glyph.data.config import PRESETS
from glyph.data.grammar import (App, K_RANGE, Lit, SHAPE_RESULT, Val, binary_names,
                           depth, enabled_ops, parse, unary_names)
from glyph.data.instance import GenerationFailed, generate
from glyph.data.interp import Interpreter
from glyph.data.measure import _score
from glyph.data.tables import IdentityTables

_ORIGINAL = inst_mod._sample


def _sample_with_stop(rng, cfg, want: str, budget: int, stop_prob: float):
    """`_sample`, but each level may stop early instead of always spending the
    whole budget.  The root never stops -- a stopped root is a bare literal,
    which is not an expression."""
    ops = enabled_ops(cfg)
    us, bs = unary_names(cfg), binary_names(cfg)
    lo, hi = cfg.list_len_range

    def leaf_list() -> Lit:
        n = int(rng.integers(lo, hi + 1))
        return Lit(tuple(int(rng.integers(cfg.n_values)) for _ in range(n)))

    def go(ttype: str, b: int, root: bool):
        cands = [(o, s) for o, s in ops if SHAPE_RESULT[s] == ttype]
        stopped = (not root) and rng.random() < stop_prob
        if b <= 0 or not cands or stopped:
            return leaf_list() if ttype == "LIST" else Val(int(rng.integers(cfg.n_values)))
        atomic = [c for c in cands if c[1] in ("UL", "LB")]
        pure = [c for c in cands if c[1] in ("L", "KL")]
        pool = atomic if (atomic and (not pure or rng.random() < cfg.atomic_ratio)) else pure
        if not pool:
            pool = cands
        op, shape = pool[int(rng.integers(len(pool)))]
        args = []
        for slot in shape:
            if slot == "U":
                args.append(us[int(rng.integers(len(us)))])
            elif slot == "B":
                args.append(bs[int(rng.integers(len(bs)))])
            elif slot == "K":
                args.append(int(rng.integers(K_RANGE[0], K_RANGE[1] + 1)))
            else:
                args.append(go("LIST", b - 1, False))
        return App(op, tuple(args))

    return go(want, budget, True)


def _describe(inst) -> dict:
    cfg = inst.cfg
    skel = Interpreter(cfg, inst.skeleton, IdentityTables())
    per_split = {}
    for s in ("iid", "comp", "depth"):
        group = inst.test_set(s)
        if not group:
            continue
        h = collections.Counter(depth(parse(t.expr_src, cfg)) for t in group)
        per_split[s] = {
            "n": len(group),
            "depth_hist": dict(sorted(h.items())),
            "mean_depth": round(sum(k * v for k, v in h.items()) / len(group), 2),
            "needs_no_table": round(
                sum(1 for t in group if not t.needs_u and not t.needs_b) / len(group), 4),
        }
    lookups = collections.Counter(len(t.needs_u) + len(t.needs_b) for t in inst.test)
    return {"n_test": len(inst.test), "per_split": per_split,
            "skeleton_ceiling": round(_score(skel, inst, inst.test, graded=False), 4),
            "mean_lookups_per_item": round(
                sum(k * c for k, c in lookups.items()) / len(inst.test), 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="pi_mid")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=1001)
    ap.add_argument("--stop-probs", type=float, nargs="+",
                    default=[0.0, 0.15, 0.3, 0.5])
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--out", default="depth_probe.jsonl")
    a = ap.parse_args()

    cfg = PRESETS[a.preset].scaled(a.n_test)
    with open(a.out, "w") as fh:
        for sp in a.stop_probs:
            if sp == 0.0:
                inst_mod._sample = _ORIGINAL          # exactly today's sampler
            else:
                inst_mod._sample = (
                    lambda rng, cfg_, want, budget, _sp=sp:
                    _sample_with_stop(rng, cfg_, want, budget, _sp))
            for seed in range(a.seed0, a.seed0 + a.seeds):
                t0 = time.time()
                try:
                    inst = generate(seed, cfg)
                    row = {"stop_prob": sp, "seed": seed, "ok": True,
                           **_describe(inst)}
                except GenerationFailed as e:
                    row = {"stop_prob": sp, "seed": seed, "ok": False,
                           "failed_split": e.split, "made": e.made,
                           "want": e.want, "diag": e.diag}
                row["seconds"] = round(time.time() - t0, 1)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                if row["ok"]:
                    hs = {s: v["depth_hist"] for s, v in row["per_split"].items()}
                    print(f"stop={sp:.2f} seed={seed}  ceil={row['skeleton_ceiling']:.3f} "
                          f"lookups={row['mean_lookups_per_item']:.2f}  {hs}  {row['seconds']}s")
                else:
                    print(f"stop={sp:.2f} seed={seed}  FAIL split={row['failed_split']} "
                          f"({row['made']}/{row['want']})  {row['seconds']}s")
    inst_mod._sample = _ORIGINAL
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""E-6: how many test items an arm comparison actually needs.

`config.py` says real runs use the full 10^4, for a ~0.5% standard error. E0
used 200, where the standard error on a proportion near 0.05 is about 1.6% --
the same size as the differences it reported. A6 at 0.035 and A2 at 0.055 are
7/200 and 11/200; nothing separates them.

The full set is not an option for every arm: A0' costs about 752,000 H100-s
over 10^4 items. So the question is how small a subset can be and still
resolve the gaps we care about, and how much a *paired* design (every arm
scored on the same items) buys over independent draws.

The arms here are synthetic but not naive. A student that has learned a table
entry gets it right in every item that needs it, so correctness is correlated
across items through the entries they share -- draw a set of known entries,
then an item is correct when all the entries it needs are known. That
correlation is exactly what determines how quickly subsample error falls, so
assuming independent per-item coin flips would give an answer that is too
optimistic.

Usage:  python scripts/subsample_power.py --preset pi_mid --seed 1001
"""
from __future__ import annotations

import argparse
import collections
import json

import numpy as np

from glyph.data.config import PRESETS
from glyph.data.instance import generate


def arm_vector(inst, p_known: float, rng) -> np.ndarray:
    """Per-item correctness for a student that knows a fraction of the table.

    Entries are drawn once and shared across items, so two items needing the
    same entry succeed or fail together.
    """
    need_u, need_b = set(), set()
    for t in inst.test:
        need_u |= set(t.needs_u); need_b |= set(t.needs_b)
    known_u = {e for e in need_u if rng.random() < p_known}
    known_b = {e for e in need_b if rng.random() < p_known}
    return np.array([
        set(t.needs_u) <= known_u and set(t.needs_b) <= known_b
        for t in inst.test], dtype=bool)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="pi_mid")
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--p-low", type=float, default=0.0,
                    help="weaker arm: fraction of table entries known")
    ap.add_argument("--p-high", type=float, default=0.5)
    ap.add_argument("--arm-seeds", type=int, default=5)
    ap.add_argument("--reps", type=int, default=3000)
    ap.add_argument("--ns", type=int, nargs="+",
                    default=[100, 200, 500, 1000, 2000, 5000, 10000])
    ap.add_argument("--out", default="subsample_power.json")
    a = ap.parse_args()

    inst = generate(a.seed, PRESETS[a.preset])
    N = len(inst.test)
    splits = np.array([t.split for t in inst.test])
    idx_by_split = {s: np.flatnonzero(splits == s) for s in ("iid", "comp", "depth")}

    rows, truths = [], []
    for arm_seed in range(a.arm_seeds):
        rng = np.random.default_rng(10_000 + arm_seed)
        lo = arm_vector(inst, a.p_low, rng)
        hi = arm_vector(inst, a.p_high, rng)
        truth = hi.mean() - lo.mean()
        truths.append(truth)
        mc = np.random.default_rng(20_000 + arm_seed)
        for n in a.ns:
            n = min(n, N)
            ind, pair, strat = [], [], []
            for _ in range(a.reps):
                i1 = mc.choice(N, n, replace=False)
                i2 = mc.choice(N, n, replace=False)
                ind.append(hi[i2].mean() - lo[i1].mean())
                pair.append(hi[i1].mean() - lo[i1].mean())
                # proportional allocation across the three splits
                take = np.concatenate([
                    mc.choice(v, min(len(v), max(1, round(n * len(v) / N))),
                              replace=False)
                    for v in idx_by_split.values()])
                strat.append(hi[take].mean() - lo[take].mean())
            rows.append({"arm_seed": arm_seed, "n": n, "truth": float(truth),
                         "se_independent": float(np.std(ind)),
                         "se_paired": float(np.std(pair)),
                         "se_paired_stratified": float(np.std(strat)),
                         "se_single_arm": float(np.std(
                             [hi[mc.choice(N, n, replace=False)].mean()
                              for _ in range(500)]))})

    print(f"{a.preset} seed {a.seed}   N={N}   true gap "
          f"{np.mean(truths):.4f} (p={a.p_low} vs p={a.p_high})\n")
    print(f"{'n':>6}  {'SE single arm':>13}  {'SE indep':>9}  {'SE paired':>9}  "
          f"{'SE paired+strat':>15}  {'paired resolves gap?':>20}")
    for n in a.ns:
        n = min(n, N)
        got = [r for r in rows if r["n"] == n]
        m = {k: float(np.mean([r[k] for r in got]))
             for k in ("se_single_arm", "se_independent", "se_paired",
                       "se_paired_stratified")}
        ok = "yes" if abs(np.mean(truths)) > 2 * m["se_paired"] else "NO"
        print(f"{n:>6}  {m['se_single_arm']:>13.4f}  {m['se_independent']:>9.4f}  "
              f"{m['se_paired']:>9.4f}  {m['se_paired_stratified']:>15.4f}  {ok:>20}")

    with open(a.out, "w") as fh:
        json.dump({"preset": a.preset, "seed": a.seed, "N": N,
                   "true_gap": float(np.mean(truths)), "rows": rows}, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

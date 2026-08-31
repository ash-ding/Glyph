#!/usr/bin/env python3
"""A0' -- the information-saturated frontier, on a paired subset.

Answers #20: does in-context learning extract table knowledge at all?

The evidence is deliberately more generous than any arm could buy. The original
design paired A0' against another arm's query log so that only the reader
varied; under the capacity framing the sharper question is whether the frontier
can do it *given more than any agent would have*, because a failure there does
not depend on how much a particular agent happened to purchase.

Two kinds of evidence, mixed:

  probes       single-level expressions that isolate one table entry at a time,
               which is how the A6 agent itself chose to read the table and the
               most information a query can carry
  in-domain    expressions drawn the way test items are, so the skeleton is
               inferable from the same distribution it will be tested on

The scored subset is stratified and drawn from a fixed seed, so every arm can
be scored on exactly the same items -- which is what makes the comparison
paired, and by #9 is worth about a halving of the standard error on a
difference.

Usage:
  python scripts/run_a0p.py --preset pi_mid --seed 1001 --n-facts 2000 --n-test 500
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

from glyph.arms import a0p_saturated
from glyph.arms.base import RunConfig
from glyph.config import PRESETS
from glyph.grammar import render, render_list, render_value
from glyph.instance import _sample_constrained, generate


def buy(inst, n_probe: int, n_domain: int, seed: int) -> list[tuple[str, str]]:
    """Purchase evidence through the real oracle, so the query log is real."""
    cfg = inst.cfg
    rng = np.random.default_rng(seed)
    out: list[tuple[str, str]] = []

    # probes: shallow, one operator, so each answer isolates table entries
    tries = 0
    while len(out) < n_probe and tries < n_probe * 60:
        tries += 1
        e = _sample_constrained(rng, cfg.with_(depth_stop_prob=0.9),
                                1, forbid=set(), require=None, min_depth=1)
        if e is None:
            continue
        src = render(e, cfg)
        try:
            out.append((src, inst.query(src)))
        except Exception:
            pass

    # in-domain: drawn the way test items are
    tries = 0
    while len(out) < n_probe + n_domain and tries < n_domain * 60:
        tries += 1
        e = _sample_constrained(rng, cfg, cfg.demo_max_depth,
                                forbid=set(), require=None, min_depth=1)
        if e is None:
            continue
        src = render(e, cfg)
        try:
            out.append((src, inst.query(src)))
        except Exception:
            pass
    return out


def paired_subset(inst, n: int, seed: int = 777):
    """Stratified, fixed -- the same items for every arm."""
    rng = np.random.default_rng(seed)
    by = collections.defaultdict(list)
    for i, t in enumerate(inst.test):
        by[t.split].append(i)
    picked: list[int] = []
    for split in sorted(by):
        g = by[split]
        k = max(1, round(n * len(g) / len(inst.test)))
        picked += [g[i] for i in rng.permutation(len(g))[:k]]
    picked.sort()
    return [inst.test[i] for i in picked]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="pi_mid")
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--n-facts", type=int, default=2000)
    ap.add_argument("--probe-frac", type=float, default=0.7)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--max-tokens", type=int, default=128000)
    ap.add_argument("--chunk", type=int, default=50,
                    help="items per call. Larger is cheaper -- the evidence "
                         "block is re-sent every call -- and safer, because "
                         "reasoning cost is per call rather than per item.")
    ap.add_argument("--out-root", default="runs")
    a = ap.parse_args()

    cfg = PRESETS[a.preset]
    inst = generate(a.seed, cfg)
    n_probe = int(a.n_facts * a.probe_frac)
    t0 = time.time()
    purchased = buy(inst, n_probe, a.n_facts - n_probe, a.seed)
    items = paired_subset(inst, a.n_test)

    depths = collections.Counter()
    from glyph.grammar import depth, parse
    for e, _ in purchased:
        depths[depth(parse(e, cfg))] += 1
    print(f"evidence  {len(purchased)} facts, depths {dict(sorted(depths.items()))}")
    print(f"          entries revealed: unary {len(inst.query_log.unary)} "
          f"binary {len(inst.query_log.binary)}")
    print(f"scored    {len(items)} items, "
          f"{dict(collections.Counter(t.split for t in items))}")
    print(f"purchase took {time.time() - t0:.1f}s", flush=True)

    rc = RunConfig(arm="a0p", preset=a.preset, instance_seed=a.seed,
                   budget_h100s=1e9, out_root=Path(a.out_root))
    report = a0p_saturated.run(rc, purchased=purchased,
                               max_tokens=a.max_tokens, items=items,
                               chunk=a.chunk)
    print("\n" + report.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

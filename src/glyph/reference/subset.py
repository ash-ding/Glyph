"""Paired scored subset and cheap CPU ceilings for the reference battery.

`paired_subset` is a verbatim port of `scripts/run_a0p.py`'s standalone
function of the same name: a stratified, seed-fixed sample of `inst.test` so
every arm can be scored on exactly the same items. `ceilings_on` reuses
`GlyphInstance.ceilings` (the skeleton/table oracles) and `glyph.seal.headroom`
rather than reimplementing any interpreter logic.
"""

from __future__ import annotations

import collections

import numpy as np

from ..data.instance import TestItem
from ..seal import headroom


def paired_subset(inst, n: int = 500, seed: int = 777) -> list[TestItem]:
    """Stratified, fixed -- the same items for every arm.

    Ported verbatim from scripts/run_a0p.py's paired_subset.
    """
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


def ceilings_on(inst, items: list[TestItem]) -> dict:
    """Skeleton / table / perfect CPU ceilings for `items`, with headroom.

    `skeleton` and `table` come straight out of `inst.ceilings(items)`.
    `tail` restricts each oracle to the items this (freshly generated) instance
    still needs a table entry for -- `None` when there are none. `perfect` is
    1.0 everywhere by construction. `headroom` reads each oracle's overall
    against the skeleton ceiling, per `glyph.seal.headroom`.
    """
    base = inst.ceilings(items)
    tail_items = [t for t in items if inst.is_tail(t)]
    tail_ceilings = inst.ceilings(tail_items) if tail_items else None

    out: dict = {}
    for name in ("skeleton", "table"):
        overall = base[name]["overall"]
        by_split = {k: v for k, v in base[name].items() if k != "overall"}
        tail = tail_ceilings[name]["overall"] if tail_ceilings is not None else None
        out[name] = {"overall": overall, "by_split": by_split, "tail": tail}

    skeleton_overall = out["skeleton"]["overall"]
    for name in ("skeleton", "table"):
        out[name]["headroom"] = headroom(out[name]["overall"], skeleton_overall)

    splits = sorted({t.split for t in items})
    out["perfect"] = {
        "overall": 1.0,
        "by_split": {s: 1.0 for s in splits},
        "tail": 1.0,
        "headroom": 1.0,
    }
    return out

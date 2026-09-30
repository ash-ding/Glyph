"""Per-op probe set: bare-atomic items generated OUTSIDE the instance.

Derived on demand from a dedicated RNG stream, so it is exempt from the
frozen-instance fingerprint (which hashes demos/test/held_pairs/val only).
`split` carries the op name, which makes score_file's by_split grouping a
per-op score for free.  Probe cells are deliberately NOT refused by the
query oracle -- refusing them would leak which cells are probes; seen vs
unseen is derived after the fact from the run's LookupLog (same stance as
`tail`).
"""

from __future__ import annotations

import numpy as np

from .grammar import AtomApp, binary_names, render, render_value, unary_names
from .instance import GlyphInstance, TestItem

PROBE_SALT = 20260929


def probe_set(inst: GlyphInstance, n_per_op: int = 100) -> list[TestItem]:
    cfg = inst.cfg
    rng = np.random.default_rng((inst.seed, PROBE_SALT))
    out: list[TestItem] = []

    for op in unary_names(cfg):
        n = min(n_per_op, cfg.n_values)
        for i in rng.choice(cfg.n_values, size=n, replace=False):
            i = int(i)
            e = AtomApp(op, (i,))
            out.append(TestItem(
                expr_src=render(e, cfg),
                answer_src=render_value(inst.tables.apply_unary(op, i), cfg),
                split=op,
                needs_u=frozenset({(op, i)}),
                needs_b=frozenset(),
            ))

    for op in binary_names(cfg):
        pairs: set[tuple[int, int]] = set()
        while len(pairs) < n_per_op:
            i, j = int(rng.integers(cfg.n_values)), int(rng.integers(cfg.n_values))
            pairs.add((i, j))
        for i, j in sorted(pairs):
            e = AtomApp(op, (i, j))
            out.append(TestItem(
                expr_src=render(e, cfg),
                answer_src=render_value(inst.tables.apply_binary(op, i, j), cfg),
                split=op,
                needs_u=frozenset(),
                needs_b=frozenset({(op, i, j)}),
            ))

    return out

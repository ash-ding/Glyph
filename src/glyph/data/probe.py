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

from .grammar import (App, AtomApp, K_RANGE, Lit, binary_names, enabled_ops,
                      render, render_value, unary_names)
from .instance import GlyphInstance, TestItem

PROBE_SALT = 20260929


def _struct_cells(shape: str) -> list[tuple]:
    """The behavior cells a single pure-structure application branches on.

    Values are mere cargo for an L/KL op (transforms permute/select them), so
    coverage enumerates what the transform/guard space actually reads: list
    length, the guard-relevant equality pattern, and k (KL only).  8 cells for
    an L op, 24 for a KL op.
    """
    cells = []
    for ln in (2, 3, 4):
        variants = ["distinct", "first_eq_last"]
        if ln >= 3:
            variants.append("internal_dup")
        for v in variants:
            if shape == "KL":
                for k in range(K_RANGE[0], K_RANGE[1] + 1):
                    cells.append((ln, v, k))
            else:
                cells.append((ln, v, None))
    return cells


def probe_set(inst: GlyphInstance, n_per_op: int = 100,
              struct_reps: int = 2) -> list[TestItem]:
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

    # Structural probes: single applications of the PURE-STRUCTURE ops (L/KL
    # shapes) -- depth-1, table-free, answer depends only on the skeleton.
    # UL/LB ops cannot be isolated (they entangle the tables).  Unlike bare
    # atomics these CAN coincide with depth-1 iid items, so collisions with
    # the sealed test/val sets are redrawn.
    if struct_reps:
        forbidden = {t.expr_src for t in inst.test} | {t.expr_src for t in inst.val}
        used = {t.expr_src for t in out}
        for op, shape in enabled_ops(cfg):
            if shape not in ("L", "KL"):
                continue
            for ln, variant, k in _struct_cells(shape):
                for _ in range(struct_reps):
                    for _attempt in range(200):
                        vals = [int(x) for x in
                                rng.choice(cfg.n_values, size=ln, replace=False)]
                        if variant == "first_eq_last":
                            vals[-1] = vals[0]
                        elif variant == "internal_dup":
                            vals[2] = vals[1]
                        args = (Lit(tuple(vals)),) if shape == "L" else (k, Lit(tuple(vals)))
                        e = App(op, args)
                        src = render(e, cfg)
                        if src not in forbidden and src not in used:
                            break
                    else:
                        raise RuntimeError(
                            f"could not draw a fresh structural probe for {op}")
                    used.add(src)
                    res = inst.P.eval(e)
                    answer = (render_value(res, cfg) if isinstance(res, int)
                              else "[" + ", ".join(render_value(i, cfg)
                                                   for i in res) + "]")
                    out.append(TestItem(
                        expr_src=src,
                        answer_src=answer,
                        split=op,
                        needs_u=frozenset(),
                        needs_b=frozenset(),
                    ))

    return out

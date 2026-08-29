"""A Glyph instance: one hidden interpreter, its demos, its query oracle and
its sealed test set.

One run binds one instance.  The prepare phase and the test phase share the
same P -- held-out is over *expressions*, not over interpreters.  Otherwise
nothing the agent bought during prepare would transfer.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field

import numpy as np

from .config import GlyphConfig
from .grammar import (App, Expr, K_RANGE, Lit, SHAPE_RESULT, Val, check,
                      depth, enabled_ops, op_pairs, parse, render,
                      render_list, render_value, syntax_spec, unary_names,
                      binary_names)
from .interp import Interpreter, LookupLog
from .semantics import sample_skeleton, trivial_skeleton
from .tables import IdentityTables, Tables

SPLITS = ("iid", "comp", "depth")

# How many consecutive rejected samples before a split is declared unfillable.
_STALL_LIMIT = 8000


class GenerationFailed(RuntimeError):
    """A split could not be filled.

    Returning a short test set silently is worse than failing.  `pi_low` seed
    1002 produced 8800 items with no `depth` split at all and no warning, and
    was then compared against 10000-item instances as though the two were the
    same measurement -- different size, different split composition, no way to
    tell from the report.

    The shortfall is essentially always structural: the grammar cannot reach
    the requested depth, or no expression can carry a held-out pair.  Resampling
    does not fix that, so this raises rather than retrying harder, and carries
    the diagnosis of *which* constraint did the rejecting.
    """

    def __init__(self, split: str, made: int, want: int, diag: dict):
        self.split, self.made, self.want, self.diag = split, made, want, diag
        detail = ", ".join(f"{k}={v}" for k, v in diag.items())
        super().__init__(
            f"split {split!r}: only {made}/{want} items could be generated "
            f"({detail}). This config cannot fill the split -- change the "
            f"config or drop the seed; do not accept a short test set.")


@dataclass(frozen=True)
class TestItem:
    expr_src: str
    answer_src: str
    split: str
    needs_u: frozenset
    needs_b: frozenset


# ---------------------------------------------------------------------
# Expression sampling
# ---------------------------------------------------------------------
def _sample(rng: np.random.Generator, cfg: GlyphConfig, want: str,
            budget: int) -> Expr:
    ops = enabled_ops(cfg)
    us, bs = unary_names(cfg), binary_names(cfg)
    lo, hi = cfg.list_len_range

    def leaf_list() -> Lit:
        n = int(rng.integers(lo, hi + 1))
        return Lit(tuple(int(rng.integers(cfg.n_values)) for _ in range(n)))

    def go(ttype: str, b: int) -> Expr:
        cands = [(o, s) for o, s in ops if SHAPE_RESULT[s] == ttype]
        if b <= 0 or not cands:
            return leaf_list() if ttype == "LIST" else Val(int(rng.integers(cfg.n_values)))
        # atomic_ratio biases toward operators that consume atomic operators
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
                args.append(go("LIST", b - 1))
        return App(op, tuple(args))

    return go(want, budget)


def _sample_constrained(rng, cfg, budget, forbid: set, require: set | None,
                        min_depth: int, tries: int = 400) -> Expr | None:
    for _ in range(tries):
        want = "VAL" if rng.random() < cfg.binary_freq else "LIST"
        e = _sample(rng, cfg, want, budget)
        if depth(e) < min_depth:
            continue
        pairs = op_pairs(e)
        if pairs & forbid:
            continue
        if require is not None and not (pairs & require):
            continue
        try:
            check(e, cfg)
        except SyntaxError:
            continue
        return e
    return None


def _diagnose(rng, cfg, budget, forbid, require, min_depth,
              tries: int = 300) -> dict:
    """Which constraint is rejecting every sample.

    Reported inside `GenerationFailed` so the failure names its own cause
    instead of leaving the reader to guess between "too deep", "held-out pair
    unreachable" and "grammar produces nothing at this shape".
    """
    hist: collections.Counter = collections.Counter()
    forbidden = missing = malformed = 0
    for _ in range(tries):
        want = "VAL" if rng.random() < cfg.binary_freq else "LIST"
        e = _sample(rng, cfg, want, budget)
        hist[depth(e)] += 1
        pairs = op_pairs(e)
        if pairs & forbid:
            forbidden += 1
        if require is not None and not (pairs & require):
            missing += 1
        try:
            check(e, cfg)
        except SyntaxError:
            malformed += 1
    return {"sampled": tries,
            "depth_hist": dict(sorted(hist.items())),
            "min_depth_required": min_depth,
            "reached_min_depth": sum(c for d, c in hist.items() if d >= min_depth),
            "hit_forbidden_pair": forbidden,
            "missing_required_pair": missing,
            "malformed": malformed}


# ---------------------------------------------------------------------
# Instance
# ---------------------------------------------------------------------
class GlyphInstance:
    def __init__(self, cfg: GlyphConfig, seed: int):
        self.cfg = cfg
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.skeleton = sample_skeleton(cfg, rng)
        self.tables = Tables(cfg, rng)
        self.P = Interpreter(cfg, self.skeleton, self.tables)

        # held-out operator pairs -> the `comp` split
        all_pairs = {(a, b) for a, _ in enabled_ops(cfg) for b, _ in enabled_ops(cfg)}
        n_hold = max(1, len(all_pairs) // 3)
        pool = sorted(all_pairs)
        self.held_pairs = {pool[i] for i in
                           rng.permutation(len(pool))[:n_hold]}

        self.demos = self._make_demos(rng)
        self.test = self._make_test(rng)
        self.query_count = 0
        self.query_log = LookupLog()

    # -- agent-visible, free ------------------------------------------
    def syntax_spec(self) -> str:
        return syntax_spec(self.cfg)

    def demo_block(self) -> str:
        return "\n".join(f"  {a} -> {b}" for a, b in self.demos)

    # -- agent-visible, metered ---------------------------------------
    def query(self, expr_src: str) -> str:
        """Evaluate one expression.  Syntax errors are still charged -- the
        agent must not be able to probe the grammar for free."""
        self.query_count += 1
        e = parse(expr_src, self.cfg)      # raises SyntaxError (already charged)
        check(e, self.cfg)
        out, log = self.P.eval_logged(e)
        self.query_log |= log
        return self._render_out(out)

    # -- sealed --------------------------------------------------------
    def test_set(self, split: str | None = None) -> list[TestItem]:
        if split is None:
            return list(self.test)
        return [t for t in self.test if t.split == split]

    def derive_tail(self, log: LookupLog | None = None) -> list[int]:
        """Indices of test items whose required table entries were never
        bought.  Derived per run, not fixed at generation time -- the agent
        may query anything, so 'never seen' is only knowable afterwards."""
        log = self.query_log if log is None else log
        out = []
        for i, t in enumerate(self.test):
            if (t.needs_u - log.unary) or (t.needs_b - log.binary):
                out.append(i)
        return out

    def measured_pi(self) -> dict[str, float]:
        from .measure import measure_pi
        return measure_pi(self)

    # -- internals -----------------------------------------------------
    def _render_out(self, out) -> str:
        return (render_value(out, self.cfg) if isinstance(out, int)
                else render_list(out, self.cfg))

    def _make_demos(self, rng) -> list[tuple[str, str]]:
        # The stall counter is not defensive: without it a config whose demo
        # constraints cannot be satisfied spins forever with no output.
        out, stall = [], 0
        while len(out) < self.cfg.n_demos:
            if stall >= _STALL_LIMIT:
                raise GenerationFailed(
                    "demos", len(out), self.cfg.n_demos,
                    _diagnose(rng, self.cfg, self.cfg.demo_max_depth,
                              self.held_pairs, None, 1))
            e = _sample_constrained(rng, self.cfg, self.cfg.demo_max_depth,
                                    forbid=self.held_pairs, require=None, min_depth=1)
            if e is None:
                stall += 1
                continue
            stall = 0
            out.append((render(e, self.cfg), self._render_out(self.P.eval(e))))
        return out

    def _make_test(self, rng) -> list[TestItem]:
        cfg = self.cfg
        spec = [("iid", cfg.n_iid, cfg.demo_max_depth, self.held_pairs, None, 1),
                ("comp", cfg.n_comp, cfg.demo_max_depth, set(), self.held_pairs, 1),
                ("depth", cfg.n_depth, cfg.max_expr_depth, self.held_pairs, None,
                 cfg.demo_max_depth + 1)]
        items: list[TestItem] = []
        seen: set[str] = {a for a, _ in self.demos}
        for name, n, budget, forbid, require, min_d in spec:
            made, stall = 0, 0
            while made < n:
                if stall >= _STALL_LIMIT:
                    raise GenerationFailed(
                        name, made, n,
                        _diagnose(rng, cfg, budget, forbid, require, min_d))
                e = _sample_constrained(rng, cfg, budget, forbid, require, min_d)
                if e is None:
                    stall += 1
                    continue
                src = render(e, cfg)
                if src in seen:
                    stall += 1
                    continue
                seen.add(src)
                out, log = self.P.eval_logged(e)
                items.append(TestItem(src, self._render_out(out), name,
                                      frozenset(log.unary), frozenset(log.binary)))
                made, stall = made + 1, 0

        # Every split present at its full size, or the instance does not exist.
        # A per-split loop that each ran to completion can still assemble a set
        # that is not what the config asked for, and the report shows only the
        # total -- so check the assembled object, not just the loops.
        by_split = collections.Counter(t.split for t in items)
        if len(items) != cfg.n_test or set(by_split) != set(SPLITS):
            raise GenerationFailed(
                "all", len(items), cfg.n_test,
                {"by_split": dict(by_split),
                 "wanted": {"iid": cfg.n_iid, "comp": cfg.n_comp,
                            "depth": cfg.n_depth}})
        return items


def generate(seed: int, cfg: GlyphConfig) -> GlyphInstance:
    return GlyphInstance(cfg, seed)

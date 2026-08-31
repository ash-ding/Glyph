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

    def go(ttype: str, b: int, root: bool = False) -> Expr:
        cands = [(o, s) for o, s in ops if SHAPE_RESULT[s] == ttype]
        # Stopping early is what makes depth a distribution rather than a
        # restatement of the budget. The root is exempt: a stopped root is a
        # bare literal, which is not an expression.
        stopped = (not root) and rng.random() < cfg.depth_stop_prob
        if b <= 0 or not cands or stopped:
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

    return go(want, budget, root=True)


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


def _draw_held_pairs(cfg: GlyphConfig, rng) -> set[tuple[str, str]]:
    """The (outer, inner) operator pairs kept out of the demo distribution.

    Two things this does that a uniform draw over the Cartesian product did
    not, and both were costing measurable damage.

    **Only pairs the grammar can build.** No shape has a VAL argument slot, so
    an operator whose shape returns VAL can never sit inside another one and
    the pair is a phantom. A Cartesian draw spent part of the quota on those,
    and `comp` requires a held-out pair to *appear*: 13 of 20 `pi_low` seeds
    could not be generated at all this way, and the seven that survived had all
    drawn the same pair, so they were a sample of nothing. Filtering also makes
    the held share exact -- 30-33% -- where it used to drift between 17% and
    50% depending on how many phantoms the draw happened to waste.

    **A fixed share of each pair type.** `comp` is sampled by rejection until a
    held pair appears, and that conditioning pulls its operator mix away from
    the one `atomic_ratio` defines. How far, and in which direction, depended
    on which pairs the draw took: measured across 20 `pi_mid` seeds, comp's
    table lookups per item swung between 0.83 and 3.12 while `iid` stayed
    between 1.80 and 2.74 -- sometimes far easier than iid, sometimes harder,
    with the median only happening to land above it. Allocating the quota
    proportionally across (outer shape, inner shape) classes fixes the *type*
    composition across seeds and leaves only which pair within a class to
    chance, which is the variation we want.

    Largest-remainder allocation, so the classes get proportional shares and
    the total is still exactly `len(realizable) // 3`. A `max(1, ...)` per class
    would look fairer and is not: `pi_mid` has 12 classes over 20 realizable
    pairs, so a floor of one each holds out 60% of the language.
    """
    ops = enabled_ops(cfg)
    shape = dict(ops)
    realizable = sorted((a, b) for a, _ in ops for b, _ in ops
                        if SHAPE_RESULT[shape[b]] == "LIST")
    by_kind: dict[tuple[str, str], list] = {}
    for pair in realizable:
        by_kind.setdefault((shape[pair[0]], shape[pair[1]]), []).append(pair)

    target = max(1, len(realizable) // 3)
    kinds = sorted(by_kind)
    exact = {k: len(by_kind[k]) * target / len(realizable) for k in kinds}
    quota = {k: int(exact[k]) for k in kinds}
    short = target - sum(quota.values())
    for k in sorted(kinds, key=lambda k: (-(exact[k] - quota[k]), k))[:short]:
        quota[k] += 1

    held: set[tuple[str, str]] = set()
    for k in kinds:
        group = by_kind[k]
        held |= {group[i] for i in rng.permutation(len(group))[:quota[k]]}
    return held


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

        self.held_pairs = _draw_held_pairs(cfg, rng)

        self.demos = self._make_demos(rng)
        self.test = self._make_test(rng)
        self.query_count = 0
        self.query_log = LookupLog()
        self._skel_interp: Interpreter | None = None
        self._tab_interp: Interpreter | None = None
        self._ceil_cache: dict[tuple, str] = {}
        self._pi: dict[str, float] | None = None

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
        """Cached: every run records this, and it is the phase diagram's axis.

        The preset name is provenance, not a condition. Measured pi ranges
        overlap between neighbouring presets -- seven of twenty `pi_mid` seeds
        fall inside `pi_high`'s range -- so an instance at pi = 0.70 cannot be
        attributed to a preset, and a figure keyed on the name would be
        grouping instances that differ more within a group than between.

        Which items this is measured on is still open (#3): `measure_pi`
        samples `test[:1500]`, which at full size is entirely `iid`. The whole
        dict is recorded rather than the ratio alone so that a later change of
        definition can be recomputed from `a_skel` and `a_tab` without
        regenerating anything.
        """
        if self._pi is None:
            from .measure import measure_pi
            self._pi = measure_pi(self)
        return self._pi

    def is_tail(self, t: TestItem) -> bool:
        """Did this run never buy a table entry this item needs?

        Asked of the item rather than of its index: `derive_tail` returns
        positions in `self.test`, which stop meaning anything the moment a run
        is scored on a subset -- and scoring on a paired subsample is where
        this is going.
        """
        return bool((t.needs_u - self.query_log.unary)
                    or (t.needs_b - self.query_log.binary))

    def ceilings(self, items: list[TestItem] | None = None) -> dict[str, dict]:
        """What perfect structural knowledge alone is worth, per split.

        Two oracles, both exact-match, both on the same items an arm was
        scored on:

          `skeleton`  true skeleton, identity tables -- every structural rule,
                      not one table entry
          `table`     trivial skeleton, true tables -- the mirror image

        The first is the line every arm score has to be read against. On
        pi_mid/1001 it is 0.222, and it equals the fraction of items needing no
        table lookup *exactly*, because a true skeleton with an identity table
        answers those and only those. A0' and A4 both scored 0.255 on 200
        items, which is one standard error above it -- so "the frontier
        extracted essentially no table knowledge" and "the frontier did well"
        are the same number until this line is drawn. A2 at 0.055 and A6 at
        0.035 are far *below* it, which says something different again: they
        have not learned the skeleton either.
        """
        items = self.test if items is None else items
        out: dict[str, dict] = {}
        for name, interp in (("skeleton", self._skeleton_only()),
                             ("table", self._table_only())):
            hits: dict[str, list[int]] = {}
            for t in items:
                ok = self._exact(interp, t)
                h, n = hits.setdefault(t.split, [0, 0])
                hits[t.split] = [h + ok, n + 1]
            out[name] = {k: h / n for k, (h, n) in hits.items()}
            out[name]["overall"] = (sum(h for h, _ in hits.values())
                                    / max(1, len(items)))
        return out

    def _skeleton_only(self) -> Interpreter:
        if self._skel_interp is None:
            self._skel_interp = Interpreter(self.cfg, self.skeleton, IdentityTables())
        return self._skel_interp

    def _table_only(self) -> Interpreter:
        if self._tab_interp is None:
            self._tab_interp = Interpreter(self.cfg, trivial_skeleton(self.cfg),
                                           self.tables)
        return self._tab_interp

    def _exact(self, interp: Interpreter, t: TestItem) -> bool:
        key = (id(interp), t.expr_src)
        got = self._ceil_cache.get(key)
        if got is None:
            try:
                got = self._render_out(interp.eval(parse(t.expr_src, self.cfg)))
            except Exception:
                got = ""
            self._ceil_cache[key] = got
        return got == t.answer_src

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

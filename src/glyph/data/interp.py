"""The reference interpreter P = skeleton o tables.

An expression is a tree, and a tree node is one of exactly three things: a
value (needs no rule), a structural operator (needs the skeleton), or an
atomic operator (needs the table).  There is no fourth kind, which is why two
rule sets are enough -- `Interpreter(skeleton, tables)` is not a simplification
of the interpreter, it *is* the interpreter.

That two-field shape is also what makes pi measurable: each crippled oracle is
just this class with one field swapped out.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import GlyphConfig
from .grammar import App, Expr, Lit, STRUCT_SHAPES, Val
from .semantics import (ARG_K, GEven, GFirstEqLast, GLenGt, StructSem, TDedup,
                        TDrop, TIdent, TMapAll, TMapSkip, TReverse, TRotate,
                        TSeq, TTake, Transform)


@dataclass
class LookupLog:
    """Which table entries an evaluation touched.  Feeds the derived `tail`
    split: a test item is in `tail` when the entries it needs were never
    bought by this run's agent."""
    unary: set[tuple[str, int]] = field(default_factory=set)
    binary: set[tuple[str, int, int]] = field(default_factory=set)

    def __ior__(self, other: "LookupLog") -> "LookupLog":
        self.unary |= other.unary
        self.binary |= other.binary
        return self

    def __len__(self) -> int:
        return len(self.unary) + len(self.binary)


class Interpreter:
    def __init__(self, cfg: GlyphConfig, skeleton: dict[str, StructSem], tables):
        self.cfg = cfg
        self.skeleton = skeleton
        self.tables = tables
        self._shape = dict(STRUCT_SHAPES)

    # -- public ---------------------------------------------------------
    def eval(self, e: Expr, log: LookupLog | None = None) -> int | list[int]:
        return self._eval(e, log)

    def eval_logged(self, e: Expr) -> tuple[int | list[int], LookupLog]:
        log = LookupLog()
        return self._eval(e, log), log

    # -- internals ------------------------------------------------------
    def _eval(self, e: Expr, log: LookupLog | None):
        if isinstance(e, Val):
            return e.idx
        if isinstance(e, Lit):
            return list(e.items)
        shape = self._shape[e.op]
        sem = self.skeleton[e.op]

        unary_name, k, sub = None, None, None
        for slot, a in zip(shape, e.args):
            if slot == "U":
                unary_name = a
            elif slot == "B":
                binary_name = a
            elif slot == "K":
                k = a
            else:
                sub = a
        lst = self._eval(sub, log)
        if isinstance(lst, int):
            lst = [lst]

        t = self._pick(sem, lst)
        out = self._apply(t, lst, unary_name, k, log)
        if sem.fold is not None:
            return self._fold(out, e.args[shape.index("B")], sem.fold, log)
        return out

    def _pick(self, sem: StructSem, lst: list[int]) -> Transform:
        if sem.guard is None:
            return sem.then_t
        g = sem.guard
        if isinstance(g, GEven):
            ok = len(lst) % 2 == 0
        elif isinstance(g, GLenGt):
            ok = len(lst) > g.k
        elif isinstance(g, GFirstEqLast):
            ok = lst[0] == lst[-1]
        else:
            raise TypeError(g)
        return sem.then_t if ok else sem.else_t

    def _apply(self, t: Transform, lst: list[int], unary: str | None,
               k: int | None, log: LookupLog | None) -> list[int]:
        if isinstance(t, TSeq):
            return self._apply(t.b, self._apply(t.a, lst, unary, k, log), unary, k, log)
        if isinstance(t, TIdent):
            return lst
        if isinstance(t, TReverse):
            return lst[::-1]
        if isinstance(t, TRotate):
            kk = (k if t.k == ARG_K else t.k) % max(1, len(lst))
            return lst[kk:] + lst[:kk]
        if isinstance(t, TDedup):
            seen, out = set(), []
            for x in lst:
                if x not in seen:
                    seen.add(x)
                    out.append(x)
            return out
        if isinstance(t, TTake):
            kk = max(1, min(k if t.k == ARG_K else t.k, len(lst)))
            return lst[:kk]
        if isinstance(t, TDrop):
            kk = k if t.k == ARG_K else t.k
            out = lst[kk:]
            return out if out else lst[-1:]
        if isinstance(t, TMapAll):
            return [self._u(unary, x, log) for x in lst]
        if isinstance(t, TMapSkip):
            j = t.j % len(lst)
            return [x if i == j else self._u(unary, x, log) for i, x in enumerate(lst)]
        raise TypeError(t)

    def _fold(self, lst: list[int], binary: str, direction: str,
              log: LookupLog | None) -> int:
        if len(lst) == 1:
            return lst[0]
        if direction == "L":
            acc = lst[0]
            for x in lst[1:]:
                acc = self._b(binary, acc, x, log)
            return acc
        acc = lst[-1]
        for x in reversed(lst[:-1]):
            acc = self._b(binary, x, acc, log)
        return acc

    def _u(self, name: str, i: int, log: LookupLog | None) -> int:
        if log is not None:
            log.unary.add((name, i))
        return self.tables.apply_unary(name, i)

    def _b(self, name: str, i: int, j: int, log: LookupLog | None) -> int:
        if log is not None:
            log.binary.add((name, i, j))
        return self.tables.apply_binary(name, i, j)

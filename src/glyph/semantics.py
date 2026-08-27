"""Structural operator semantics -- the "skeleton" half of the interpreter.

Each structural operator's meaning is *sampled from a combinator grammar*:

    structural ::= <transform>
                 | if <guard> then <transform> else <transform>

    transform  ::= ident | reverse | rotate k | dedup | take k | drop k
                 | map_all | map_skip j        (uses the caller's unary op)
                 | <transform> then <transform>

    guard      ::= len is even | len > k | first == last

Sampling from a *finite* grammar is what guarantees the skeleton is always
describable in finitely many words -- a property the tables deliberately do
not have.  Every transform is a combinator: it rearranges structure and never
interprets a value.  That is where the skeleton/table split comes from; it is
not a convention we imposed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import GlyphConfig
from .grammar import K_RANGE, enabled_ops

ARG_K = -1   # sentinel: "use the integer the caller passed in"


# ---------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class TIdent: pass
@dataclass(frozen=True)
class TReverse: pass
@dataclass(frozen=True)
class TRotate: k: int
@dataclass(frozen=True)
class TDedup: pass
@dataclass(frozen=True)
class TTake: k: int
@dataclass(frozen=True)
class TDrop: k: int
@dataclass(frozen=True)
class TMapAll: pass
@dataclass(frozen=True)
class TMapSkip: j: int
@dataclass(frozen=True)
class TSeq:
    a: "Transform"
    b: "Transform"

Transform = (TIdent | TReverse | TRotate | TDedup | TTake | TDrop
             | TMapAll | TMapSkip | TSeq)


@dataclass(frozen=True)
class GEven: pass
@dataclass(frozen=True)
class GLenGt: k: int
@dataclass(frozen=True)
class GFirstEqLast: pass

Guard = GEven | GLenGt | GFirstEqLast


@dataclass(frozen=True)
class StructSem:
    op: str
    shape: str
    guard: Guard | None
    then_t: Transform
    else_t: Transform | None
    fold: str | None            # "L" or "R" for LB shapes, else None


# ---------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------
_FREE = ("ident", "reverse", "rotate", "dedup", "take", "drop")


def _sample_transform(rng: np.random.Generator, cfg: GlyphConfig,
                      allow_atomic: bool, depth: int) -> Transform:
    if depth > 0 and rng.random() < 0.45:
        return TSeq(_sample_transform(rng, cfg, allow_atomic, depth - 1),
                    _sample_transform(rng, cfg, allow_atomic, depth - 1))
    kinds = list(_FREE)
    if allow_atomic:
        kinds += ["map_all", "map_skip"]
    kind = kinds[int(rng.integers(len(kinds)))]
    k = int(rng.integers(K_RANGE[0], K_RANGE[1] + 1))
    return {
        "ident": TIdent(),
        "reverse": TReverse(),
        "rotate": TRotate(k),
        "dedup": TDedup(),
        "take": TTake(k),
        "drop": TDrop(k),
        "map_all": TMapAll(),
        "map_skip": TMapSkip(int(rng.integers(0, 3))),
    }[kind]


def _uses_atomic(t: Transform) -> bool:
    if isinstance(t, (TMapAll, TMapSkip)):
        return True
    if isinstance(t, TSeq):
        return _uses_atomic(t.a) or _uses_atomic(t.b)
    return False


def _uses_k(t: Transform) -> bool:
    if isinstance(t, (TRotate, TTake, TDrop)):
        return t.k == ARG_K
    if isinstance(t, TSeq):
        return _uses_k(t.a) or _uses_k(t.b)
    return False


def _sample_guard(rng: np.random.Generator) -> Guard:
    kind = int(rng.integers(3))
    if kind == 0:
        return GEven()
    if kind == 1:
        return GLenGt(int(rng.integers(K_RANGE[0], K_RANGE[1] + 1)))
    return GFirstEqLast()


def sample_skeleton(cfg: GlyphConfig, rng: np.random.Generator) -> dict[str, StructSem]:
    """Draw one semantics per enabled structural operator."""
    out: dict[str, StructSem] = {}
    for op, shape in enabled_ops(cfg):
        allow_atomic = shape == "UL"
        d = cfg.max_transform_depth

        def draw() -> Transform:
            for _ in range(24):
                t = _sample_transform(rng, cfg, allow_atomic, d)
                if allow_atomic and not _uses_atomic(t):
                    continue          # a UL op must actually use its unary arg
                if not allow_atomic and _uses_atomic(t):
                    continue
                return t
            return TMapAll() if allow_atomic else TIdent()

        then_t = draw()
        if shape == "KL":
            # a KL op must actually use its integer argument
            k_node = [TTake(ARG_K), TDrop(ARG_K), TRotate(ARG_K)][int(rng.integers(3))]
            then_t = k_node if d == 0 else TSeq(k_node, then_t)

        guard = _sample_guard(rng) if rng.random() < cfg.guard_prob else None
        else_t = draw() if guard is not None else None
        if shape == "KL" and else_t is not None and not _uses_k(else_t):
            k_node = [TTake(ARG_K), TDrop(ARG_K)][int(rng.integers(2))]
            else_t = TSeq(k_node, else_t)

        fold = ("L" if rng.random() < 0.5 else "R") if shape == "LB" else None
        out[op] = StructSem(op, shape, guard, then_t, else_t, fold)
    return out


def trivial_skeleton(cfg: GlyphConfig) -> dict[str, StructSem]:
    """The degenerate skeleton baseline used by crippled oracle B.

    Textbook-default semantics, NOT "do nothing".  This matters: if the
    structural operators did nothing, the atomic operators would never be
    invoked and the baseline would silently ablate the table as well, folding
    L_table into L_skel and biasing pi.  s0 must still map, s1 must still fold.
    """
    out: dict[str, StructSem] = {}
    for op, shape in enabled_ops(cfg):
        if shape == "UL":
            t: Transform = TMapAll()                 # plain map, skips nothing
        elif shape == "KL":
            t = TTake(ARG_K)
        else:
            t = TIdent()
        fold = "L" if shape == "LB" else None        # always left-fold
        out[op] = StructSem(op, shape, None, t, None, fold)
    return out


def describe(sem: StructSem) -> str:
    """Human-readable rendering, for debugging and for the paper's appendix."""
    def d(t: Transform) -> str:
        if isinstance(t, TIdent): return "ident"
        if isinstance(t, TReverse): return "reverse"
        if isinstance(t, TRotate): return f"rotate({'k' if t.k == ARG_K else t.k})"
        if isinstance(t, TDedup): return "dedup"
        if isinstance(t, TTake): return f"take({'k' if t.k == ARG_K else t.k})"
        if isinstance(t, TDrop): return f"drop({'k' if t.k == ARG_K else t.k})"
        if isinstance(t, TMapAll): return "map_all(u)"
        if isinstance(t, TMapSkip): return f"map_skip(u, {t.j})"
        if isinstance(t, TSeq): return f"{d(t.a)} then {d(t.b)}"
        raise TypeError(t)

    def g(x: Guard) -> str:
        if isinstance(x, GEven): return "len is even"
        if isinstance(x, GLenGt): return f"len > {x.k}"
        return "first == last"

    body = d(sem.then_t) if sem.guard is None else \
        f"if {g(sem.guard)} then {d(sem.then_t)} else {d(sem.else_t)}"
    if sem.fold:
        body += f"  ;  fold-{sem.fold} with b"
    return f"{sem.op}: {body}"

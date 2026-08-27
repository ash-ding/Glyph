"""Atomic operator semantics -- the "table" half of the interpreter.

A table is never materialised.  It is a *function*: every call decomposes the
value into digits, embeds them, pushes the result through a frozen randomly
initialised MLP, and decodes back to the nearest symbol.

    unary:   V     -> V     ~4913 entries if you tabulated it
    binary:  V x V -> V     ~24 million entries if you tabulated it

Two properties have to hold simultaneously, and each comes from a different
part of the construction:

  * structured    -- digit-wise embeddings are shared across all values, so
                     values with similar digits get similar embeddings, and a
                     continuous MLP maps them to similar outputs.  This is the
                     only reason anything can extrapolate to unqueried entries.
  * indescribable -- the MLP weights are random, so writing the function down
                     means transcribing the weight matrices.

Binary operators (D2, settled as option C+B) are digit-wise-dominant with a
weak global coupling term.  Pure digit-wise (coupling = 0) would only have
n_digits * base**2 distinct "parts" -- learnable, but also small enough for an
agent to enumerate into a lookup table, which hands the win to the code arm.
The coupling term keeps it learnable but not tabulatable.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .config import GlyphConfig
from .grammar import digits


class FrozenMLP:
    """Two-layer tanh MLP, random weights, never trained."""

    def __init__(self, rng: np.random.Generator, d_in: int, d_out: int,
                 width: int, temp: float = 1.0):
        self.w1 = rng.normal(scale=1.0 / np.sqrt(d_in), size=(d_in, width))
        self.b1 = rng.normal(scale=0.1, size=width)
        self.w2 = rng.normal(scale=1.0 / np.sqrt(width), size=(width, d_out))
        self.b2 = rng.normal(scale=0.1, size=d_out)
        self.temp = temp

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(self.temp * (x @ self.w1 + self.b1))
        return np.tanh(self.temp * (h @ self.w2 + self.b2))

    @property
    def n_params(self) -> int:
        return self.w1.size + self.b1.size + self.w2.size + self.b2.size


@dataclass
class UnaryOp:
    name: str
    mlp: FrozenMLP


@dataclass
class BinaryOp:
    name: str
    per_digit: list[FrozenMLP]   # one per digit position: 2*d_digit -> d_digit
    mix: FrozenMLP               # global coupling: 2*d_total -> d_total
    alpha: float


class Tables:
    """The atomic-operator half of an instance."""

    def __init__(self, cfg: GlyphConfig, rng: np.random.Generator):
        self.cfg = cfg
        # digit embeddings: (n_digits, base, d_digit) -- only n_digits*base
        # vectors in total, shared by all n_values symbols.
        self.digit_emb = rng.normal(size=(cfg.n_digits, cfg.base, cfg.d_digit))
        self.all_emb = np.stack([self._embed(i) for i in range(cfg.n_values)])

        self.unary = [
            UnaryOp(f"u{k}", FrozenMLP(rng, cfg.d_total, cfg.d_total,
                                       cfg.mlp_width, cfg.mlp_temp))
            for k in range(cfg.n_unary)
        ]
        self.binary = [
            BinaryOp(
                f"b{k}",
                per_digit=[
                    FrozenMLP(rng, 2 * cfg.d_digit, cfg.d_digit,
                              max(8, cfg.mlp_width // 2), cfg.mlp_temp)
                    for _ in range(cfg.n_digits)
                ],
                mix=FrozenMLP(rng, 2 * cfg.d_total, cfg.d_total,
                              cfg.mlp_width, cfg.mlp_temp),
                alpha=cfg.binary_coupling,
            )
            for k in range(cfg.n_binary)
        ]
        self._u = {o.name: o for o in self.unary}
        self._b = {o.name: o for o in self.binary}
        self._cache_u: dict[tuple[str, int], int] = {}
        self._cache_b: dict[tuple[str, int, int], int] = {}

    # -- embedding -----------------------------------------------------
    def _embed(self, idx: int) -> np.ndarray:
        ds = digits(idx, self.cfg)
        return np.concatenate([self.digit_emb[k][d] for k, d in enumerate(ds)])

    def embed(self, idx: int) -> np.ndarray:
        return self.all_emb[idx]

    def _decode(self, y: np.ndarray) -> int:
        # nearest neighbour in embedding space -> back to a legal symbol
        return int(np.argmin(((self.all_emb - y) ** 2).sum(axis=1)))

    # -- application (memoised; the table is computed on demand) --------
    def apply_unary(self, name: str, i: int) -> int:
        key = (name, i)
        hit = self._cache_u.get(key)
        if hit is None:
            hit = self._decode(self._u[name].mlp(self.embed(i)))
            self._cache_u[key] = hit
        return hit

    def apply_binary(self, name: str, i: int, j: int) -> int:
        key = (name, i, j)
        hit = self._cache_b.get(key)
        if hit is not None:
            return hit
        op = self._b[name]
        di, dj = digits(i, self.cfg), digits(j, self.cfg)
        parts = [
            op.per_digit[k](np.concatenate([self.digit_emb[k][di[k]],
                                            self.digit_emb[k][dj[k]]]))
            for k in range(self.cfg.n_digits)
        ]
        y = np.concatenate(parts)
        if op.alpha:
            y = y + op.alpha * op.mix(np.concatenate([self.embed(i), self.embed(j)]))
        hit = self._decode(y)
        self._cache_b[key] = hit
        return hit

    # -- reporting -----------------------------------------------------
    def param_count(self) -> int:
        n = self.digit_emb.size
        n += sum(o.mlp.n_params for o in self.unary)
        for o in self.binary:
            n += sum(m.n_params for m in o.per_digit) + o.mix.n_params
        return n


class IdentityTables:
    """The degenerate table baseline used by crippled oracle A.

    `u(x) = x` and `b(x, y) = x`.  Crucially this leaves the *skeleton* fully
    exercised -- s0 still skips whichever element it skips, s1 still folds in
    whichever direction it folds -- so the loss it incurs is attributable to
    the table alone.
    """

    def apply_unary(self, name: str, i: int) -> int:
        return i

    def apply_binary(self, name: str, i: int, j: int) -> int:
        return i

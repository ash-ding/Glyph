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
    mlp: FrozenMLP                          # joint map over the whole embedding
    # Present only when cfg.unary_coupling is set: the digit-wise form, built
    # exactly like BinaryOp so the two halves are comparable.
    per_digit: list[FrozenMLP] | None = None
    mix: FrozenMLP | None = None
    alpha: float = 0.0


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

        self.unary = [self._make_unary(f"u{k}", cfg, rng)
                      for k in range(cfg.n_unary)]
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
        self._e_mu = self.all_emb.mean(axis=0)
        self._e_sd = self.all_emb.std(axis=0) + 1e-8
        self._y_mu = np.zeros(cfg.d_total)
        self._y_sd = np.ones(cfg.d_total)
        self._assign: dict[str, np.ndarray] = {}
        if cfg.decode == "whiten":
            self._fit_output_scale()
        elif cfg.decode == "assign":
            self._build_assignments()
        self._cache_u: dict[tuple[str, int], int] = {}
        self._cache_b: dict[tuple[str, int, int], int] = {}

    @staticmethod
    def _make_unary(name: str, cfg: GlyphConfig, rng) -> UnaryOp:
        joint = FrozenMLP(rng, cfg.d_total, cfg.d_total, cfg.mlp_width, cfg.mlp_temp)
        if cfg.unary_coupling is None:
            return UnaryOp(name, joint)
        return UnaryOp(
            name, joint,
            per_digit=[FrozenMLP(rng, cfg.d_digit, cfg.d_digit,
                                 max(8, cfg.mlp_width // 2), cfg.mlp_temp)
                       for _ in range(cfg.n_digits)],
            mix=FrozenMLP(rng, cfg.d_total, cfg.d_total, cfg.mlp_width, cfg.mlp_temp),
            alpha=cfg.unary_coupling,
        )

    def _raw(self, op, i: int) -> np.ndarray:
        if op.per_digit is None:
            return op.mlp(self.embed(i))
        di = digits(i, self.cfg)
        y = np.concatenate([op.per_digit[k](self.digit_emb[k][di[k]])
                            for k in range(self.cfg.n_digits)])
        return y + op.alpha * op.mix(self.embed(i)) if op.alpha else y

    def _fit_output_scale(self) -> None:
        """Per-dimension mean and spread of what the MLPs actually emit."""
        sample = [self._raw(self.unary[0], i)
                  for i in range(0, self.cfg.n_values,
                                 max(1, self.cfg.n_values // 800))]
        arr = np.stack(sample)
        self._y_mu = arr.mean(axis=0)
        self._y_sd = arr.std(axis=0) + 1e-8

    def _build_assignments(self) -> None:
        """Greedy nearest-unclaimed matching: every value gets its own image.

        Unary only. Binary would need 24 million outputs materialised, which
        is the thing the table exists not to do.
        """
        n = self.cfg.n_values
        for op in self.unary:
            ys = np.stack([self._raw(op, i) for i in range(n)])
            d = ((ys[:, None, :] - self.all_emb[None, :, :]) ** 2).sum(axis=2)
            out = np.full(n, -1, dtype=np.int64)
            taken = np.zeros(n, dtype=bool)
            # Settle the confident ones first: an input whose best match is
            # far better than its second is the one that should get it.
            order = np.argsort(np.partition(d, 1, axis=1)[:, 1] - d.min(axis=1))[::-1]
            for i in order:
                row = np.where(taken, np.inf, d[i])
                j = int(np.argmin(row))
                out[i], taken[j] = j, True
            self._assign[op.name] = out
    def _embed(self, idx: int) -> np.ndarray:
        ds = digits(idx, self.cfg)
        return np.concatenate([self.digit_emb[k][d] for k, d in enumerate(ds)])

    def embed(self, idx: int) -> np.ndarray:
        return self.all_emb[idx]

    def _decode(self, y: np.ndarray) -> int:
        mode = self.cfg.decode
        if mode == "nearest":
            # The original. Collapses: see GlyphConfig.decode.
            return int(np.argmin(((self.all_emb - y) ** 2).sum(axis=1)))

        if mode == "whiten":
            # Put the MLP output on the embeddings' own scale before asking
            # which one is nearest. Removes the part of the collapse caused by
            # tanh squashing outputs into a smaller region than the
            # embeddings occupy; does nothing about hubness itself.
            z = (y - self._y_mu) / self._y_sd * self._e_sd + self._e_mu
            return int(np.argmin(((self.all_emb - z) ** 2).sum(axis=1)))

        if mode == "per_digit":
            # Decode each digit against its own 17 candidates instead of the
            # whole space against 4913. Hubness is a high-dimensional,
            # many-candidate effect and mostly disappears at this size, and it
            # is the decode the digit-structure argument implies: the digits
            # are the parts, so the parts are what should be rounded.
            ds = []
            for k in range(self.cfg.n_digits):
                seg = y[k * self.cfg.d_digit:(k + 1) * self.cfg.d_digit]
                bank = self.digit_emb[k]
                ds.append(int(np.argmin(((bank - seg) ** 2).sum(axis=1))))
            return int(sum(d * self.cfg.base ** k for k, d in enumerate(ds)))

        if mode == "assign":
            # A true bijection, built once per unary operator by giving each
            # input its nearest still-unclaimed embedding. Exact by
            # construction, and impossible for binary: it needs every output
            # materialised, and the binary table has 24 million of them.
            raise RuntimeError("assign decodes through the precomputed map")
        raise ValueError(f"unknown decode {mode!r}")

    # -- application (memoised; the table is computed on demand) --------
    def apply_unary(self, name: str, i: int) -> int:
        key = (name, i)
        hit = self._cache_u.get(key)
        if hit is not None:
            return hit
        op = self._u[name]
        if self.cfg.decode == "assign":
            hit = int(self._assign[name][i])
        else:
            hit = self._decode(self._raw(op, i))
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
        if self.cfg.decode == "assign":
            # Deliberate: a bijection over 24 million pairs cannot be built
            # without materialising the table. Binary keeps `nearest`, which
            # means `assign` is not a whole-benchmark answer -- only a bound
            # on what a bijective unary decode would look like.
            hit = int(np.argmin(((self.all_emb - y) ** 2).sum(axis=1)))
        else:
            hit = self._decode(y)
        self._cache_b[key] = hit
        return hit

    # -- reporting -----------------------------------------------------
    def param_count(self) -> int:
        n = self.digit_emb.size
        for o in self.unary:
            n += (o.mlp.n_params if o.per_digit is None
                  else sum(m.n_params for m in o.per_digit) + o.mix.n_params)
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

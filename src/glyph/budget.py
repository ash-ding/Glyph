"""The budget ledger -- the single metered entry point.

Every operation that consumes compute goes through `Ledger.charge`.  Bypass it
once and the crossover figure is worthless, so the rule is enforced by making
`charge` the only way to advance the clock and by keeping the cost model in
one place.

Everything is converted to one currency.  The primary unit is H100-seconds;
`usd` is reported alongside so the paper can show the conclusion survives both
conversions.
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass, field
from typing import Any


class BudgetExhausted(RuntimeError):
    pass


@dataclass(frozen=True)
class CostModel:
    """Conversion rates.  Frontier token prices follow published API rates so
    that prompt-cache savings in the ledger are real savings."""
    usd_per_h100_second: float = 0.00111        # ~$4/hr
    usd_per_in_token: float = 5.0e-6            # $5 / MTok
    usd_per_out_token: float = 25.0e-6          # $25 / MTok
    cache_read_mult: float = 0.1
    cache_write_mult: float = 1.25
    batch_mult: float = 0.5                     # Batch API is half price
    usd_per_oracle_query: float = 1.0e-5        # sandbox/CPU cost of one query

    def to_h100s(self, usd: float) -> float:
        return usd / self.usd_per_h100_second


@dataclass
class Record:
    kind: str
    n: float
    usd: float
    h100s: float
    meta: dict[str, Any] = field(default_factory=dict)


class Ledger:
    def __init__(self, total_h100s: float | None = None,
                 costs: CostModel | None = None):
        self.total = total_h100s
        self.costs = costs or CostModel()
        self.records: list[Record] = []
        self.spent_h100s = 0.0
        self.spent_usd = 0.0
        self.sealed = False

    # -- core ----------------------------------------------------------
    def charge(self, kind: str, n: float = 1.0, usd: float | None = None,
               **meta) -> Record:
        if usd is None:
            usd = self._price(kind, n, meta)
        rec = Record(kind, n, usd, self.costs.to_h100s(usd), meta)
        self.records.append(rec)
        self.spent_usd += usd
        self.spent_h100s += rec.h100s
        if self.total is not None and self.spent_h100s > self.total:
            raise BudgetExhausted(
                f"{self.spent_h100s:.1f} / {self.total:.1f} H100-s after {kind}")
        return rec

    def _price(self, kind: str, n: float, meta: dict) -> float:
        c = self.costs
        if kind == "oracle_query":
            return n * c.usd_per_oracle_query
        if kind == "frontier_in":
            mult = (c.cache_read_mult if meta.get("cached") else
                    c.cache_write_mult if meta.get("cache_write") else 1.0)
            return n * c.usd_per_in_token * mult * (c.batch_mult if meta.get("batch") else 1.0)
        if kind == "frontier_out":
            return n * c.usd_per_out_token * (c.batch_mult if meta.get("batch") else 1.0)
        if kind == "gpu_second":
            return n * c.usd_per_h100_second
        raise KeyError(f"unpriced cost kind {kind!r}")

    # -- helpers -------------------------------------------------------
    @property
    def remaining(self) -> float:
        return float("inf") if self.total is None else self.total - self.spent_h100s

    def gpu_timer(self, label: str = "gpu"):
        return _GpuTimer(self, label)

    def breakdown(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for r in self.records:
            out[r.kind] = out.get(r.kind, 0.0) + r.h100s
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "spent_h100s": round(self.spent_h100s, 4),
            "spent_usd": round(self.spent_usd, 6),
            "total_h100s": self.total,
            "n_records": len(self.records),
            "by_kind": {k: round(v, 4) for k, v in self.breakdown().items()},
        }


class _GpuTimer:
    """Wall-clock GPU time is *measured*, never modelled from a formula.
    (Prefix caching makes any analytic model of the context arm's per-query
    cost wrong; measuring sidesteps the question entirely.)"""

    def __init__(self, ledger: Ledger, label: str):
        self.ledger, self.label = ledger, label

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ledger.charge("gpu_second", time.perf_counter() - self.t0,
                           label=self.label)
        return False


def meters(kind: str, n: float = 1.0):
    """Decorator form.  The wrapped callable must accept a `ledger=` kwarg."""
    def deco(fn):
        @functools.wraps(fn)
        def wrap(*a, ledger: Ledger | None = None, **kw):
            if ledger is not None:
                ledger.charge(kind, n, fn=fn.__qualname__)
            return fn(*a, **kw)
        return wrap
    return deco

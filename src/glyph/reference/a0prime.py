"""A0' -- the information-saturated frontier, on a paired subset (#20).

Answers #20: does in-context learning extract table knowledge at all? The
evidence is deliberately more generous than any arm could buy: probes that
isolate one table entry at a time (the way the A6 agent itself chose to read
the table), mixed with in-domain expressions drawn the way test items are, so
the skeleton is inferable from the same distribution it will be tested on.

Ported from `scripts/run_a0p.py`. Two structural changes from that script:

  - `buy()` is split out as `buy_evidence`, and its stopping condition changes
    from a fixed fact count to the number of *distinct unary table entries*
    revealed -- the quantity the evidence is actually meant to saturate.
  - the answering step (`a0p_saturated.run`, which called a real frontier
    model) is replaced by an injectable `answer_fn` seam. The real frontier
    call is a later [API] step; this module never makes one. A CPU test
    injects a fake/oracle `answer_fn` to exercise the parse+score wiring.

`src/glyph/reference/` may import `glyph.data`, `glyph.seal` and
`glyph.reference.subset` -- not `glyph.arms` (deleted).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..data.grammar import parse_value, render, render_list, render_value
from ..data.instance import GlyphInstance, TestItem, _sample_constrained
from ..seal import headroom as _headroom

AnswerFn = Callable[[str, list[str]], list[str]]


def buy_evidence(inst: GlyphInstance, entries_target: int, *,
                 probe_frac: float = 0.7,
                 seed: int = 0) -> tuple[list[tuple[str, str]],
                                         tuple[frozenset, frozenset]]:
    """Buy probes + in-domain facts through `inst.query` until the number of
    distinct unary entries revealed reaches ~`entries_target`.

    Two phases, both reusing `_sample_constrained`/`render` verbatim from
    `run_a0p.buy`:

      probes      shallow, one operator (`depth_stop_prob=0.9`, budget 1), so
                  each answer isolates a single table entry -- the way the A6
                  agent itself chose to read the table
      in-domain   drawn at `cfg.demo_max_depth`, the way test items are, so
                  the skeleton is inferable from the same distribution

    `probe_frac` splits `entries_target` into an intermediate probe-phase
    checkpoint (`entries_target * probe_frac`) and a final target
    (`entries_target`) reached (if at all) by the in-domain phase. Each phase
    caps its tries at 60x its own share of the target, same shape as
    `run_a0p.buy`'s `n_probe * 60` / `n_domain * 60`, so a saturated or
    unreachable target cannot spin forever.

    Returns `(facts, revealed)`: `facts` is the list of `(src, answer)` pairs
    bought (in order); `revealed` is
    `(frozenset(inst.query_log.unary), frozenset(inst.query_log.binary))`
    captured after buying.
    """
    cfg = inst.cfg
    rng = np.random.default_rng(seed)
    out: list[tuple[str, str]] = []

    n_probe_target = max(1, round(entries_target * probe_frac))
    n_domain_target = max(1, entries_target - n_probe_target)

    # probes: shallow, one operator, so each answer isolates table entries
    tries = 0
    while (len(inst.query_log.unary) < n_probe_target
          and tries < n_probe_target * 60):
        tries += 1
        e = _sample_constrained(rng, cfg.with_(depth_stop_prob=0.9),
                                1, forbid=set(), require=None, min_depth=1)
        if e is None:
            continue
        src = render(e, cfg)
        try:
            out.append((src, inst.query(src)))
        except Exception:
            pass

    # in-domain: drawn the way test items are
    tries = 0
    while (len(inst.query_log.unary) < entries_target
          and tries < n_domain_target * 60):
        tries += 1
        e = _sample_constrained(rng, cfg, cfg.demo_max_depth,
                                forbid=set(), require=None, min_depth=1)
        if e is None:
            continue
        src = render(e, cfg)
        try:
            out.append((src, inst.query(src)))
        except Exception:
            pass

    revealed = (frozenset(inst.query_log.unary), frozenset(inst.query_log.binary))
    return out, revealed


def retrieval_split(items: list[TestItem],
                    revealed: tuple[frozenset, frozenset]) -> dict:
    """Split `items` by coverage against `revealed = (u, b)`.

    An item is **covered** iff every table entry it needs was revealed
    (`t.needs_u <= u and t.needs_b <= b`), else **uncovered**.
    """
    u, b = revealed
    covered = [t for t in items if t.needs_u <= u and t.needs_b <= b]
    uncovered = [t for t in items if not (t.needs_u <= u and t.needs_b <= b)]
    return {
        "covered_n": len(covered),
        "covered": covered,
        "uncovered_n": len(uncovered),
        "uncovered": uncovered,
    }


def _parse_and_render(raw: str, cfg) -> str | None:
    """Normalize a raw answer string to its canonical rendered form, or None
    if it does not parse as a legal answer.

    Deliberately does not use the expression parser or `grammar.check`: an
    answer is just a value or a list of values, not an expression, and an
    output list is allowed to be longer or shorter than `list_len_range`.
    """
    t = raw.strip()
    try:
        if t.startswith("["):
            if not t.endswith("]"):
                return None
            inner = t[1:-1].strip()
            if not inner:
                return None
            vals = [parse_value(tok.strip(), cfg) for tok in inner.split(",")]
            return render_list(vals, cfg)
        return render_value(parse_value(t, cfg), cfg)
    except Exception:
        return None


def run_a0prime(inst: GlyphInstance, items: list[TestItem], evidence: str,
                answer_fn: AnswerFn, *, chunk: int = 50) -> dict:
    """Score `answer_fn` on `items`, given `evidence`, in batches of `chunk`.

    For each chunk of `items`, calls
    `answer_fn(evidence, [item.expr_src for item in chunk]) -> list[str]`
    (one answer string per item), parses each answer to a value/list via the
    grammar, and exact-matches it against the item's true answer
    (`t.answer_src`, itself already the canonical rendered form).

    `answer_fn` is the seam: the real one (a later step) sends `evidence` and
    the questions to the frontier; this function never calls a real one --
    tests inject a fake/oracle.

    Returns `{overall, by_split, tail, headroom}`, same shapes as
    `glyph.reference.subset.ceilings_on`: `tail` restricts the score to items
    this run's `inst` never bought a table entry for (`None` when there are
    none in `items`); `headroom` reads `overall` against
    `inst.ceilings(items)["skeleton"]["overall"]`.
    """
    cfg = inst.cfg
    answers: list[str] = []
    for s in range(0, len(items), chunk):
        batch = items[s:s + chunk]
        out = answer_fn(evidence, [t.expr_src for t in batch])
        out = list(out)
        if len(out) != len(batch):
            raise ValueError(
                f"answer_fn returned {len(out)} answers for {len(batch)} items")
        answers.extend(out)

    hits: dict[str, list[int]] = {}
    tail_hit = tail_n = 0
    for t, raw in zip(items, answers):
        ok = _parse_and_render(raw, cfg) == t.answer_src
        h, n = hits.setdefault(t.split, [0, 0])
        hits[t.split] = [h + ok, n + 1]
        if inst.is_tail(t):
            tail_n += 1
            tail_hit += ok

    overall = sum(h for h, _ in hits.values()) / max(1, len(items))
    by_split = {k: h / n for k, (h, n) in hits.items()}
    tail = (tail_hit / tail_n) if tail_n else None

    skeleton_overall = inst.ceilings(items)["skeleton"]["overall"]

    return {
        "overall": overall,
        "by_split": by_split,
        "tail": tail,
        "headroom": _headroom(overall, skeleton_overall),
    }

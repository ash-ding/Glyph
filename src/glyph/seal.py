"""Scoring primitives shared across protocol implementations.

Sealing was the v1 idea: an agent prepares an artifact, seals it, and the
artifact is scored with no further access to the oracle. v2 replaced that
prepare/seal split with a single continuous session (see `glyph.v2`), but
the scoring primitives below are protocol-agnostic and both versions of the
harness score a run against the same reference lines.
"""

from __future__ import annotations

from .data.instance import TestItem


def headroom(score: float, ceiling: float) -> float | None:
    """Where a score sits between the skeleton-only ceiling and perfect.

    A raw score is not comparable across splits or instances, because the
    ceilings are not. On pi_mid the true skeleton with an identity table scores
    0.285 on `iid`, 0.407 on `comp` and 0.144 on `depth` -- so the same raw
    number means three different things, and `comp`, which is supposed to be
    the harder compositional split, is the one where knowing no table entry
    gets you furthest. Across instances the spread is wider still: the overall
    skeleton ceiling ranges 0.168-0.743 over 20 pi_mid seeds, which is more
    than the gap between any two arms.

    0.0 means "no better than knowing every structural rule and no table
    entry"; 1.0 means perfect. Negative is meaningful and not clipped -- an
    arm below the ceiling has not learned the skeleton either, which is a
    different finding from having learned it and no table.

    None when the ceiling is already 1.0: there is nothing left to measure.
    """
    if ceiling >= 1.0:
        return None
    return (score - ceiling) / (1.0 - ceiling)


def score_answers(items: list[TestItem], answers: list[str]) -> tuple[float, dict[str, float]]:
    """Exact match, overall and per split.

    Arms are scored by exact match, deliberately.  The graded credit in
    `measure.py` exists so that pi does not saturate; using it here as well
    would quietly change what the crossover figure is measuring.
    """
    assert len(items) == len(answers)
    hits = {}
    for t, a in zip(items, answers):
        ok = a.strip() == t.answer_src.strip()
        h, n = hits.setdefault(t.split, [0, 0])
        hits[t.split] = [h + ok, n + 1]
    by_split = {k: h / n for k, (h, n) in hits.items()}
    total = sum(h for h, _ in hits.values()) / max(1, len(items))
    return total, by_split

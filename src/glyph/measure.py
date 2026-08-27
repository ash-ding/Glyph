"""pi -- how much of an instance's difficulty sits in the skeleton.

    a_skel = accuracy(true skeleton + identity table)     # only the table is missing
    a_tab  = accuracy(trivial skeleton + true table)      # only the skeleton is missing

    L_table = 1 - a_skel        loss caused by not knowing the table
    L_skel  = 1 - a_tab         loss caused by not knowing the skeleton

    pi = L_skel / (L_skel + L_table)          # the SKELETON's share

The numerator is L_skel, not L_table: pi is *programmability*, so it measures
the skeleton's share.  Flip it and the whole phase diagram runs backwards --
and the failure is silent, because the curves still plot.

pi is not an absolute quantity.  It is a ratio relative to a chosen pair of
baselines, so those two baselines are fixed once and never changed; changing
them invalidates every previously measured pi.
"""

from __future__ import annotations

from .interp import Interpreter
from .grammar import parse, render_list, render_value
from .semantics import trivial_skeleton
from .tables import IdentityTables


def _digit_match(a: str, b: str) -> float:
    """Fraction of digit positions two rendered values agree on."""
    if a == b:
        return 1.0
    da, db = a.split("_")[1:], b.split("_")[1:]
    if len(da) != len(db) or not da:
        return 0.0
    return sum(x == y for x, y in zip(da, db)) / len(da)


def _graded(pred: str, gold: str) -> float:
    """Partial credit.  Exact match saturates both losses near 1.0 -- any single
    wrong operator or lookup ruins the whole answer -- which drives every pi
    toward 0.5 regardless of where the difficulty actually sits.  Grading by
    digit and by list position desaturates the measurement.

    This affects the pi *measurement only*; the arms are still scored by
    exact match (D4).
    """
    if pred == gold:
        return 1.0
    pl, gl = pred.startswith("["), gold.startswith("[")
    if pl != gl:
        return 0.0
    if not gl:
        return _digit_match(pred, gold)
    pa = [x.strip() for x in pred[1:-1].split(",")] if pred[1:-1].strip() else []
    ga = [x.strip() for x in gold[1:-1].split(",")] if gold[1:-1].strip() else []
    n = max(len(pa), len(ga))
    if n == 0:
        return 1.0
    return sum(_digit_match(x, y) for x, y in zip(pa, ga)) / n


def _score(interp, inst, items, graded: bool = True) -> float:
    tot = 0.0
    for t in items:
        try:
            out = interp.eval(parse(t.expr_src, inst.cfg))
        except Exception:
            continue
        src = (render_value(out, inst.cfg) if isinstance(out, int)
               else render_list(out, inst.cfg))
        tot += _graded(src, t.answer_src) if graded else float(src == t.answer_src)
    return tot / max(1, len(items))


def measure_pi(inst, sample: int = 1500) -> dict[str, float]:
    items = inst.test[:sample]
    full = _score(inst.P, inst, items)
    a_skel = _score(Interpreter(inst.cfg, inst.skeleton, IdentityTables()), inst, items)
    a_tab = _score(Interpreter(inst.cfg, trivial_skeleton(inst.cfg), inst.tables),
                   inst, items)
    l_table, l_skel = 1.0 - a_skel, 1.0 - a_tab
    denom = l_table + l_skel
    return {
        "full": full,
        "a_skel": a_skel,
        "a_tab": a_tab,
        "L_table": l_table,
        "L_skel": l_skel,
        "pi": (l_skel / denom) if denom > 0 else float("nan"),
        "n": len(items),
    }

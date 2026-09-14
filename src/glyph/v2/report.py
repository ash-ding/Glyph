"""Assembles the final ScoreReport dict for a completed protocol v2 run.

One pure function: score the committed held-out-test answer file, attach
ceilings and headroom (headroom is the primary read of a run -- see
`glyph.seal.headroom`), the instance block, the validation trajectory, and
the covariates the analysis needs. Reads defensively from `session` because
several fields on it are populated by the harness/tools (submit, final_answer)
rather than by Session itself -- see Task 10/12.
"""

from __future__ import annotations

from glyph.seal import headroom as _headroom
from glyph.v2.answers import score_file


def _zero_score(items, cfg) -> dict:
    from glyph.data import depth, parse

    by_split: dict[str, list] = {}
    by_depth: dict[str, list] = {}
    for t in items:
        by_split.setdefault(t.split, []).append(0)
        d = str(depth(parse(t.expr_src, cfg)))
        by_depth.setdefault(d, []).append(0)
    return {
        "overall": 0.0,
        "by_split": {k: 0.0 for k in by_split},
        "by_depth": {k: 0.0 for k in by_depth},
    }


def build_report(session, committed_path, id_of) -> dict:
    inst = session.inst
    cfg = inst.cfg
    items = inst.test

    if committed_path is None:
        score = _zero_score(items, cfg)
    else:
        score = score_file(committed_path, items, cfg, id_of)

    tail_items = [t for t in items if inst.is_tail(t)]
    if committed_path is None or not tail_items:
        tail_score = 0.0
    else:
        tail_scored = score_file(committed_path, tail_items, cfg, id_of)
        tail_score = tail_scored["overall"]

    ceiling = inst.ceilings(items)
    skel_ceiling = ceiling["skeleton"]

    hr: dict[str, float | None] = {}
    for split in ("overall", "iid", "comp", "depth"):
        if split == "overall":
            s = score["overall"]
        else:
            s = score["by_split"].get(split, 0.0)
        c = skel_ceiling.get(split)
        hr[split] = None if c is None else _headroom(s, c)
    if tail_items:
        tail_ceiling = inst.ceilings(tail_items)["skeleton"]["overall"]
    else:
        tail_ceiling = skel_ceiling.get("overall", 0.0)
    hr["tail"] = _headroom(tail_score, tail_ceiling)

    instance = {
        "seed": inst.seed,
        "preset": getattr(session, "preset", None),
        "pi": inst.measured_pi(),
        "n_structural": cfg.n_structural,
        "atomic_ratio": cfg.atomic_ratio,
        "uses_binary_tables": any(t.needs_b for t in inst.test),
    }

    history = [e for e in session.events if e.get("kind") == "validation_submit"]
    overalls = [e["overall"] for e in history]
    best = max(overalls) if overalls else None
    last = overalls[-1] if overalls else None
    gap = None
    if last is not None:
        test_iid = score["by_split"].get("iid")
        if test_iid is not None:
            gap = last - test_iid
    validation = {
        "history": history,
        "best": best,
        "last": last,
        "gap_last_vs_test_iid": gap,
    }

    turns_practice = getattr(session, "turns_practice", 0)
    turns_final = getattr(
        session, "turns_final",
        session.turns if session.phase == "final" else 0,
    )
    covariates = {
        "q_used": session.q_used,
        "submissions": session.submissions,
        "turns_practice": turns_practice,
        "turns_final": turns_final,
        "finished_practice_early": getattr(session, "finished_practice_early", False),
        "val_lookup_solvable": inst.lookup_solvable(inst.val, inst.query_log),
        "test_covered": inst.lookup_solvable(inst.test, inst.query_log),
        "final_from_student": getattr(session, "final_from_student", None),
        "final_commit": getattr(
            session, "final_commit", "agent" if committed_path else "none"
        ),
        "run_status": getattr(session, "run_status", "completed"),
    }

    return {
        "overall": score["overall"],
        "by_split": score["by_split"],
        "by_depth": score["by_depth"],
        "tail": tail_score,
        "ceiling": ceiling,
        "headroom": hr,
        "instance": instance,
        "validation": validation,
        "covariates": covariates,
        "ledger": session.ledger.summary(),
    }

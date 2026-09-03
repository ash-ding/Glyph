"""The sealed artifact, and the one place a run is scored.

Sealing is the moment the agent stops existing.  Whatever it produced is
frozen, its size is recorded, and the test set is answered with no further
access to the oracle, the teacher, or the ground truth.

Three container shapes, one scoring entry point.  That is the whole reason
the arms are comparable: A2 hands over a prompt, A4 hands over a program, A6
hands over an adapter, and every one of them is then run through the same
loop against the same 10^4 items and reduced to a single number.  A scoring
path that branched per arm would let a difference in the harness masquerade
as a difference in the container.

Sizes are recorded because they are costs.  A 200k-token context is not a
free artifact -- it is re-paid on every test query, and an arm that does not
have to declare its size can look free while being the most expensive one
there is.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from .budget import Ledger
from .instance import GlyphInstance, TestItem

Entry = Literal["program", "model"]


@dataclass(frozen=True)
class SealedArtifact:
    """What survives the prepare phase.

    `entry` says which one actually answers at test time.  A run may carry a
    context *and* an adapter -- A7 is allowed to combine them -- but exactly
    one thing drives the loop.
    """
    arm: str
    entry: Entry
    context: str | None = None          # A2: the prompt prefix
    program: str | None = None          # A4: source of the solver
    adapter_path: str | None = None     # A6: LoRA (or full) checkpoint
    base_model: str | None = None
    notes: dict = field(default_factory=dict)

    def sizes(self, tokens_in=None) -> dict[str, int]:
        """Context tokens, program bytes, adapter bytes.

        `tokens_in` is the student's tokenizer when one is available; without
        it the context is measured in characters and flagged as such, because
        a size silently in the wrong unit is worse than a missing one.
        """
        out: dict[str, int] = {}
        if self.context is not None:
            if tokens_in is not None:
                out["context_tokens"] = len(tokens_in(self.context))
            else:
                out["context_chars"] = len(self.context)
        if self.program is not None:
            out["program_bytes"] = len(self.program.encode())
        if self.adapter_path is not None:
            p = Path(self.adapter_path)
            out["adapter_bytes"] = sum(f.stat().st_size
                                       for f in p.rglob("*") if f.is_file()) if p.exists() else 0
        return out

    def digest(self) -> str:
        blob = json.dumps({k: v for k, v in asdict(self).items() if k != "notes"},
                          sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class ScoreReport:
    arm: str
    overall: float
    by_split: dict[str, float]
    tail: float | None
    n: int
    sizes: dict[str, int]
    ledger: dict
    digest: str
    # What perfect structural knowledge alone scores on these same items, and
    # where the arm sits between that and perfect. See `headroom`.
    ceiling: dict = field(default_factory=dict)
    headroom: dict = field(default_factory=dict)
    # What instance this was: its seed and its measured pi. The preset name is
    # added by the arm runner and is provenance only -- the phase diagram's
    # axis is `instance["pi"]["pi"]`, never the name. See GlyphInstance.
    instance: dict = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


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


def answer_with(artifact: SealedArtifact, base_model: str | None,
                ledger: Ledger, exprs: list[str]) -> list[str]:
    """Answer expressions with whatever an artifact holds.

    One path, used at dev time by the tool layer and at test time by the arm
    runners. Two implementations would let the number the agent steers on drift
    from the number it is finally graded by -- different answer cleaning,
    different caching -- and that kind of drift is invisible from either side.
    Sharing the path makes it impossible rather than unlikely.

    A program runs in the sandbox; anything else is the student carrying
    whatever prefix and adapter the artifact has, which covers a bare base
    model, a prompt, a checkpoint, or both at once.
    """
    if not exprs:
        return []
    if artifact.entry == "program":
        from .sandbox import run_solver
        res = run_solver(artifact.program or "", exprs, ledger=ledger, timeout=900)
        return res.answers if res.ok else [""] * len(exprs)

    from .train.infer import Student
    student = Student(base_model or artifact.base_model or "Qwen/Qwen3-1.7B",
                      adapter_path=artifact.adapter_path,
                      context=artifact.context)
    try:
        return student.answer(exprs, ledger=ledger).answers
    finally:
        student.close()


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


def evaluate(inst: GlyphInstance, artifact: SealedArtifact, ledger: Ledger, *,
             answer_fn, items: list[TestItem] | None = None) -> ScoreReport:
    """Score a sealed artifact.  `answer_fn(exprs) -> answers` is the only
    thing that differs between arms, and it is supplied by the arm runner.

    `tail` is derived here rather than fixed at generation time, because it
    depends on what this particular agent bought: the items whose table
    entries it never queried.  It doubles as a read on how good its query
    strategy was.
    """
    items = items if items is not None else inst.test_set()
    # Everything from here on is the test phase. It is billed to the same
    # ledger -- the deployment cost of each container is a result, not an
    # overhead -- but the prepare budget no longer gates it.
    with ledger.sealed_mode():
        answers = answer_fn([t.expr_src for t in items])
    overall, by_split = score_answers(items, answers)

    # `tail` comes from this run's own query log: the items whose table
    # entries this agent never bought. It cannot be fixed at generation time
    # because it depends on what the agent chose to ask, and it doubles as a
    # read on how good that choice was.
    # Asked of each item rather than by position: `derive_tail` returns indices
    # into `inst.test`, which mean nothing once a run is scored on a subset,
    # and scoring on a paired subsample is where this is going.
    tail = None
    picked = [(t, a) for t, a in zip(items, answers) if inst.is_tail(t)]
    if picked:
        tail = sum(a.strip() == t.answer_src.strip() for t, a in picked) / len(picked)

    # The ceilings are computed on the items that were actually scored, not on
    # the full test set: a ceiling from a different sample is not the line this
    # score should be read against.
    ceiling = inst.ceilings(items)
    skel = ceiling["skeleton"]
    head = {k: headroom(v, skel[k]) for k, v in by_split.items() if k in skel}
    head["overall"] = headroom(overall, skel["overall"])
    # `tail` gets its own line. It is a different subset from any split -- the
    # items whose table entries this agent never bought -- so no split's
    # ceiling applies to it, and without one it was the only number in the
    # report with nothing to read it against.
    #
    # Its ceiling is near zero, which is what makes `tail` the cleanest read on
    # table knowledge: almost nothing there is free.
    #
    # Near, not exactly, and the gap says something. `needs_u`/`needs_b` record
    # entries *touched*, not entries the answer *depends on* -- `eval_logged`
    # logs every table call, including calls a later transform discards. On
    # pi_mid seed 1002, `s0` is `map_skip(j=1) -> dedup -> rotate(1)` and the
    # outer `s3` keeps only the element the map skipped: two lookups happen and
    # neither matters. So `tail` is over-inclusive, and this ceiling measures
    # exactly how much.
    if picked:
        tc = inst.ceilings([t for t, _ in picked])
        ceiling["skeleton"]["tail"] = tc["skeleton"]["overall"]
        ceiling["table"]["tail"] = tc["table"]["overall"]
        head["tail"] = headroom(tail, tc["skeleton"]["overall"])

    return ScoreReport(
        arm=artifact.arm, overall=overall, by_split=by_split, tail=tail,
        n=len(items), sizes=artifact.sizes(), ledger=ledger.summary(),
        digest=artifact.digest(), ceiling=ceiling, headroom=head,
        instance={"seed": inst.seed, "pi": inst.measured_pi(),
                  "n_structural": inst.cfg.n_structural,
                  "atomic_ratio": inst.cfg.atomic_ratio},
    )

"""A0' -- the information-saturated frontier.

Everything the agent purchased is handed to the teacher with no limit on
context or thinking, and the teacher answers the test set directly.

This arm decides how strong H1's claim is allowed to be. If A0' wins, the
weights arm's advantage is **economics** -- the frontier could learn it all,
it just cannot afford to re-read the evidence on every query -- and the paper
has to say so plainly. If A0' loses even when saturated and unconstrained,
the advantage is **learning-algorithmic**: some structure gradient descent
reaches and in-context learning does not. That is a much stronger claim, and
it is not ours to assert without this arm.

It also tests the benchmark's own foundations. The pi=0 end assumes the
tables are learnable-but-not-describable; an A0' that does well from partial
observations would mean in-context learning extrapolates better than the
design supposes.
"""

from __future__ import annotations

import re

from ..seal import ScoreReport, SealedArtifact, evaluate
from ..vertex import client, request_kwargs
from .base import RunConfig, prepare
from ..agent.prompts import SATURATED

CHUNK = 25


def run(rc: RunConfig, *, purchased: list[tuple[str, str]],
        max_tokens: int = 64000, items=None) -> ScoreReport:
    """`purchased` is the evidence A0' reads; it does not buy its own.

    The original design fed it another arm's query log, so that the information
    was held constant and only the reader varied. Under the capacity framing
    that pairing is no longer the point: A0' is now the control for "can
    in-context learning extract this table at all", and the sharper form of that
    question hands it **more** evidence than any arm could buy. If it still
    cannot, the claim does not depend on how much a particular agent happened to
    purchase. Either shape is passed in here; the caller decides which.

    `items` scores a subset rather than the whole test set. A0' costs about 75
    H100-s per item against A6's 0.003, so the full 10^4 is not reachable for it
    at any sensible budget -- and by #9 a paired subset resolves a difference
    better than independent full runs would anyway.
    """
    p = prepare(rc)
    c = client()
    evidence = "\n".join(f"  {e} = {a}" for e, a in purchased)
    items = p.inst.test_set() if items is None else items

    def answer(exprs):
        out: list[str] = []
        for s in range(0, len(exprs), CHUNK):
            chunk = exprs[s:s + CHUNK]
            numbered = "\n".join(f"  {i + 1}. {e}" for i, e in enumerate(chunk))
            prompt = SATURATED.format(spec=p.inst.syntax_spec(),
                                      evidence=evidence, n=len(chunk),
                                      items=numbered)
            kwargs = request_kwargs(rc.teacher, max_tokens=max_tokens)
            with c.messages.stream(messages=[{"role": "user", "content": prompt}],
                                   **kwargs) as stream:
                msg = stream.get_final_message()
            p.ledger.charge("frontier_in", int(msg.usage.input_tokens or 0))
            p.ledger.charge("frontier_out", int(msg.usage.output_tokens or 0))
            text = "".join(b.text for b in msg.content if b.type == "text")
            got = [""] * len(chunk)
            for line in text.splitlines():
                m = re.match(r"\s*(\d+)\s*[:.]\s*(.+?)\s*$", line)
                if m and 0 <= int(m.group(1)) - 1 < len(chunk):
                    got[int(m.group(1)) - 1] = m.group(2).strip()
            p.trace.emit("a0p_chunk", start=s, n=len(chunk),
                         answered=sum(1 for g in got if g),
                         truncated=msg.stop_reason == "max_tokens")
            out.extend(got)
        return out

    artifact = SealedArtifact(arm="a0p", entry="model",
                              notes={"purchased": len(purchased),
                                     "n_scored": len(items),
                                     "note": "frontier answers directly"})
    report = evaluate(p.inst, artifact, p.ledger, answer_fn=answer, items=items)
    report.instance["preset"] = rc.preset      # provenance, not a condition
    (p.work_dir / "report.json").write_text(report.to_json())
    p.trace.emit("report", arm="a0p", overall=report.overall,
                 by_split=report.by_split, tail=report.tail,
                 ceiling=report.ceiling, headroom=report.headroom,
                 instance=report.instance)
    p.trace.close()
    return report

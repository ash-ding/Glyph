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
        max_tokens: int = 64000) -> ScoreReport:
    """`purchased` is another arm's query log -- A0' does not buy its own.

    Reusing the same evidence is the point: it holds the information constant
    and varies only who is reading it.
    """
    p = prepare(rc)
    c = client()
    evidence = "\n".join(f"  {e} = {a}" for e, a in purchased)
    items = p.inst.test_set()

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
                                     "note": "frontier answers directly"})
    report = evaluate(p.inst, artifact, p.ledger, answer_fn=answer, items=items)
    (p.work_dir / "report.json").write_text(report.to_json())
    p.trace.emit("report", arm="a0p", overall=report.overall,
                 by_split=report.by_split, tail=report.tail)
    p.trace.close()
    return report

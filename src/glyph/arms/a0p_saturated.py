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

# Items per call. Bigger is both cheaper and safer: the evidence block is
# re-sent on every call, so halving the number of calls halves the input spend
# -- and self-check #4 established that reasoning cost is per *call* rather than
# per item, so a larger chunk amortises the thinking rather than compounding it.
CHUNK = 25


class Truncated(RuntimeError):
    """Raised rather than scored.

    Unanswered items are filled with the empty string, which scores as wrong.
    So a run that truncates reports a *low number* rather than an error, and
    the low number is indistinguishable from the frontier genuinely failing --
    which is exactly how self-check #4 was misread once already. A0' is the
    control for "can in-context learning do this at all", so a quiet
    under-report there is the most expensive kind of bug available here.
    """


def run(rc: RunConfig, *, purchased: list[tuple[str, str]],
        max_tokens: int = 64000, items=None, chunk: int = CHUNK,
        min_coverage: float = 0.95) -> ScoreReport:
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
        n_trunc = 0
        for s in range(0, len(exprs), chunk):
            block = exprs[s:s + chunk]
            numbered = "\n".join(f"  {i + 1}. {e}" for i, e in enumerate(block))
            prompt = SATURATED.format(spec=p.inst.syntax_spec(),
                                      evidence=evidence, n=len(block),
                                      items=numbered)
            kwargs = request_kwargs(rc.teacher, max_tokens=max_tokens)
            with c.messages.stream(messages=[{"role": "user", "content": prompt}],
                                   **kwargs) as stream:
                msg = stream.get_final_message()
            p.ledger.charge("frontier_in", int(msg.usage.input_tokens or 0))
            p.ledger.charge("frontier_out", int(msg.usage.output_tokens or 0))
            text = "".join(b.text for b in msg.content if b.type == "text")
            got = [""] * len(block)
            for line in text.splitlines():
                m = re.match(r"\s*(\d+)\s*[:.]\s*(.+?)\s*$", line)
                if m and 0 <= int(m.group(1)) - 1 < len(block):
                    got[int(m.group(1)) - 1] = m.group(2).strip()
            trunc = msg.stop_reason == "max_tokens"
            n_trunc += trunc
            p.trace.emit("a0p_chunk", start=s, n=len(block),
                         answered=sum(1 for g in got if g),
                         out_tokens=int(msg.usage.output_tokens or 0),
                         truncated=trunc)
            print(f"    chunk {s:4d}  answered {sum(1 for g in got if g):3d}/{len(block)}"
                  f"  out {int(msg.usage.output_tokens or 0):6d}"
                  f"{'  TRUNCATED' if trunc else ''}", flush=True)
            out.extend(got)

        cov = sum(1 for g in out if g) / max(1, len(out))
        if cov < min_coverage:
            raise Truncated(
                f"only {cov:.1%} of items were answered ({n_trunc} truncated "
                f"call(s) at max_tokens={max_tokens}, chunk={chunk}). Scoring "
                f"this would report the frontier failing when the harness is "
                f"what failed -- raise --max-tokens or raise --chunk.")
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

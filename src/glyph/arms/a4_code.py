"""A4 -- capability into code.

The agent buys what it can and writes a Python solver.  At test time the
student does not participate at all: the program answers on its own, at
essentially zero marginal cost.

Hard-coding the queried entries into a lookup table is a legitimate artifact
and is deliberately allowed.  It scores well on `iid` and collapses on
`tail`, which is the precise weakness of this container and half the
explanation of the phase diagram -- so it is something to record, not to
forbid.
"""

from __future__ import annotations

from ..agent import Container, run_agent
from ..seal import ScoreReport, answer_with
from .base import RunConfig, finish, prepare


def run(rc: RunConfig) -> ScoreReport:
    p = prepare(rc)
    box = run_agent(p.inst, p.ledger, p.trace, p.cache, arm="a4",
                    allowed={Container.CODE}, work_dir=p.work_dir,
                    base_model=rc.base_model, model=rc.teacher,
                    max_turns=rc.max_turns)

    # `answer_with` is the same path the agent's own `evaluate` uses during
    # prepare. A solver that will not run scores zero rather than crashing the
    # run: a broken artifact is a result, not a missing data point.
    def answer(exprs):
        return answer_with(box.sealed, rc.base_model, p.ledger, exprs)

    return finish(p, rc, box.sealed, answer,
                  dev=box._split()[1], purchased=box.queried)

"""A2 -- capability into context.

The agent buys what it can and writes a prompt prefix.  At test time a base
student carries that prefix in front of every one of the 10^4 queries, which
is exactly where this container's cost lives: cheap to produce, re-paid
forever.

Prefix caching stays on.  Turning it off would triple A2's measured cost and
hand the weights arm a crossover it did not earn.
"""

from __future__ import annotations

from ..agent import Container, run_agent
from ..seal import ScoreReport
from .base import RunConfig, finish, prepare


def run(rc: RunConfig) -> ScoreReport:
    p = prepare(rc)
    box = run_agent(p.inst, p.ledger, p.trace, p.cache, arm="a2",
                    allowed={Container.CONTEXT}, work_dir=p.work_dir,
                    base_model=rc.base_model, model=rc.teacher,
                    max_turns=rc.max_turns)

    from ..train.infer import Student
    student = Student(rc.base_model, context=box.sealed.context)

    def answer(exprs):
        return student.answer(exprs, ledger=p.ledger).answers

    return finish(p, rc, box.sealed, answer)

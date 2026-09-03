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
from ..seal import ScoreReport, answer_with
from .base import RunConfig, finish, prepare


def run(rc: RunConfig) -> ScoreReport:
    p = prepare(rc)
    box = run_agent(p.inst, p.ledger, p.trace, p.cache, arm="a2",
                    allowed={Container.CONTEXT}, work_dir=p.work_dir,
                    base_model=rc.base_model, model=rc.teacher,
                    max_turns=rc.max_turns)

    # `answer_with` is the same path the agent's own `evaluate` uses during
    # prepare, so the number it steered on and the number it is graded by are
    # produced by one implementation.
    #
    # box.sealed is guaranteed non-None: the harness seals if the agent does
    # not. It may however be empty -- training that all failed leaves a base
    # student with nothing attached, which scores badly and is exactly the
    # comparable data point the grid needs.
    def answer(exprs):
        return answer_with(box.sealed, rc.base_model, p.ledger, exprs)

    return finish(p, rc, box.sealed, answer)

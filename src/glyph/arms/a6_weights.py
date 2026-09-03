"""A6 -- capability into weights.

The agent buys what it can, declares what it is teaching, builds a training
set from what it bought, and fine-tunes the student.  At test time the model
answers with no prompt and no program: the cost was paid once, up front.

`declare_target` is mandatory here.  Without it "what did the agent choose"
would be an inference from prose rather than a field in the trace, and Fig. 4
would be our reading of its behaviour instead of a measurement of it.
"""

from __future__ import annotations

from ..agent import Container, run_agent
from ..seal import ScoreReport, answer_with
from .base import RunConfig, finish, prepare


def run(rc: RunConfig) -> ScoreReport:
    p = prepare(rc)
    box = run_agent(p.inst, p.ledger, p.trace, p.cache, arm="a6",
                    allowed={Container.WEIGHTS}, work_dir=p.work_dir,
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
